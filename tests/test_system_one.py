"""The System One client: request shape, error mapping, retries, circuit breaker."""

from __future__ import annotations

import asyncio

import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
    mock_aiohttp_client,
)

from custom_components.typesafe_conversation.const import (
    API_TIMEOUT,
    PROBE_MAX_OPTIONS,
    TYPESAFE_API_URL,
)
from custom_components.typesafe_conversation.system_one import (
    ChoiceAnswer,
    ServerProfile,
    SystemOneAuthError,
    SystemOneClient,
    SystemOneRequestError,
    SystemOneUnavailableError,
)

LOCAL = "http://ollama.invalid:11434"

OK = {
    "model": "jev-1.13.0",
    "answers": {
        "category": {
            "type": "choice",
            "choice": "command",
            "probabilities": {"command": 0.9, "query": 0.08, "information": 0.02},
            "confidence": 0.85,
        },
        "compound": {"type": "noul", "noul": 0.04},
        "frustration": {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "Calm", "1": "Cross"},
            "probabilities": {"0": 0.2, "1": 0.8},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 6482, "output_tokens": 210},
}


@pytest.fixture(name="mocker")
def mocker_fixture():
    with mock_aiohttp_client() as mocker:
        yield mocker


@pytest.fixture(name="client")
async def client_fixture(mocker: AiohttpClientMocker):
    session = mocker.create_session(asyncio.get_running_loop())
    return SystemOneClient(session, "sk-test", "jev-latest")


async def test_request_shape_and_typed_answers(client, mocker):
    mocker.post(TYPESAFE_API_URL, json=OK)
    response = await client.async_ask({"request": {"text": "hi"}}, {"category": {}})

    _method, _url, body, headers = mocker.mock_calls[0]
    assert body["model"] == "jev-latest"
    assert body["state"] == {"request": {"text": "hi"}}
    assert headers["Authorization"] == "Bearer sk-test"

    assert response.model == "jev-1.13.0"
    assert response.input_tokens == 6482
    assert response.choice("category").choice == "command"
    assert response.noul("compound") == 0.04
    assert response.score("frustration").score == 1.05
    # Wrong-typed access returns None rather than raising.
    assert response.choice("compound") is None


def test_margin_catches_a_confident_looking_tie():
    """Confidence and margin fail differently, which is why we check both."""
    tied = ChoiceAnswer("a", {"a": 0.45, "b": 0.44, "c": 0.11}, 0.62)
    assert tied.confidence > 0.6
    assert tied.margin < 0.05, "top two are effectively tied"


async def test_auth_failure_is_not_retried(client, mocker):
    mocker.post(TYPESAFE_API_URL, status=401, text="nope")
    with pytest.raises(SystemOneAuthError):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 1


async def test_validation_failure_is_not_retried(client, mocker):
    """A 422 is our bug, not a transient one - retrying just wastes time."""
    mocker.post(TYPESAFE_API_URL, status=422, text='{"detail":"questions.x.criteria"}')
    with pytest.raises(SystemOneRequestError, match="criteria"):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 1


async def test_rate_limit_is_retried_then_gives_up(client, mocker):
    mocker.post(TYPESAFE_API_URL, status=429, text="slow down")
    with pytest.raises(SystemOneUnavailableError):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 3, "three attempts, then the fallback ladder"


async def test_circuit_opens_after_repeated_failure(client, mocker):
    """Three failed requests, then stop trying for a while.

    A dead API must not add six seconds of timeout to every utterance.
    """
    mocker.post(TYPESAFE_API_URL, status=500, text="boom")
    for _ in range(3):
        with pytest.raises(SystemOneUnavailableError):
            await client.async_ask({}, {})
    assert client.circuit_open

    before = len(mocker.mock_calls)
    with pytest.raises(SystemOneUnavailableError, match="circuit breaker"):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == before, "no request while the circuit is open"


async def test_success_resets_the_failure_count(client, mocker):
    mocker.post(TYPESAFE_API_URL, status=500, text="boom")
    with pytest.raises(SystemOneUnavailableError):
        await client.async_ask({}, {})
    mocker.clear_requests()
    mocker.post(TYPESAFE_API_URL, json=OK)
    await client.async_ask({}, {})
    assert not client.circuit_open


# --- any System One server ----------------------------------------------------


async def test_a_custom_server_is_used_for_every_endpoint(mocker):
    session = mocker.create_session(asyncio.get_running_loop())
    client = SystemOneClient(session, None, "nimble", base_url=LOCAL + "/")
    mocker.get(LOCAL + "/v1/models", json={"data": [{"id": "nimble:latest"}]})
    mocker.post(LOCAL + "/v1/systemone", json=OK)

    assert await client.async_validate() == ["nimble"], "`:latest` is normalised"
    await client.async_ask({}, {"category": {}})

    urls = [str(url) for _m, url, _b, _h in mocker.mock_calls]
    assert urls == [LOCAL + "/v1/models", LOCAL + "/v1/systemone"]


async def test_no_key_means_no_authorization_header(mocker):
    """A local server usually takes no key; sending an empty one is noise."""
    session = mocker.create_session(asyncio.get_running_loop())
    mocker.post(LOCAL + "/v1/systemone", json=OK)
    await SystemOneClient(session, "", "nimble", base_url=LOCAL).async_ask({}, {})
    _m, _u, _b, headers = mocker.mock_calls[0]
    assert "Authorization" not in headers


async def test_both_model_list_formats_are_read(client, mocker):
    mocker.get(
        "https://api.typesafe.ai/v1/models",
        json={"models": [{"name": "jev-latest"}, {"name": "jev-1.13.0"}]},
    )
    assert await client.async_validate() == ["jev-latest", "jev-1.13.0"]


async def test_a_rejected_request_is_not_retried_and_keeps_the_reason(mocker):
    """A context overflow is deterministic, and its message says what to change."""
    session = mocker.create_session(asyncio.get_running_loop())
    reason = "prompt 0 has 5373 tokens; expected 1-2050 (input is never truncated)"
    mocker.post(LOCAL + "/v1/systemone", status=400, json={"error": reason})
    client = SystemOneClient(session, None, "tev1:4b", base_url=LOCAL)
    with pytest.raises(SystemOneRequestError, match="5373 tokens"):
        await client.async_ask({}, {})
    assert len(mocker.mock_calls) == 1


async def test_keep_alive_is_never_sent(mocker):
    """How long a model stays loaded is the server owner's call, not ours."""
    session = mocker.create_session(asyncio.get_running_loop())
    mocker.post(LOCAL + "/v1/systemone", json=OK)
    await SystemOneClient(session, None, "nimble", base_url=LOCAL).async_ask({}, {})
    _m, _u, body, _h = mocker.mock_calls[0]
    assert "keep_alive" not in body


def test_an_entry_without_a_profile_behaves_as_before():
    """Entries from before discovery all point at TypeSafe."""
    legacy = ServerProfile.from_dict(None)
    assert legacy.max_options == 255
    assert legacy.timeout == API_TIMEOUT


# --- endpoint discovery -------------------------------------------------------


def _fake_server(cap: int, *, num_ctx: int | None = 8194):
    """A System One server that rejects Choices over ``cap`` options."""

    async def systemone(method, url, body):
        options = len(body["questions"]["probe"]["criteria"])
        if options > cap:
            return AiohttpClientMockResponse(
                method, url, status=400, json={"error": "too many candidates"}
            )
        return AiohttpClientMockResponse(method, url, json=OK)

    async def show(method, url, body):
        if num_ctx is None:
            return AiohttpClientMockResponse(method, url, status=404)
        return AiohttpClientMockResponse(
            method,
            url,
            json={
                "capabilities": ["decision", "completion"],
                "parameters": f"num_ctx                        {num_ctx}",
            },
        )

    return systemone, show


async def test_probe_finds_a_small_cap_by_binary_search(mocker):
    session = mocker.create_session(asyncio.get_running_loop())
    systemone, show = _fake_server(26)
    mocker.post(LOCAL + "/v1/systemone", side_effect=systemone)
    mocker.post(LOCAL + "/api/show", side_effect=show)

    profile = await SystemOneClient(
        session, None, "nimble", base_url=LOCAL
    ).async_probe()

    assert profile.max_options == 26
    assert profile.num_ctx == 8194
    assert profile.matches(LOCAL, "nimble:latest")
    assert profile.timeout >= API_TIMEOUT


async def test_probe_stops_at_the_ceiling_when_everything_fits(mocker):
    """TypeSafe accepts 255: one probe at the ceiling, no search."""
    session = mocker.create_session(asyncio.get_running_loop())
    systemone, show = _fake_server(PROBE_MAX_OPTIONS, num_ctx=None)
    mocker.post(TYPESAFE_API_URL, side_effect=systemone)
    mocker.post("https://api.typesafe.ai/api/show", side_effect=show)

    profile = await SystemOneClient(session, "sk", "jev-latest").async_probe()

    assert profile.max_options == PROBE_MAX_OPTIONS
    assert profile.num_ctx is None, "a server without /api/show loses nothing"
    probes = [b for _m, u, b, _h in mocker.mock_calls if str(u) == TYPESAFE_API_URL]
    assert len(probes) == 2, "model check, then the ceiling: no search needed"


async def test_probe_reports_why_a_model_cannot_be_used(mocker):
    session = mocker.create_session(asyncio.get_running_loop())
    reason = 'model "qwen3:4b" is not supported by System One'
    mocker.post(LOCAL + "/v1/systemone", status=400, json={"error": reason})
    client = SystemOneClient(session, None, "qwen3:4b", base_url=LOCAL)
    with pytest.raises(SystemOneRequestError, match="not supported by System One"):
        await client.async_probe()


async def test_decision_models_are_filtered_where_the_server_says(mocker):
    session = mocker.create_session(asyncio.get_running_loop())

    async def show(method, url, body):
        caps = ["decision"] if body["model"] == "nimble" else ["completion"]
        return AiohttpClientMockResponse(method, url, json={"capabilities": caps})

    mocker.post(LOCAL + "/api/show", side_effect=show)
    client = SystemOneClient(session, None, "nimble", base_url=LOCAL)
    assert await client.async_decision_models(["nimble", "qwen3"]) == ["nimble"]


async def test_without_api_show_the_model_list_is_kept_whole(client, mocker):
    mocker.post("https://api.typesafe.ai/api/show", status=404)
    names = ["jev-latest", "jev-1.13.0"]
    assert await client.async_decision_models(names) == names


# --- timing a real-sized request ----------------------------------------------


def test_the_timeout_covers_a_cold_load_and_a_typical_request():
    """The model may have been unloaded since the last request, so both count."""
    profile = ServerProfile(cold_load_s=7.0).with_typical(46.0)
    assert profile.typical_s == 46.0
    assert profile.timeout == 80, "1.5 x (7 + 46), rounded up"


def test_a_fast_server_keeps_the_floor_timeout():
    """TypeSafe answers in a fraction of a second; 6 s stays the minimum."""
    assert ServerProfile(cold_load_s=0.14).with_typical(0.3).timeout == API_TIMEOUT


async def test_timing_a_request_reports_how_long_it_took(mocker):
    session = mocker.create_session(asyncio.get_running_loop())
    mocker.post(LOCAL + "/v1/systemone", json=OK)
    client = SystemOneClient(session, None, "nimble", base_url=LOCAL)
    elapsed = await client.async_time_request({"home": {}}, {"category": {}})
    assert elapsed >= 0
    _m, _u, body, _h = mocker.mock_calls[0]
    assert body["state"] == {"home": {}}, "the real-sized state, not a probe"


async def test_timing_a_request_the_server_rejects_says_why(mocker):
    """A home too large for the model's context fails here, at setup."""
    session = mocker.create_session(asyncio.get_running_loop())
    reason = "prompt 0 has 5373 tokens; expected 1-2050"
    mocker.post(LOCAL + "/v1/systemone", status=400, json={"error": reason})
    client = SystemOneClient(session, None, "tev1:4b", base_url=LOCAL)
    with pytest.raises(SystemOneRequestError, match="5373 tokens"):
        await client.async_time_request({}, {})
