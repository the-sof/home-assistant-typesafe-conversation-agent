"""The two LLM operations, against mocked HTTP."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    mock_aiohttp_client,
)

from custom_components.typesafe_conversation.llm_backend import (
    LLMBackendError,
    OllamaBackend,
    OpenAICompatBackend,
)


@pytest.fixture(name="mocker")
def mocker_fixture():
    """Home Assistant's own aiohttp mock.

    aioresponses does not track the aiohttp version HA pins, so use the mock
    that ships with the test harness instead.
    """
    with mock_aiohttp_client() as mocker:
        yield mocker


@pytest.fixture(name="session")
async def session_fixture(mocker: AiohttpClientMocker):
    import asyncio

    return mocker.create_session(asyncio.get_running_loop())


def _ollama(text: str) -> dict:
    return {"message": {"role": "assistant", "content": text}}


def _openai(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '["turn off the lights", "lock the door"]',
            ["turn off the lights", "lock the door"],
        ),
        ('```json\n["a", "b"]\n```', ["a", "b"]),
        ('Sure! Output: ["a", "b"]', ["a", "b"]),
        ('["only one"]', ["only one"]),
    ],
)
async def test_split_parses_the_shapes_models_actually_return(
    session, mocker, raw, expected
):
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", json=_ollama(raw))
    assert await backend.split_compound("anything") == expected


@pytest.mark.parametrize(
    "raw", ["I cannot do that", "", "{}", "[1, 2, 3]", '["a", 2]', '["a", " "]']
)
async def test_a_useless_split_runs_nothing(session, mocker, raw):
    """Running the sentence as one command would act on a guess."""
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", json=_ollama(raw))
    assert await backend.split_compound("turn on the lamp") == []


async def test_split_with_the_llm_down_runs_nothing(session, mocker):
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", status=500, text="boom")
    assert await backend.split_compound("a and b") == []


async def test_too_many_parts_are_refused_not_truncated(session, mocker):
    """Keeping the first six would silently drop the rest of the request."""
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    raw = "[" + ",".join(f'"cmd {i}"' for i in range(7)) + "]"
    mocker.post("http://ollama:11434/api/chat", json=_ollama(raw))
    assert await backend.split_compound("x") == []


async def test_ollama_request_shape(session, mocker):
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", json=_ollama("[]"))
    await backend.split_compound("x")
    body = mocker.mock_calls[0][2]
    assert body["stream"] is False, "streaming would only add latency here"
    assert "keep_alive" not in body, "how long it stays loaded is the server's call"
    assert body["options"]["temperature"] == 0.0


async def test_openrouter_headers_and_shape(session, mocker):
    backend = OpenAICompatBackend(
        session,
        "https://openrouter.ai/api",
        "deepseek/deepseek-v4-flash",
        api_key="sk-test",
        referer="https://example.invalid",
        title="Test",
    )
    mocker.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=_openai("The Oakland A's."),
    )
    answer = await backend.answer_freeform(
        "who won the 1989 world series",
        [],
        home_state="",
        local_time="10:00",
        weekday="Monday",
        speaker_area="Kitchen",
    )
    _method, _url, body, headers = mocker.mock_calls[0]
    assert answer == "The Oakland A's."
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["HTTP-Referer"] == "https://example.invalid"
    assert headers["X-Title"] == "Test"
    assert body["stream"] is False


async def test_answer_raises_when_the_backend_fails(session, mocker):
    """Unlike split, a failed answer has no safe default - it must surface."""
    backend = OpenAICompatBackend(session, "https://x.invalid", "m", api_key="k")
    mocker.post("https://x.invalid/v1/chat/completions", status=502, text="bad")
    with pytest.raises(LLMBackendError):
        await backend.answer_freeform(
            "hi",
            [],
            home_state="",
            local_time="10:00",
            weekday="Monday",
            speaker_area=None,
        )


async def test_home_state_reaches_the_prompt(session, mocker):
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", json=_ollama("ok"))
    await backend.answer_freeform(
        "is the garage shut?",
        [("earlier", "reply")],
        home_state="Garage Door (cover, Garage): closed",
        local_time="22:40",
        weekday="Sunday",
        speaker_area="Kitchen",
    )
    messages = mocker.mock_calls[0][2]["messages"]
    assert "Garage Door (cover, Garage): closed" in messages[0]["content"]
    assert "Kitchen" in messages[0]["content"]
    # The model is told plainly that it cannot act.
    assert "cannot control any device" in messages[0]["content"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


# --- metrics and logging -----------------------------------------------------
# Ollama and an OpenAI-compatible endpoint report different things, so each
# backend normalizes into shared key names. Everything below asserts on those
# shared names, never on a provider's raw field.

OLLAMA_METRIC_BODY = {
    "message": {"role": "assistant", "content": "A joke."},
    "load_duration": 4_210_000_000,
    "prompt_eval_count": 612,
    "prompt_eval_duration": 830_000_000,
    "eval_count": 41,
    "eval_duration": 2_100_000_000,
    "total_duration": 7_150_000_000,
}


async def test_ollama_metrics_are_normalized(session, mocker):
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", json=OLLAMA_METRIC_BODY)
    text, metrics = await backend._chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.0,
        timeout=5,
    )
    assert text == "A joke."
    assert metrics["prompt_tokens"] == 612
    assert metrics["completion_tokens"] == 41
    # Nanoseconds become seconds.
    assert metrics["load_s"] == pytest.approx(4.21)
    assert metrics["eval_s"] == pytest.approx(2.10)
    # Measured by us, not the provider, so it is always present.
    assert metrics["elapsed_s"] >= 0


async def test_a_response_with_no_metrics_does_not_raise(session, mocker):
    """Not every server fills those fields in; the text still has to arrive."""
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post(
        "http://ollama:11434/api/chat",
        json={"message": {"role": "assistant", "content": "still fine"}},
    )
    text, metrics = await backend._chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.0,
        timeout=5,
    )
    assert text == "still fine"
    assert set(metrics) == {"elapsed_s"}


async def test_openai_compatible_reports_tokens_but_no_timing(session, mocker):
    """There is no local model to load, so there is no cold-load figure."""
    backend = OpenAICompatBackend(
        session, "https://openrouter.ai/api", "some/model", api_key="k"
    )
    mocker.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 7},
        },
    )
    _text, metrics = await backend._chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.0,
        timeout=5,
    )
    assert metrics["prompt_tokens"] == 100
    assert metrics["completion_tokens"] == 7
    assert "load_s" not in metrics and "eval_s" not in metrics
    assert metrics["elapsed_s"] >= 0


async def test_the_home_catalog_is_truncated_out_of_the_log(session, mocker):
    """The system prompt carries entity names, areas and states.

    home-assistant.log is what people paste into issue reports, so only the
    first PROMPT_LOG_CHARS may appear. The reply is short and logged whole.
    """
    import logging

    from custom_components.typesafe_conversation.const import PROMPT_LOG_CHARS

    # pytest-homeassistant-custom-component wraps `caplog`, and requesting it
    # here trips a recursive-fixture error, so capture directly.
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger("custom_components.typesafe_conversation")
    handler = _Capture()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)

    secret = "light.bedroom_secret_fixture_entity"
    home_state = f"{secret}: on\n" * 200
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    mocker.post("http://ollama:11434/api/chat", json=OLLAMA_METRIC_BODY)

    try:
        await backend.answer_freeform(
            "tell me a joke",
            [],
            home_state=home_state,
            local_time="10:00",
            weekday="Monday",
            speaker_area="Kitchen",
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    logged = "\n".join(records)
    assert "A joke." in logged, "the reply must be logged in full"
    assert "4.21s" in logged and "cold model load" in logged
    assert "612 tok" in logged
    # The catalog appears at most once, inside the truncated head.
    assert logged.count(secret) <= 1
    assert len(home_state) > PROMPT_LOG_CHARS


async def test_keep_alive_is_sent_only_when_opted_in(session, mocker):
    """Off by default - the server's idle setting decides - on only on request."""
    mocker.post("http://ollama:11434/api/chat", json=_ollama("[]"))
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    await backend.split_compound("x")
    backend.keep_loaded = True
    await backend.split_compound("y")
    first, second = (call[2] for call in mocker.mock_calls)
    assert "keep_alive" not in first
    assert second["keep_alive"] == "30m"


async def test_warm_up_does_nothing_unless_opted_in(session, mocker):
    mocker.post("http://ollama:11434/api/chat", json=_ollama("ok"))
    backend = OllamaBackend(session, "http://ollama:11434", "qwen")
    await backend.async_warm_up()
    assert mocker.mock_calls == []
    backend.keep_loaded = True
    await backend.async_warm_up()
    assert len(mocker.mock_calls) == 1
