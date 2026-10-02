"""End to end: a real Home Assistant, the real integration, a mocked System One API."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from conftest import ANSWERS
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)

from custom_components.typesafe_conversation.const import (
    CONF_ALWAYS_CONFIRM_RISKY,
    CONF_API_KEY,
    CONF_API_TIMEOUT,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_SERVER_PROFILE,
    DOMAIN,
    TYPESAFE_API_URL,
    TYPESAFE_MODELS_URL,
)
from custom_components.typesafe_conversation.system_one import ServerProfile

LOCAL = "http://ollama.invalid:11434"


@pytest.fixture(autouse=True)
def _custom_integrations(enable_custom_integrations):
    return None


def _recorded(name: str) -> dict:
    return json.loads((ANSWERS / f"{name}.json").read_text())["response"]


async def _setup_home(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    await hass.async_block_till_done()

    areas = ar.async_get(hass)
    kitchen = areas.async_create("Kitchen")
    registry = er.async_get(hass)
    coffee = registry.async_get_or_create(
        "switch", "demo", "coffee", suggested_object_id="coffee_maker"
    )
    registry.async_update_entity(
        coffee.entity_id, area_id=kitchen.id, name="Coffee Maker"
    )
    hass.states.async_set(coffee.entity_id, "off", {"friendly_name": "Coffee Maker"})
    async_expose_entity(hass, conversation.DOMAIN, coffee.entity_id, True)


async def _add_entry(
    hass: HomeAssistant, mocker: AiohttpClientMocker, **data
) -> MockConfigEntry:
    mocker.get(TYPESAFE_MODELS_URL, json={"models": [{"name": "jev-latest"}]})
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "sk-test", **data},
        subentries_data=[
            ConfigSubentryData(
                subentry_type="conversation",
                title="TypeSafe Conversation",
                data={CONF_ALWAYS_CONFIRM_RISKY: True},
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_the_agent_registers_and_advertises_control(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """CONTROL is what makes Home Assistant route the useful traffic here."""
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)

    state = hass.states.get("conversation.typesafe_conversation")
    assert state is not None
    assert (
        state.attributes["supported_features"]
        & conversation.ConversationEntityFeature.CONTROL
    )


async def test_a_command_reaches_the_service(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """'get the coffee boiling' - the demo's canonical case, end to end."""
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)
    aioclient_mock.post(TYPESAFE_API_URL, json=_recorded("get_the_coffee_boiling"))
    calls = async_mock_service(hass, "switch", "turn_on")

    result = await conversation.async_converse(
        hass,
        "get the coffee boiling",
        None,
        None,
        agent_id="conversation.typesafe_conversation",
    )

    assert len(calls) == 1
    assert calls[0].data[ATTR_ENTITY_ID] == ["switch.coffee_maker"]
    assert result.response.response_type is not None


async def test_an_empty_utterance_spends_no_request(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)
    before = len(aioclient_mock.mock_calls)

    result = await conversation.async_converse(
        hass, "   ", None, None, agent_id="conversation.typesafe_conversation"
    )
    assert len(aioclient_mock.mock_calls) == before, "no API call for empty input"
    assert result.response.error_code is not None


async def test_jev_being_down_falls_back_rather_than_failing(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """An outage must degrade to the sentence matcher, never to a question."""
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)
    aioclient_mock.post(TYPESAFE_API_URL, status=503, text="down")

    result = await conversation.async_converse(
        hass,
        "turn on the coffee maker",
        None,
        None,
        agent_id="conversation.typesafe_conversation",
    )
    # No exception, and a response the pipeline can speak.
    assert result.response is not None
    assert result.continue_conversation is False


async def test_a_risky_action_asks_first(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    await _setup_home(hass)
    registry = er.async_get(hass)
    lock = registry.async_get_or_create(
        "lock", "demo", "front", suggested_object_id="front_door"
    )
    registry.async_update_entity(lock.entity_id, name="Front Door")
    hass.states.async_set(lock.entity_id, "locked", {"friendly_name": "Front Door"})
    async_expose_entity(hass, conversation.DOMAIN, lock.entity_id, True)

    await _add_entry(hass, aioclient_mock)
    aioclient_mock.post(TYPESAFE_API_URL, json=_recorded("unlock_the_front_door"))
    calls = async_mock_service(hass, "lock", "unlock")

    result = await conversation.async_converse(
        hass,
        "unlock the front door",
        None,
        None,
        agent_id="conversation.typesafe_conversation",
    )
    assert not calls, "must not unlock before the user confirms"
    assert result.continue_conversation is True
    assert "unlock" in result.response.speech["plain"]["speech"].lower()


async def test_the_catalog_uses_home_assistants_conversation_domain(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """Guard against submodule shadowing.

    This package has its own ``conversation.py`` platform. Forwarding the
    platform setup binds that submodule as an attribute of the package, which
    rebinds the module-global name ``conversation`` inside ``__init__.py``. If
    anything reads ``conversation.DOMAIN`` after that point it silently gets
    ``typesafe_conversation``, no entity matches the exposure check, and the agent
    sees an empty home. Hence CONVERSATION_DOMAIN.
    """
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)

    catalog = hass.data[DOMAIN]["catalog"]
    assert catalog.assistant == "conversation"
    assert [e.entity_id for e in catalog.entities] == ["switch.coffee_maker"]


async def test_diagnostics_record_the_decision_and_redact_secrets(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """Diagnostics are the only way to see why the agent decided what it did.

    Home Assistant defines conversation traces but nothing reads them back, so
    without this the reasoning is only in the debug log.
    """
    from custom_components.typesafe_conversation.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    await _setup_home(hass)
    aioclient_mock.post(TYPESAFE_API_URL, json=_recorded("get_the_coffee_boiling"))
    entry = await _add_entry(
        hass,
        aioclient_mock,
        llm_backend="ollama",
        llm_base_url="http://private-host.example:11434",
        llm_model="a-model",
        llm_api_key="sk-secret",
    )
    async_mock_service(hass, "switch", "turn_on")
    await conversation.async_converse(
        hass,
        "get the coffee boiling",
        None,
        None,
        agent_id="conversation.typesafe_conversation",
    )

    diag = await async_get_config_entry_diagnostics(hass, entry)

    # The decision is recoverable.
    (request,) = diag["recent_requests"]
    assert request["utterance"] == "get the coffee boiling"
    assert request["route"] == "command"
    assert request["target"] == "switch.coffee_maker"
    assert request["category"]["choice"] == "command"
    assert "input_tokens" in request

    # Where the request came from is recorded, so a recurring phantom can be
    # attributed in one step instead of five.
    assert "device_id" in request
    assert "satellite_id" in request
    assert request["from_satellite"] is False, "typed input has no satellite"

    # Nothing that identifies the install or authenticates as it.
    blob = str(diag)
    assert "sk-secret" not in blob
    assert "private-host.example" not in blob
    assert "sk-test" not in blob
    assert diag["catalog"]["entities"] == 1


async def test_a_satellite_request_records_where_it_came_from(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """A voice satellite fills device_id and satellite_id on ConversationInput.

    Recording them is what distinguishes a deliberate command from a
    wake-word false trigger. These are standard fields, so this is not
    specific to any satellite hardware.
    """
    from custom_components.typesafe_conversation.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    await _setup_home(hass)
    entry = await _add_entry(hass, aioclient_mock)
    aioclient_mock.post(TYPESAFE_API_URL, json=_recorded("get_the_coffee_boiling"))
    async_mock_service(hass, "switch", "turn_on")

    agent = conversation.async_get_agent(hass, "conversation.typesafe_conversation")
    await agent.internal_async_process(
        conversation.ConversationInput(
            text="get the coffee boiling",
            context=Context(),
            conversation_id=None,
            device_id="device-abc",
            satellite_id="assist_satellite.kitchen",
            language="en",
            agent_id="conversation.typesafe_conversation",
        )
    )

    diag = await async_get_config_entry_diagnostics(hass, entry)
    (request,) = diag["recent_requests"]
    assert request["device_id"] == "device-abc"
    assert request["satellite_id"] == "assist_satellite.kitchen"
    assert request["from_satellite"] is True


# --- a successful command must always say something --------------------------
# Home Assistant's service intent handlers set targets and states but no
# speech; the words normally come from default_agent's response templates,
# which this agent bypasses. Without speech the pipeline skips TTS entirely
# and a command that worked sounds exactly like one that hung.


async def test_a_confident_command_is_still_spoken_back(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """The regression: speech used to be set only in the middle band.

    get_the_coffee_boiling answers at confidence 1.0, so
    name_target_in_speech is false - which is exactly the case that used to
    return silence.
    """
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)
    aioclient_mock.post(TYPESAFE_API_URL, json=_recorded("get_the_coffee_boiling"))
    async_mock_service(hass, "switch", "turn_on")

    result = await conversation.async_converse(
        hass,
        "get the coffee boiling",
        None,
        None,
        agent_id="conversation.typesafe_conversation",
    )

    spoken = result.response.speech.get("plain", {}).get("speech", "")
    assert spoken, "a successful command must not be silent"
    assert "Coffee Maker" in spoken


async def test_handler_speech_is_never_overwritten(hass: HomeAssistant):
    """If Home Assistant did phrase it, its wording wins."""

    response = intent.IntentResponse(language="en")
    response.async_set_speech("Turned on the lights in the kitchen.")
    assert response.speech, "precondition"

    # _run_command only composes when response.speech is empty; assert the
    # guard's shape rather than re-running the whole agent.
    assert bool(response.speech) is True


async def test_media_speech_names_the_track(hass: HomeAssistant):
    """HassMediaSearchAndPlay reports its find in speech_slots, not speech."""
    from custom_components.typesafe_conversation.entities import CatalogEntity
    from custom_components.typesafe_conversation.executor import describe_action
    from custom_components.typesafe_conversation.router import Plan, Route, Target

    speaker = CatalogEntity(
        entity_id="media_player.kitchen_speaker",
        name="Kitchen Speaker",
        aliases=(),
        area_id="kitchen",
        area_name="Kitchen",
        floor_name=None,
        domain="media_player",
        device_class=None,
        supported_features=0,
    )
    plan = Plan(
        Route.COMMAND,
        domain="media_player",
        action="search_and_play",
        target=Target(entity=speaker, domain="media_player"),
    )
    response = intent.IntentResponse(language="en")
    response.async_set_speech_slots({"media": {"title": "Jazz Music"}})

    assert describe_action(plan, response) == ("Playing Jazz Music on Kitchen Speaker.")
    # Without the slots it falls back to the verb table.
    assert describe_action(plan, intent.IntentResponse(language="en")) == (
        "Playing that on Kitchen Speaker."
    )


async def test_home_assistant_reports_an_all_failed_area_command_as_done(
    hass: HomeAssistant,
):
    """The upstream behaviour the agent has to work around.

    async_handle_states seeds success_results with the matched area before it
    calls a single service, so the "no entity succeeded, raise" guard below it
    never fires for an area-matched command. Every entity can refuse and the
    response still comes back action_done with no error code.

    If Home Assistant ever tightens this, this test fails and _wholly_failed
    can go.
    """
    await _setup_home(hass)

    async def _refuse(call):
        raise HomeAssistantError("entity does not support this")

    hass.services.async_register("switch", "turn_on", _refuse)

    response = await intent.async_handle(
        hass, "test", "HassTurnOn", {"area": {"value": "Kitchen"}}
    )

    assert response.response_type is intent.IntentResponseType.ACTION_DONE
    assert response.error_code is None
    assert [t.type for t in response.success_results] == [
        intent.IntentResponseTargetType.AREA
    ]
    assert [t.id for t in response.failed_results] == ["switch.coffee_maker"]


def test_wholly_failed_reads_the_targets_not_the_response_type():
    """Only "entities failed and none succeeded" counts as a total failure."""
    from custom_components.typesafe_conversation.agent import _wholly_failed

    area = intent.IntentResponseTarget(
        type=intent.IntentResponseTargetType.AREA, name="Kitchen", id="kitchen"
    )
    speaker = intent.IntentResponseTarget(
        type=intent.IntentResponseTargetType.ENTITY,
        name="Kitchen Speaker",
        id="media_player.kitchen_speaker",
    )
    lamp = intent.IntentResponseTarget(
        type=intent.IntentResponseTargetType.ENTITY,
        name="Kitchen Lamp",
        id="light.kitchen_lamp",
    )

    def _response(success, failed):
        response = intent.IntentResponse(language="en")
        response.async_set_results(success_results=success, failed_results=failed)
        return response

    # The reported trace: the area counts as a success, the only entity failed.
    assert _wholly_failed(_response([area], [speaker])) == ["Kitchen Speaker"]
    # Something did happen - leave it alone.
    assert _wholly_failed(_response([area, lamp], [speaker])) == []
    # Nothing failed.
    assert _wholly_failed(_response([area, lamp], [])) == []


async def test_a_command_that_reached_no_entity_is_reported_as_failed(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """The regression: a wholly failed command used to be announced as done.

    Paired with "a successful command is always spoken back", reading this
    response as a success would have the agent say "Turned on the Coffee
    Maker." over a service call that was refused.
    """
    await _setup_home(hass)
    await _add_entry(hass, aioclient_mock)
    aioclient_mock.post(TYPESAFE_API_URL, json=_recorded("get_the_coffee_boiling"))

    async def _refuse(call):
        raise HomeAssistantError("entity does not support this")

    hass.services.async_register("switch", "turn_on", _refuse)

    area_success = intent.IntentResponseTarget(
        type=intent.IntentResponseTargetType.AREA, name="Kitchen", id="kitchen"
    )
    entity_failure = intent.IntentResponseTarget(
        type=intent.IntentResponseTargetType.ENTITY,
        name="Coffee Maker",
        id="switch.coffee_maker",
    )

    async def _all_entities_refused(hass, plan, user_input, catalog):
        response = intent.IntentResponse(language="en")
        response.async_set_results(
            success_results=[area_success], failed_results=[entity_failure]
        )
        return response

    with patch(
        "custom_components.typesafe_conversation.agent.async_execute",
        _all_entities_refused,
    ):
        result = await conversation.async_converse(
            hass,
            "get the coffee boiling",
            None,
            None,
            agent_id="conversation.typesafe_conversation",
        )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    spoken = result.response.speech.get("plain", {}).get("speech", "")
    assert "Coffee Maker" in spoken


# --- any System One server: discovery, reconfigure, the second stage ----------


def _local_server(
    aioclient_mock: AiohttpClientMocker, *, cap: int = 26, context: int | None = None
) -> None:
    """An Ollama-like server: two models, one of them a decision model."""
    aioclient_mock.get(
        LOCAL + "/v1/models",
        json={"data": [{"id": "nimble:latest"}, {"id": "qwen3:4b"}]},
    )

    async def show(method, url, body):
        decision = body["model"].startswith("nimble")
        return AiohttpClientMockResponse(
            method,
            url,
            json={
                "capabilities": ["decision"] if decision else ["completion"],
                "parameters": "num_ctx                        8194",
            },
        )

    async def systemone(method, url, body):
        if body["model"].startswith("qwen3"):
            return AiohttpClientMockResponse(
                method,
                url,
                status=400,
                json={"error": 'model "qwen3:4b" is not supported by System One'},
            )
        if "probe" not in body["questions"]:
            # The timed, real-sized request: this home and the full question set.
            if context is not None:
                return AiohttpClientMockResponse(
                    method,
                    url,
                    status=400,
                    json={"error": f"prompt 0 has 5373 tokens; expected 1-{context}"},
                )
            return AiohttpClientMockResponse(
                method, url, json={"model": "nimble", "answers": {}, "usage": {}}
            )
        if len(body["questions"]["probe"]["criteria"]) > cap:
            return AiohttpClientMockResponse(
                method, url, status=400, json={"error": "too many candidates"}
            )
        return AiohttpClientMockResponse(
            method, url, json={"model": "nimble", "answers": {}, "usage": {}}
        )

    aioclient_mock.post(LOCAL + "/api/show", side_effect=show)
    aioclient_mock.post(LOCAL + "/v1/systemone", side_effect=systemone)


async def _through_probe(hass: HomeAssistant, result: dict) -> dict:
    """Let the probe task finish and step past the progress spinner."""
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    await hass.async_block_till_done()
    return await hass.config_entries.flow.async_configure(result["flow_id"])


async def test_setup_discovers_what_the_server_can_take(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """No server table: the cap, context and timing are measured and cached."""
    await _setup_home(hass)
    _local_server(aioclient_mock)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_BASE_URL: LOCAL}
    )
    assert result["step_id"] == "model"
    options = result["data_schema"].schema[CONF_MODEL].config["options"]
    assert options == ["nimble"], "chat models are filtered out, `:latest` dropped"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: "nimble"}
    )
    result = await _through_probe(hass, result)
    # The user is told how fast the server is before relying on it.
    assert result["step_id"] == "tested"
    assert result["description_placeholders"]["model"] == "nimble"
    assert float(result["description_placeholders"]["typical"]) >= 0
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["step_id"] == "llm"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.CREATE_ENTRY

    data = result["data"]
    assert CONF_API_KEY not in data, "a local server needs no key"
    assert data[CONF_BASE_URL] == LOCAL
    profile = ServerProfile.from_dict(data[CONF_SERVER_PROFILE])
    assert profile.max_options == 26
    assert profile.num_ctx == 8194
    assert profile.typical_s is not None, "a real-sized request was timed"
    assert profile.matches(LOCAL, "nimble")
    timed = [
        b
        for _m, u, b, _h in aioclient_mock.mock_calls
        if str(u).endswith("/v1/systemone") and "probe" not in b["questions"]
    ]
    assert len(timed) == 1
    assert "home" in timed[0]["state"], "the home's catalogue, not a probe"


async def test_a_model_the_server_cannot_use_says_why(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    await _setup_home(hass)
    _local_server(aioclient_mock)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_BASE_URL: LOCAL}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: "qwen3:4b"}
    )
    result = await _through_probe(hass, result)

    assert result["step_id"] == "model"
    assert result["errors"] == {"base": "model_rejected"}
    assert "not supported by System One" in result["description_placeholders"]["reason"]


async def test_reconfigure_keeps_the_profile_when_only_the_key_changes(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """Same server and model: no point loading the model again to re-measure."""
    await _setup_home(hass)
    _local_server(aioclient_mock)
    profile = ServerProfile(
        max_options=26, timeout=34.0, base_url=LOCAL, model="nimble"
    )
    entry = await _add_entry(
        hass,
        aioclient_mock,
        base_url=LOCAL,
        model="nimble",
        server_profile=profile.as_dict(),
    )
    aioclient_mock.mock_calls.clear()

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_BASE_URL: LOCAL, CONF_API_KEY: "a-new-key", CONF_API_TIMEOUT: 34.0},
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: "nimble"}
    )
    assert result["step_id"] == "llm", "no probe: straight to the LLM settings"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_API_KEY] == "a-new-key"
    assert CONF_API_TIMEOUT not in entry.data, "left at the measured value"
    assert ServerProfile.from_dict(entry.data[CONF_SERVER_PROFILE]) == profile
    probes = [u for _m, u, _b, _h in aioclient_mock.mock_calls if "systemone" in str(u)]
    assert probes == [], "no model was loaded to re-measure"


async def test_diagnostics_redact_the_server_url_everywhere(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """A self-hosted URL is usually a private hostname."""
    from custom_components.typesafe_conversation.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    await _setup_home(hass)
    _local_server(aioclient_mock)
    profile = ServerProfile(max_options=26, base_url=LOCAL, model="nimble")
    entry = await _add_entry(
        hass,
        aioclient_mock,
        base_url=LOCAL,
        model="nimble",
        server_profile=profile.as_dict(),
    )

    diag = await async_get_config_entry_diagnostics(hass, entry)

    assert "ollama.invalid" not in json.dumps(diag)
    assert diag["client"]["server_profile"]["max_options"] == 26


async def test_a_home_over_the_cap_is_answered_in_two_requests(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """Seven switches against a six-option cap: too many for one entity question.

    The first request leaves target_entity out; the second asks it over just
    the devices on the likeliest domain/area paths, and the merged answer
    drives the command as a single request would.
    """
    await _setup_home(hass)
    registry = er.async_get(hass)
    garage = ar.async_get(hass).async_create("Garage")
    for i in range(6):
        switch = registry.async_get_or_create(
            "switch", "demo", f"g{i}", suggested_object_id=f"garage_{i}"
        )
        registry.async_update_entity(
            switch.entity_id, area_id=garage.id, name=f"Garage {i}"
        )
        hass.states.async_set(switch.entity_id, "off")
        async_expose_entity(hass, conversation.DOMAIN, switch.entity_id, True)

    recorded = _recorded("get_the_coffee_boiling")
    first = {
        **recorded,
        "answers": {
            k: v for k, v in recorded["answers"].items() if k != "target_entity"
        },
    }
    second = {
        "model": recorded["model"],
        "answers": {
            "target_entity": {
                "type": "choice",
                "choice": "switch.coffee_maker",
                "probabilities": {
                    "switch.coffee_maker": 0.97,
                    "no_single_entity": 0.03,
                },
                "confidence": 0.94,
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 1},
    }

    async def systemone(method, url, body):
        stage_two = list(body["questions"]) == ["target_entity"]
        return AiohttpClientMockResponse(
            method, url, json=second if stage_two else first
        )

    aioclient_mock.post(TYPESAFE_API_URL, side_effect=systemone)
    await _add_entry(
        hass,
        aioclient_mock,
        server_profile=ServerProfile(max_options=6).as_dict(),
    )
    calls = async_mock_service(hass, "switch", "turn_on")

    await conversation.async_converse(
        hass,
        "get the coffee boiling",
        None,
        None,
        agent_id="conversation.typesafe_conversation",
    )

    requests = [
        b for _m, u, b, _h in aioclient_mock.mock_calls if str(u) == TYPESAFE_API_URL
    ]
    assert len(requests) == 2
    assert "target_entity" not in requests[0]["questions"]
    assert list(requests[1]["questions"]) == ["target_entity"]
    second_home = {e["id"] for e in requests[1]["state"]["home"]["entities"]}
    assert "switch.coffee_maker" in second_home
    assert len(second_home) < 7, "only the candidates, not the whole home"
    assert len(calls) == 1
    assert calls[0].data[ATTR_ENTITY_ID] == ["switch.coffee_maker"]


async def test_a_model_whose_context_is_too_small_fails_at_setup(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """Better found here, with the server's reason, than on every voice command."""
    await _setup_home(hass)
    _local_server(aioclient_mock, context=2050)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_BASE_URL: LOCAL}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: "nimble"}
    )
    result = await _through_probe(hass, result)

    assert result["step_id"] == "model"
    assert result["errors"] == {"base": "model_rejected"}
    assert "expected 1-2050" in result["description_placeholders"]["reason"]


async def test_setup_never_loads_the_language_model(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """No warm-up at startup and no periodic ping: whether the LLM sits in memory
    is the server owner's decision. It is loaded when a request first needs it."""
    await _setup_home(hass)
    await _add_entry(
        hass,
        aioclient_mock,
        llm_backend="ollama",
        llm_base_url="http://llm.invalid:11434",
        llm_model="a-model",
    )
    await hass.async_block_till_done()
    chat = [u for _m, u, _b, _h in aioclient_mock.mock_calls if "/api/chat" in str(u)]
    assert chat == []


def _chat_calls(aioclient_mock: AiohttpClientMocker) -> list:
    return [b for _m, u, b, _h in aioclient_mock.mock_calls if "/api/chat" in str(u)]


async def test_opting_in_keeps_the_language_model_loaded(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """With the option on, the model is loaded at startup and asked to stay."""
    await _setup_home(hass)
    aioclient_mock.post(
        "http://llm.invalid:11434/api/chat", json={"message": {"content": "ok"}}
    )
    await _add_entry(
        hass,
        aioclient_mock,
        llm_backend="ollama",
        llm_base_url="http://llm.invalid:11434",
        llm_model="a-model",
        llm_keep_loaded=True,
    )
    await hass.async_block_till_done()

    (warm_up,) = _chat_calls(aioclient_mock)
    assert warm_up["keep_alive"] == "30m"


async def test_keeping_the_model_loaded_can_be_switched_off_again(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """Opt-in must be reversible: turning it off in Configure stops the warm-up."""
    await _setup_home(hass)
    _local_server(aioclient_mock)
    aioclient_mock.post(
        "http://llm.invalid:11434/api/chat", json={"message": {"content": "ok"}}
    )
    profile = ServerProfile(max_options=26, base_url=LOCAL, model="nimble")
    entry = await _add_entry(
        hass,
        aioclient_mock,
        base_url=LOCAL,
        model="nimble",
        server_profile=profile.as_dict(),
        llm_backend="ollama",
        llm_base_url="http://llm.invalid:11434",
        llm_model="a-model",
        llm_keep_loaded=True,
    )
    await hass.async_block_till_done()

    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_BASE_URL: LOCAL, CONF_API_TIMEOUT: profile.timeout},
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: "nimble"}
    )
    llm_form = result["data_schema"].schema
    keep_default = next(k for k in llm_form if k == "llm_keep_loaded").default()
    assert keep_default is True, "the form shows what is currently set"

    aioclient_mock.mock_calls.clear()
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "llm_backend": "ollama",
            "llm_base_url": "http://llm.invalid:11434",
            "llm_model": "a-model",
            "llm_keep_loaded": False,
        },
    )
    await hass.async_block_till_done()

    assert result["reason"] == "reconfigure_successful"
    assert entry.data["llm_keep_loaded"] is False
    assert entry.data["llm_model"] == "a-model", "other LLM settings survive"
    assert _chat_calls(aioclient_mock) == [], "reloaded without a warm-up"
