"""Scripts with fields: Jev picks the script, the language model fills it in.

Everything here uses a synthetic alarm script with a required time and a
required speaker choice, the shape scripts written for LLM agents take.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component

from custom_components.typesafe_conversation import questions as Q
from custom_components.typesafe_conversation.actions import spec_for
from custom_components.typesafe_conversation.agent import AgentSettings, TypeSafeAgent
from custom_components.typesafe_conversation.entities import CatalogEntity
from custom_components.typesafe_conversation.router import Plan, Route, Target, route
from custom_components.typesafe_conversation.script_fields import (
    ScriptField,
    async_script_fields,
    describe_values,
)
from custom_components.typesafe_conversation.system_one import (
    ChoiceAnswer,
    NoulAnswer,
    SystemOneResponse,
)

AGENT = "custom_components.typesafe_conversation.agent"

ALARM = {
    "alias": "Set an alarm",
    "description": "Set an alarm clock for a time of day.",
    "fields": {
        "alarm_time": {
            "name": "Time",
            "description": "The time of day, 24-hour HH:MM:SS.",
            "required": True,
            "selector": {"time": {}},
        },
        "location": {
            "name": "Location",
            "description": "Which speaker: the room the user is in if not said.",
            "required": True,
            "selector": {"select": {"options": ["upstairs", "downstairs"]}},
        },
        "note": {"name": "Note", "selector": {"text": {}}},
        "speakers": {
            "name": "Speakers",
            "selector": {
                "select": {
                    "multiple": True,
                    "options": [
                        {"value": "speaker_a", "label": "Kitchen"},
                        {"value": "speaker_b", "label": "Bedroom"},
                        {"value": "speaker_c", "label": "Study"},
                    ],
                }
            },
        },
    },
    "sequence": [
        {
            "variables": {
                "result": {"message": "Alarm set for {{ alarm_time }} {{ location }}."}
            }
        },
        {"stop": "done", "response_variable": "result"},
    ],
}


async def _scripts(hass: HomeAssistant) -> None:
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "set_alarm": ALARM,
                "bedtime": {"alias": "Bedtime", "sequence": [{"delay": 0}]},
            }
        },
    )
    await hass.async_block_till_done()


# --- reading a script's fields -------------------------------------------------


async def test_a_scripts_fields_are_read_from_home_assistant(hass: HomeAssistant):
    await _scripts(hass)
    fields = async_script_fields(hass, "script.set_alarm")

    assert fields is not None
    assert fields.title == "Set an alarm"
    assert [f.key for f in fields.fields] == [
        "alarm_time",
        "location",
        "note",
        "speakers",
    ]
    schema = fields.json_schema()
    assert schema["properties"]["location"]["enum"] == ["upstairs", "downstairs"]
    assert "required" not in schema, "a required field is asked for, never guessed"


async def test_a_script_without_fields_needs_no_filling(hass: HomeAssistant):
    await _scripts(hass)
    assert async_script_fields(hass, "script.bedtime") is None


async def test_values_are_checked_against_each_selector(hass: HomeAssistant):
    await _scripts(hass)
    fields = async_script_fields(hass, "script.set_alarm")

    values, missing = fields.check(
        {
            "alarm_time": "5:15 am",  # not a time the selector accepts
            "location": "attic",  # not one of the options
            "note": "pack the bag",
            "invented": "x",
        }
    )

    assert values == {"note": "pack the bag"}, "nothing coerced, nothing invented"
    assert [f.key for f in missing] == ["alarm_time", "location"]


def test_values_are_said_in_plain_words():
    assert describe_values({"t": "05:15:00", "l": "upstairs"}) == "5:15 AM, upstairs"
    assert describe_values({"t": "17:05"}) == "5:05 PM"


def test_a_field_without_a_selector_takes_text():
    field = ScriptField("note", "Note", "", False, None)
    assert field.json_schema()["type"] == "string"
    assert field.check(" hi ") == "hi"


# --- a confident routine needs no vote on the action ---------------------------


def _choice(choice: str, p: float, other: str = "other") -> ChoiceAnswer:
    return ChoiceAnswer(
        choice=choice, probabilities={choice: p, other: 1 - p}, confidence=p
    )


def test_a_confident_script_runs_even_when_the_action_vote_splits():
    """The live failure: script.set_alarm at 0.96, then 'run' lost 45/55."""
    alarm = CatalogEntity(
        entity_id="script.set_alarm",
        name="Set an alarm",
        aliases=(),
        area_id=None,
        area_name=None,
        floor_name=None,
        domain="script",
        device_class=None,
        supported_features=0,
    )
    response = SystemOneResponse(
        model="jev",
        answers={
            Q.Q_CATEGORY: _choice("command", 0.99),
            Q.Q_COMPOUND: NoulAnswer(0.05),
            Q.Q_QUERY_KIND: _choice("not_a_query", 1.0),
            Q.Q_SCOPE: _choice("single", 0.74, "not_a_target"),
            Q.Q_TARGET_ENTITY: _choice("script.set_alarm", 0.96, Q.NO_SINGLE_ENTITY),
            Q.Q_TARGET_DOMAIN: _choice("script", 0.85, Q.NO_DOMAIN),
            Q.Q_TARGET_AREA: _choice(Q.NO_AREA, 0.9),
            Q.Q_RISKY: NoulAnswer(0.03),
            Q.Q_HERE_RELATIVE: NoulAnswer(0.19),
            Q.action_question_id("script"): _choice(Q.NOT_TARGETED, 0.55, "run"),
        },
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
    )
    from custom_components.typesafe_conversation.extraction import extract

    plan = route(
        response,
        entities_by_id={alarm.entity_id: alarm},
        extraction=extract("wake me up", want_media=False, want_color=False),
        speaker_area_id=None,
        available_domains=frozenset({"script"}),
    )

    assert plan.route is Route.COMMAND
    assert plan.action == "run"
    assert plan.trace["action"] == "implied"


# --- filling, asking, and running ----------------------------------------------


def _alarm_plan() -> Plan:
    alarm = CatalogEntity(
        entity_id="script.set_alarm",
        name="Set an alarm",
        aliases=(),
        area_id=None,
        area_name=None,
        floor_name=None,
        domain="script",
        device_class=None,
        supported_features=0,
    )
    return Plan(
        Route.COMMAND,
        domain="script",
        action="run",
        spec=spec_for("script", "run"),
        target=Target(entity=alarm, domain="script"),
    )


def _agent(hass: HomeAssistant, llm, pending: dict) -> TypeSafeAgent:
    agent = TypeSafeAgent(
        hass,
        MagicMock(domains=("script",), areas=[]),
        MagicMock(),
        llm,
        AgentSettings(),
        traces=[],
        pending_fills=pending,
    )
    agent._ask = AsyncMock(return_value=(MagicMock(raw={}), []))
    agent._unavailable_ids = MagicMock(return_value=frozenset())
    return agent


def _input(text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id="c1",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.typesafe_conversation",
    )


async def _say(agent, text):
    with patch(f"{AGENT}.route", return_value=_alarm_plan()):
        return await agent.async_process(_input(text), MagicMock(conversation_id="c1"))


def _speech(response: intent.IntentResponse) -> str:
    return response.speech["plain"]["speech"]


async def test_a_script_runs_with_the_values_the_model_filled(hass: HomeAssistant):
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        return_value={"alarm_time": "05:15:00", "location": "upstairs"}
    )

    response = await _say(_agent(hass, llm, {}), "wake me up at 5.15 a.m. tomorrow")

    assert _speech(response) == "Alarm set for 05:15:00 upstairs."
    assert llm.fill_fields.await_args.kwargs["title"] == "Set an alarm"


async def test_a_missing_value_is_asked_for_then_used(hass: HomeAssistant):
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {"location": "upstairs"},
            {"alarm_time": "05:15:00", "location": "upstairs"},
        ]
    )
    pending: dict = {}
    agent = _agent(hass, llm, pending)

    asked = await _say(agent, "set an alarm")
    assert _speech(asked) == "What time should I use?"
    assert agent.continue_conversation is True
    assert "c1" in pending

    done = await _say(agent, "5:15 in the morning")
    assert _speech(done) == "Alarm set for 05:15:00 upstairs."
    assert llm.fill_fields.await_args.kwargs["earlier"] == "set an alarm"
    assert not pending


async def test_an_unrelated_answer_drops_the_wait(hass: HomeAssistant):
    """ "Never mind" must not be taken as the time, nor leave the agent stuck."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(side_effect=[{"location": "upstairs"}, {}, {}])
    pending: dict = {}
    agent = _agent(hass, llm, pending)

    await _say(agent, "set an alarm")
    agent._carry_out = AsyncMock(return_value=intent.IntentResponse(language="en"))
    with patch(f"{AGENT}.route", return_value=Plan(Route.CANCEL)):
        await agent.async_process(_input("never mind"), MagicMock(conversation_id="c1"))

    agent._carry_out.assert_awaited_once()  # handled as a new request
    assert not pending


async def test_without_a_language_model_the_script_is_not_run_blind(
    hass: HomeAssistant,
):
    await _scripts(hass)
    response = await _say(_agent(hass, None, {}), "wake me up at 5")
    assert response.response_type is intent.IntentResponseType.ERROR
    assert "language model" in _speech(response)


@pytest.mark.parametrize("values", [{"alarm_time": "05:15:00"}, {}])
async def test_a_compound_part_missing_a_value_runs_nothing(hass, values):
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(return_value=values)
    llm.split_compound = AsyncMock(return_value=["set an alarm", "turn on the lamp"])
    agent = _agent(hass, llm, {})
    execute = AsyncMock()

    with (
        patch(f"{AGENT}.route", return_value=_alarm_plan()),
        patch(f"{AGENT}.async_execute", execute),
    ):
        response = await agent._handle_compound(
            _input("set an alarm and turn on the lamp"), MagicMock()
        )

    execute.assert_not_called()
    assert response.response_type is intent.IntentResponseType.ERROR


async def test_answers_are_kept_while_another_value_is_still_needed(
    hass: HomeAssistant,
):
    """Two values missing: the first answer is kept, and the second is asked for."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {},
            {"alarm_time": "05:15:00"},
            {"location": "upstairs"},
        ]
    )
    pending: dict = {}
    agent = _agent(hass, llm, pending)

    assert _speech(await _say(agent, "set an alarm")) == "What time should I use?"
    assert _speech(await _say(agent, "5:15 am")) == "What location should I use?"
    done = await _say(agent, "upstairs")

    assert _speech(done) == "Alarm set for 05:15:00 upstairs."
    assert llm.fill_fields.await_args.kwargs["earlier"] == "set an alarm"


async def test_an_unanswered_question_is_forgotten_in_time(hass: HomeAssistant):
    """Conversations that never answer must not pile up for the life of the entry."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(return_value={})
    stale = MagicMock(expires=0.0)
    pending: dict = {"long-gone": stale}

    await _say(_agent(hass, llm, pending), "set an alarm")

    assert "long-gone" not in pending
    assert "c1" in pending


async def test_a_default_cannot_replace_what_the_user_said(hass: HomeAssistant):
    """ "downstairs" stays when a later fill only guesses the current room."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {"location": "downstairs"},
            {"alarm_time": "05:15:00", "location": "upstairs"},
        ]
    )
    agent = _agent(hass, llm, {})

    await _say(agent, "set an alarm downstairs")
    done = await _say(agent, "5:15 am")

    assert _speech(done) == "Alarm set for 05:15:00 downstairs."
    kwargs = llm.fill_fields.await_args.kwargs
    assert kwargs["known"] == {"location": "downstairs"}
    assert kwargs["asking"] == "Time"


async def test_a_change_of_mind_wins(hass: HomeAssistant):
    """ "Actually, upstairs" replaces the settled "downstairs"."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {"location": "downstairs"},
            {"alarm_time": "05:15:00", "location": "upstairs"},
        ]
    )
    agent = _agent(hass, llm, {})

    await _say(agent, "set an alarm downstairs")
    done = await _say(agent, "5:15 am, actually upstairs")

    assert _speech(done) == "Alarm set for 05:15:00 upstairs."


async def test_a_changed_time_is_trusted(hass: HomeAssistant):
    """A time can't come from a default, so the model's change is taken."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {"alarm_time": "05:00:00"},
            {"alarm_time": "05:30:00", "location": "upstairs"},
        ]
    )
    agent = _agent(hass, llm, {})

    await _say(agent, "set an alarm for 5")
    done = await _say(agent, "upstairs, and make it half past")

    assert _speech(done) == "Alarm set for 05:30:00 upstairs."


@pytest.mark.parametrize(
    ("reply", "used"),
    [
        ("5:15, actually use Kitchen and Bedroom", ["speaker_a", "speaker_b"]),
        ("5:15 am", ["speaker_c"]),
    ],
)
async def test_several_choices_are_checked_one_by_one(hass: HomeAssistant, reply, used):
    """Each added value is named by its label; an unnamed one keeps the old choice."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {"location": "upstairs", "speakers": ["speaker_c"]},
            {"alarm_time": "05:15:00", "speakers": ["speaker_a", "speaker_b"]},
        ]
    )
    agent = _agent(hass, llm, {})
    done = intent.IntentResponse(language="en")
    done.async_set_speech("ok")
    execute = AsyncMock(return_value=done)

    await _say(agent, "set an alarm upstairs on the study speaker")
    with patch(f"{AGENT}.async_execute", execute):
        await _say(agent, reply)

    assert execute.await_args.args[1].script_data["speakers"] == used


@pytest.mark.parametrize(
    ("reply", "used"),
    [
        ("5:15", ["speaker_a", "speaker_b"]),
        ("5:15, just the kitchen", ["speaker_a"]),
        ("5:15, not the bedroom", ["speaker_a"]),
    ],
)
async def test_a_choice_is_only_dropped_when_asked(hass: HomeAssistant, reply, used):
    """A reply that only gives the time must not lose a speaker chosen earlier."""
    await _scripts(hass)
    llm = MagicMock()
    llm.fill_fields = AsyncMock(
        side_effect=[
            {"location": "upstairs", "speakers": ["speaker_a", "speaker_b"]},
            {"alarm_time": "05:15:00", "speakers": ["speaker_a"]},
        ]
    )
    agent = _agent(hass, llm, {})
    done = intent.IntentResponse(language="en")
    done.async_set_speech("ok")
    execute = AsyncMock(return_value=done)

    await _say(agent, "set an alarm upstairs on the kitchen and bedroom speakers")
    with patch(f"{AGENT}.async_execute", execute):
        await _say(agent, reply)

    assert execute.await_args.args[1].script_data["speakers"] == used


# --- questions answered by scripts, and what scripts reply ----------------------

REPLIES = {
    "list_alarms": {
        "alias": "List alarms",
        "sequence": [
            {"variables": {"reply": {"result": "Two alarms are set."}}},
            {"stop": "listed", "response_variable": "reply"},
        ],
    },
    "full_alarm": {
        "alias": "Full alarm",
        "sequence": [
            {"variables": {"reply": {"result": "All three alarm slots are in use."}}},
            {"stop": "no slot", "response_variable": "reply"},
        ],
    },
    "only_string": {
        "alias": "Only string",
        "sequence": [
            {"variables": {"reply": {"anything": "Said in its own key.", "n": 2}}},
            {"stop": "ok", "response_variable": "reply"},
        ],
    },
    "silent": {"alias": "Silent", "sequence": [{"delay": 0}]},
    "slow": {"alias": "Slow routine", "sequence": [{"delay": {"seconds": 1}}]},
}


async def _reply_scripts(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "script", {"script": REPLIES})
    await hass.async_block_till_done()


def _script_plan(object_id: str, title: str) -> Plan:
    entity = CatalogEntity(
        entity_id=f"script.{object_id}",
        name=title,
        aliases=(),
        area_id=None,
        area_name=None,
        floor_name=None,
        domain="script",
        device_class=None,
        supported_features=0,
    )
    return Plan(
        Route.COMMAND,
        domain="script",
        action="run",
        spec=spec_for("script", "run"),
        target=Target(entity=entity, domain="script"),
    )


async def _run(hass, object_id, title):
    agent = _agent(hass, None, {})
    with patch(f"{AGENT}.route", return_value=_script_plan(object_id, title)):
        return await agent.async_process(_input("go"), MagicMock(conversation_id="c1"))


@pytest.mark.parametrize(
    ("object_id", "title", "said"),
    [
        ("list_alarms", "List alarms", "Two alarms are set."),
        ("full_alarm", "Full alarm", "All three alarm slots are in use."),
        ("only_string", "Only string", "Said in its own key."),
        ("silent", "Silent", "Done: Silent."),
    ],
)
async def test_what_a_script_replies_is_what_is_said(
    hass: HomeAssistant, object_id, title, said
):
    """Including a refusal: "slots in use" must never be announced as "Done"."""
    await _reply_scripts(hass)
    assert _speech(await _run(hass, object_id, title)) == said


async def test_a_long_routine_is_left_running(hass: HomeAssistant):
    """A delay in the script must not hold up the voice reply."""
    await _reply_scripts(hass)
    with patch(
        "custom_components.typesafe_conversation.executor.SCRIPT_REPLY_SECONDS", 0.05
    ):
        response = await _run(hass, "slow", "Slow routine")
    assert _speech(response) == "Started Slow routine."
    assert hass.states.get("script.slow").state == "on"
    await hass.async_block_till_done()
