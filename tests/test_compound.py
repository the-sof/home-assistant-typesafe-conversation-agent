"""Compound requests: resolve every part first, act in order, stop on failure.

Physical actions cannot be undone, so the agent must not act on the first part
of "turn off the lamp and start the cleaner" before it knows it understood the
second, and must not carry on after a step that failed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent

from custom_components.typesafe_conversation.agent import AgentSettings, TypeSafeAgent
from custom_components.typesafe_conversation.executor import ExecutionError
from custom_components.typesafe_conversation.router import Plan, Route, Target

AGENT = "custom_components.typesafe_conversation.agent"


def _plan(route: Route, name: str) -> Plan:
    return Plan(route=route, target=Target(entity=SimpleNamespace(name=name)))


def _done() -> intent.IntentResponse:
    response = intent.IntentResponse(language="en")
    response.async_set_speech("ok")
    return response


def _agent(hass: HomeAssistant, parts: list[str]) -> TypeSafeAgent:
    llm = MagicMock()
    llm.split_compound = AsyncMock(return_value=parts)
    agent = TypeSafeAgent(
        hass,
        MagicMock(domains=("light", "vacuum")),
        MagicMock(),
        llm,
        AgentSettings(),
    )
    # The part itself stands in for its System One response, so route() below
    # can map each part to a plan.
    agent._ask = AsyncMock(side_effect=lambda part, *_: (part, []))
    agent._unavailable_ids = MagicMock(return_value=frozenset())
    return agent


def _input(text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id=None,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.typesafe_conversation",
    )


async def _run(agent, text, plans, execute):
    with (
        patch(f"{AGENT}.route", side_effect=lambda part, **_: plans[part]),
        patch(f"{AGENT}.async_execute", execute),
    ):
        return await agent._handle_compound(_input(text), MagicMock())


def _speech(response: intent.IntentResponse) -> str:
    return response.speech["plain"]["speech"]


async def test_an_unclear_part_means_nothing_is_done(hass: HomeAssistant):
    parts = ["turn off the lamp", "start the thing"]
    plans = {
        "turn off the lamp": _plan(Route.COMMAND, "Lamp"),
        "start the thing": _plan(Route.CLARIFY, "Thing"),
    }
    execute = AsyncMock(return_value=_done())

    response = await _run(_agent(hass, parts), " and ".join(parts), plans, execute)

    execute.assert_not_called()
    assert response.response_type is intent.IntentResponseType.ERROR
    assert "start the thing" in _speech(response)


async def test_a_failed_step_stops_the_rest(hass: HomeAssistant):
    parts = ["turn off the lamp", "start the cleaner", "lock the door"]
    plans = {
        "turn off the lamp": _plan(Route.COMMAND, "Lamp"),
        "start the cleaner": _plan(Route.COMMAND, "Cleaner"),
        "lock the door": _plan(Route.COMMAND, "Door"),
    }
    execute = AsyncMock(side_effect=[_done(), ExecutionError("refused"), _done()])

    response = await _run(_agent(hass, parts), "...", plans, execute)

    assert execute.await_count == 2, "the door was never touched"
    assert _speech(response) == (
        "Done: Lamp. But I couldn't start the cleaner, "
        "so I stopped before lock the door."
    )


async def test_every_part_runs_in_the_order_it_was_said(hass: HomeAssistant):
    parts = ["turn on the lamp", "start the cleaner"]
    plans = {
        "turn on the lamp": _plan(Route.COMMAND, "Lamp"),
        "start the cleaner": _plan(Route.COMMAND, "Cleaner"),
    }
    execute = AsyncMock(return_value=_done())

    response = await _run(_agent(hass, parts), "...", plans, execute)

    order = [call.args[1].target.described for call in execute.await_args_list]
    assert order == ["Lamp", "Cleaner"]
    assert _speech(response) == "Done: Lamp and Cleaner."


async def test_a_request_that_cannot_be_split_is_not_guessed_at(hass: HomeAssistant):
    """A failed split used to run the whole sentence as one command."""
    execute = AsyncMock(return_value=_done())

    response = await _run(_agent(hass, []), "do this and that", {}, execute)

    execute.assert_not_called()
    assert response.response_type is intent.IntentResponseType.ERROR
    assert "nothing was done" in _speech(response)
