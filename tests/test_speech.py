"""Replies in the request's language, from Home Assistant's own templates."""

from __future__ import annotations

from types import SimpleNamespace

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import intent

from custom_components.typesafe_conversation.actions import spec_for
from custom_components.typesafe_conversation.router import Plan, Route, Target
from custom_components.typesafe_conversation.speech import (
    async_error_text,
    async_localized_action,
    async_localized_query,
    async_render_intent,
    is_english,
)


def _light_off(hass: HomeAssistant) -> tuple[Plan, intent.IntentResponse]:
    hass.states.async_set("light.reading", "off", {"friendly_name": "Reading Light"})
    plan = Plan(
        Route.COMMAND,
        domain="light",
        action="turn_off",
        spec=spec_for("light", "turn_off"),
        target=Target(
            entity=SimpleNamespace(entity_id="light.reading", name="Reading Light"),
            domain="light",
        ),
    )
    return plan, intent.IntentResponse(language="es")


def test_english_keeps_the_agents_own_wording():
    assert is_english("en")
    assert is_english("en-GB")
    assert is_english(None)
    assert not is_english("fr")


async def test_a_command_is_confirmed_in_has_own_spanish(hass: HomeAssistant):
    plan, response = _light_off(hass)

    said = await async_localized_action(hass, "es", plan, response)

    assert said == "Se ha apagado la luz"


async def test_a_regional_code_finds_its_language(hass: HomeAssistant):
    plan, response = _light_off(hass)

    assert await async_localized_action(hass, "es-ES", plan, response) == (
        "Se ha apagado la luz"
    )


async def test_a_state_answer_in_french(hass: HomeAssistant):
    response = intent.IntentResponse(language="fr")
    response.async_set_states([State("switch.valve", "on", {"friendly_name": "Vanne"})])

    said = await async_localized_query(
        hass, "fr", Plan(Route.QUERY, query_kind="device_state"), response
    )

    assert said is not None
    assert said.startswith("Vanne est")


async def test_not_understood_in_spanish(hass: HomeAssistant):
    assert await async_error_text(hass, "es", "no_intent") == (
        "Lo siento, no he entendido"
    )


async def test_no_template_means_no_localized_reply(hass: HomeAssistant):
    response = intent.IntentResponse(language="xx")

    assert await async_error_text(hass, "xx", "no_intent") is None
    assert (
        await async_render_intent(
            hass, "es", "HassNoSuchIntent", "default", response, {}
        )
        is None
    )
