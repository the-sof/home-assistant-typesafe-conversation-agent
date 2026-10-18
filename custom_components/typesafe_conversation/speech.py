"""Home Assistant's own localized replies, for requests not in English.

The built-in agent words its replies from the response templates shipped in
``home_assistant_intents``, one set per language. This agent bypasses that
agent, so it renders the same templates itself, the way
``default_agent._build_speech`` does. English keeps the agent's own wording.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import intent, template
from homeassistant.util import language as language_util

from .const import LOGGER
from .router import Plan

_CACHE = "typesafe_conversation_intent_texts"


def is_english(language: str | None) -> bool:
    """Whether the agent's own (English) wording already fits."""
    return (language or "en").split("-")[0].lower() == "en"


async def async_intent_texts(hass: HomeAssistant, language: str) -> dict[str, Any]:
    """The ``responses`` section of HA's intents for a language, or empty."""
    cache: dict[str, dict[str, Any]] = hass.data.setdefault(_CACHE, {})
    if language in cache:
        return cache[language]

    def load() -> dict[str, Any]:
        from home_assistant_intents import get_intents, get_languages

        for match in language_util.matches(language, get_languages()):
            if data := get_intents(match):
                return data.get("responses") or {}
        return {}

    try:
        cache[language] = await hass.async_add_executor_job(load)
    except ImportError:
        cache[language] = {}
    return cache[language]


async def async_render_intent(
    hass: HomeAssistant,
    language: str,
    intent_type: str,
    key: str,
    response: intent.IntentResponse,
    slots: dict[str, Any],
) -> str | None:
    """Render HA's reply for an intent, or None when it has none to give."""
    texts = (await async_intent_texts(hass, language)).get("intents") or {}
    source = (texts.get(intent_type) or {}).get(key)
    if not source:
        return None
    first = next(iter([*response.matched_states, *response.unmatched_states]), None)
    return _render(
        hass,
        source,
        {
            "slots": {**slots, **(response.speech_slots or {})},
            "state": _wrap(hass, first),
            "query": {
                "matched": [_wrap(hass, s) for s in response.matched_states],
                "unmatched": [_wrap(hass, s) for s in response.unmatched_states],
            },
        },
    )


async def async_error_text(
    hass: HomeAssistant, language: str, key: str, **args: Any
) -> str | None:
    """HA's localized error reply, such as ``no_intent``, or None."""
    texts = (await async_intent_texts(hass, language)).get("errors") or {}
    if not (source := texts.get(key)):
        return None
    return _render(hass, source, args)


async def async_localized_action(
    hass: HomeAssistant,
    language: str,
    plan: Plan,
    response: intent.IntentResponse,
) -> str | None:
    """HA's confirmation of a command in the request's language, or None."""
    if plan.spec is None or response.failed_results:
        return None
    # Templates read the state they acted on; not every handler sets it.
    if (
        not response.matched_states
        and plan.target.entity is not None
        and (state := hass.states.get(plan.target.entity.entity_id)) is not None
    ):
        response.async_set_states([state])
    intent_type = plan.spec.intent_type
    key = _LIGHT_SET_KEYS.get(plan.action or "", "default")
    if intent_type in (intent.INTENT_TURN_ON, intent.INTENT_TURN_OFF) and (
        plan.domain in _DOMAIN_KEYS
    ):
        key = plan.domain
    slots = {"name": plan.target.described, "domain": plan.domain}
    if plan.target.area_id:
        slots["area"] = plan.target.area_id
    return await async_render_intent(
        hass, language, intent_type, key, response, slots
    ) or await async_render_intent(
        hass, language, intent_type, "default", response, slots
    )


async def async_localized_query(
    hass: HomeAssistant,
    language: str,
    plan: Plan,
    response: intent.IntentResponse,
) -> str | None:
    """HA's answer to a question in the request's language, or None."""
    slots = response.speech_slots or {}
    states = [*response.matched_states, *response.unmatched_states]
    if "time" in slots:
        return await async_render_intent(
            hass, language, intent.INTENT_GET_CURRENT_TIME, "default", response, {}
        )
    if "date" in slots:
        return await async_render_intent(
            hass, language, intent.INTENT_GET_CURRENT_DATE, "default", response, {}
        )
    if len(states) != 1:
        return None
    state = states[0]
    if state.domain == "weather":
        intent_type, key = _GET_WEATHER, "default"
    elif plan.query_kind == "temperature" and state.domain == "climate":
        intent_type, key = intent.INTENT_GET_TEMPERATURE, "default"
    else:
        intent_type, key = intent.INTENT_GET_STATE, "one"
    return await async_render_intent(
        hass, language, intent_type, key, response, {"name": state.name}
    )


_GET_WEATHER = "HassGetWeather"
_LIGHT_SET_KEYS = {
    "set_brightness": "brightness",
    "brighter": "brightness",
    "dimmer": "brightness",
    "set_color": "color",
}
_DOMAIN_KEYS = frozenset({"lock", "valve", "cover"})
"""Domains whose on/off reply HA words differently: locked, opened..."""


def _wrap(hass: HomeAssistant, state: State | None) -> template.TemplateState | None:
    return template.TemplateState(hass, state) if state is not None else None


def _render(hass: HomeAssistant, source: str, variables: dict[str, Any]) -> str | None:
    try:
        text = template.Template(source, hass).async_render(variables)
    except TemplateError as err:
        LOGGER.debug("Could not render a localized reply: %s", err)
        return None
    text = " ".join(str(text).split())
    return text or None
