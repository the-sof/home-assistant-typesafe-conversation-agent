"""Turn a Plan into Home Assistant intent calls.

Targeting goes through ``intent.async_handle`` with the resolved ``entity_id``
in the ``name`` slot. ``intent._filter_by_name`` matches an entity id exactly
before it tries any alias, and ``find_areas`` accepts an ``area_id``, so this
is deterministic - two lamps called "Ceiling Light" cannot be confused - while
still getting Home Assistant's own exposure check and response speech.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from homeassistant.components import conversation
from homeassistant.const import (
    STATE_OFF,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from .actions import (
    ACTION_VERBS,
    INTENT_CLIMATE_GET_TEMPERATURE,
    INTENT_GET_CURRENT_TIME,
    INTENT_GET_STATE,
    INTENT_NEVERMIND,
    INTENT_TURN_OFF,
    INTENT_TURN_ON,
)
from .const import CONVERSATION_DOMAIN, DOMAIN, LOGGER
from .entities import CONTROLLABLE_DOMAINS, EntityCatalog
from .router import Plan, Target


class ExecutionError(Exception):
    """The plan could not be carried out."""

    def __init__(self, message: str, code: str = "failed_to_handle") -> None:
        super().__init__(message)
        self.code = code


def _slot(value: Any, text: str | None = None) -> dict[str, Any]:
    """Build one intent slot.

    ``text`` rides alongside the value: the handler matches on ``value`` and
    then swaps in ``text`` for its response template, so we can target by
    entity id and still have the user hear the friendly name. SLOT_SCHEMA
    allows the extra key.
    """
    slot: dict[str, Any] = {"value": value}
    if text is not None:
        slot["text"] = text
    return slot


def build_slots(
    target: Target,
    spec_name_only: bool = False,
    preferred_area_id: str | None = None,
) -> dict[str, Any]:
    """Express a Target as intent slots.

    ``preferred_area_id`` is sent *instead of* a hard ``area`` when the router
    found that area holds nothing of the target domain. Home Assistant treats
    it as a tie-breaker rather than a filter, so the match widens to the home
    while still preferring that room.
    """
    slots: dict[str, Any] = {}
    if target.entity is not None:
        slots["name"] = _slot(target.entity.entity_id, target.entity.name)
        if target.entity.area_id and not spec_name_only:
            # Belt and braces: narrows the match without changing the result.
            slots["area"] = _slot(target.entity.area_id)
        return slots

    if target.floor_name:
        slots["floor"] = _slot(target.floor_name)
    elif target.area_id:
        slots["area"] = _slot(target.area_id)
    elif preferred_area_id:
        slots["preferred_area_id"] = _slot(preferred_area_id)

    if target.domain:
        slots["domain"] = _slot([target.domain])
    return slots


def _temperature_unit(hass: HomeAssistant, plan: Plan) -> str:
    """Decide what unit a bare "21 degrees" meant.

    Never asked of the model: the entity knows, and failing that the house
    does. Only an explicit C or F in the utterance overrides them.
    """
    if plan.value_unit in ("c", "f"):
        return (
            UnitOfTemperature.CELSIUS
            if plan.value_unit == "c"
            else UnitOfTemperature.FAHRENHEIT
        )
    if plan.target.entity is not None:
        state = hass.states.get(plan.target.entity.entity_id)
        if state is not None and (unit := state.attributes.get("temperature_unit")):
            return unit
    return hass.config.units.temperature_unit


async def async_execute(
    hass: HomeAssistant,
    plan: Plan,
    user_input: conversation.ConversationInput,
    catalog: EntityCatalog,
) -> intent.IntentResponse:
    """Carry out a COMMAND plan."""
    if plan.target.whole_house and plan.target.domain is None:
        return await _execute_whole_house(hass, plan, user_input, catalog)

    slots = build_slots(
        plan.target,
        spec_name_only=bool(plan.spec and plan.spec.name_only),
        preferred_area_id=plan.preferred_area_id,
    )
    if not slots:
        raise ExecutionError("Nothing to target", code="no_valid_targets")

    await _add_value_slots(hass, plan, slots, catalog)

    if plan.spec is None:
        raise ExecutionError("No action to perform")

    return await _handle(hass, plan.spec.intent_type, slots, user_input)


async def _add_value_slots(
    hass: HomeAssistant, plan: Plan, slots: dict[str, Any], catalog: EntityCatalog
) -> None:
    spec = plan.spec
    if spec is None:
        return

    if plan.text_slot is not None:
        slots[plan.text_slot[0]] = _slot(plan.text_slot[1])
    if plan.color_temp_kelvin is not None:
        slots["temperature"] = _slot(plan.color_temp_kelvin)

    if spec.value_slot is None:
        return

    if spec.relative:
        step = plan.relative_step or 0
        if spec.value_kind == "volume_step":
            # HassSetVolumeRelative takes the delta directly.
            slots[spec.value_slot] = _slot(max(-100, min(100, step)))
            return
        if spec.value_kind == "temperature":
            current = _current_value(hass, plan, catalog, step)
            # A percentage-point step makes no sense for temperature; treat the
            # magnitude as tenths of a degree per point (1pp -> 0.1 degree).
            slots[spec.value_slot] = _slot(round(current + step * 0.1, 1))
            return
        current = _current_value(hass, plan, catalog, step)
        slots[spec.value_slot] = _slot(max(1, min(100, int(current + step))))
        return

    if plan.value is None:
        raise ExecutionError(f"{plan.action} needs a value")

    if spec.value_kind == "temperature":
        slots[spec.value_slot] = _slot(float(plan.value))
        slots.setdefault("unit", _slot(_temperature_unit(hass, plan)))
    elif spec.value_kind in ("percent", "volume"):
        slots[spec.value_slot] = _slot(max(0, min(100, int(plan.value))))
    else:
        slots[spec.value_slot] = _slot(plan.value)


_RELATIVE_ATTRIBUTE = {
    "light": "brightness",
    "fan": "percentage",
    "cover": "current_position",
    "climate": "temperature",
    "water_heater": "temperature",
}


def _current_value(
    hass: HomeAssistant, plan: Plan, catalog: EntityCatalog, step: int
) -> float:
    """What a relative change is relative *to*, read from the real states.

    Never a made-up default: changing "a bit" from an assumed 50% can brighten
    a light the user wanted dimmer. A room or floor uses the mean of what its
    targeted devices report. With nothing to read, the user is asked for an
    absolute value instead.
    """
    attribute = _RELATIVE_ATTRIBUTE.get(plan.domain or "")
    values: list[float] = []
    all_off = True
    for entity_id in _relative_targets(plan, catalog):
        state = hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            continue
        if plan.domain == "light" and state.state == STATE_OFF:
            values.append(0.0)
            continue
        all_off = False
        raw = state.attributes.get(attribute) if attribute else None
        if _finite(raw):
            values.append(raw / 255 * 100 if plan.domain == "light" else float(raw))
    if values and plan.domain == "light" and all_off and step <= 0:
        raise ExecutionError("It's already off.")
    if not values:
        raise ExecutionError(
            "I don't know its current level, so please say a value instead."
        )
    return fmean(values)


def _relative_targets(plan: Plan, catalog: EntityCatalog) -> list[str]:
    target = plan.target
    if target.entity is not None:
        return [target.entity.entity_id]
    area_id = target.area_id or plan.preferred_area_id
    return [
        e.entity_id
        for e in catalog.entities
        if e.domain == plan.domain
        and (
            target.whole_house
            or (area_id is not None and e.area_id == area_id)
            or (target.floor_name is not None and e.floor_name == target.floor_name)
        )
    ]


def _finite(value: Any) -> bool:
    """A real number: not a bool, not NaN or infinity."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


async def _execute_whole_house(
    hass: HomeAssistant,
    plan: Plan,
    user_input: conversation.ConversationInput,
    catalog: EntityCatalog,
) -> intent.IntentResponse:
    """Handle "turn off everything", which names no domain.

    An intent with no name, area, floor or domain raises IntentHandleError
    ("Service handler cannot target all devices"), so we cannot just pass an
    empty slot set. Fan out over the controllable domains this home actually
    has instead.
    """
    intent_type = INTENT_TURN_OFF if plan.action == "turn_off" else INTENT_TURN_ON
    domains = [d for d in catalog.domains if d in CONTROLLABLE_DOMAINS]
    # Scenes, scripts and buttons are triggers, not states; "turn everything
    # off" should not fire them.
    domains = [d for d in domains if d not in ("scene", "script", "button")]
    if not domains:
        raise ExecutionError("Nothing in this home can be switched")

    succeeded: list[str] = []
    last_error: Exception | None = None
    for domain in domains:
        try:
            await _handle(hass, intent_type, {"domain": _slot([domain])}, user_input)
            succeeded.append(domain)
        except intent.MatchFailedError:
            # Nothing of that kind was in a state worth changing.
            continue
        except intent.IntentError as err:
            last_error = err
            LOGGER.debug("Whole-house %s failed for %s: %s", intent_type, domain, err)

    response = intent.IntentResponse(language=user_input.language)
    if not succeeded:
        raise ExecutionError(
            str(last_error) if last_error else "Nothing to change",
            code="no_valid_targets",
        )
    verb = "Turned off" if intent_type == INTENT_TURN_OFF else "Turned on"
    response.async_set_speech(f"{verb} everything.")
    return response


async def async_execute_query(
    hass: HomeAssistant,
    plan: Plan,
    user_input: conversation.ConversationInput,
) -> intent.IntentResponse:
    """Answer a question about the home using the built-in handlers."""
    kind = plan.query_kind
    if kind == "time_or_date":
        return await _handle(hass, INTENT_GET_CURRENT_TIME, {}, user_input)
    if kind == "temperature":
        slots: dict[str, Any] = {}
        if plan.target.area_id:
            slots["area"] = _slot(plan.target.area_id)
        if plan.target.entity is not None:
            slots["name"] = _slot(plan.target.entity.entity_id, plan.target.entity.name)
        return await _handle(hass, INTENT_CLIMATE_GET_TEMPERATURE, slots, user_input)

    slots = {}
    if plan.target.entity is not None:
        slots["name"] = _slot(plan.target.entity.entity_id, plan.target.entity.name)
    if plan.target.area_id:
        slots["area"] = _slot(plan.target.area_id)
    if plan.target.domain:
        slots["domain"] = _slot([plan.target.domain])
    if not slots:
        raise ExecutionError("Not enough to look up", code="no_valid_targets")
    return await _handle(hass, INTENT_GET_STATE, slots, user_input)


async def async_execute_cancel(
    hass: HomeAssistant, user_input: conversation.ConversationInput
) -> intent.IntentResponse:
    return await _handle(hass, INTENT_NEVERMIND, {}, user_input)


async def _handle(
    hass: HomeAssistant,
    intent_type: str,
    slots: dict[str, Any],
    user_input: conversation.ConversationInput,
) -> intent.IntentResponse:
    LOGGER.debug("intent %s slots=%s", intent_type, slots)
    return await intent.async_handle(
        hass,
        DOMAIN,
        intent_type,
        slots,
        text_input=user_input.text,
        context=user_input.context,
        language=user_input.language,
        assistant=CONVERSATION_DOMAIN,
        device_id=user_input.device_id,
        satellite_id=user_input.satellite_id,
        conversation_agent_id=user_input.agent_id,
    )


def describe_action(plan: Plan, response: intent.IntentResponse | None = None) -> str:
    """Compose speech for a command that succeeded.

    Home Assistant's service intent handlers set targets and states but no
    speech - the words normally come from default_agent's response templates,
    which this agent bypasses. So if we do not say something, nothing does,
    the pipeline skips TTS entirely, and a command that worked is
    indistinguishable from one that hung.
    """
    # HassMediaSearchAndPlay reports what it found in speech_slots rather than
    # speech. Naming the track is far better than a generic acknowledgement.
    if response is not None:
        media = (response.speech_slots or {}).get("media")
        if isinstance(media, dict) and media.get("title"):
            return f"Playing {media['title']} on {plan.target.described}."

    verb = ACTION_VERBS.get(plan.action or "", "Did that to")
    return f"{verb} {plan.target.described}."


__all__ = [
    "ExecutionError",
    "async_execute",
    "async_execute_cancel",
    "async_execute_query",
    "build_slots",
    "describe_action",
]
