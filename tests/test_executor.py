"""Execution, against real Home Assistant intent handlers.

The central claim of this integration's targeting is that an ``entity_id`` can
be passed in the ``name`` slot and will resolve exactly. These tests hold it to
that, including the case it exists to solve: two entities with the same name.
"""

from __future__ import annotations

import pytest
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.typesafe_conversation.actions import spec_for
from custom_components.typesafe_conversation.entities import EntityCatalog
from custom_components.typesafe_conversation.executor import async_execute, build_slots
from custom_components.typesafe_conversation.router import Plan, Route, Target


async def _setup(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, "intent", {})
    await hass.async_block_till_done()


def _input(hass: HomeAssistant) -> conversation.ConversationInput:
    from homeassistant.core import Context

    return conversation.ConversationInput(
        text="test",
        context=Context(),
        conversation_id=None,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.typesafe",
    )


def _expose(hass: HomeAssistant, entity_id: str) -> None:
    async_expose_entity(hass, conversation.DOMAIN, entity_id, True)


async def test_entity_id_in_the_name_slot_resolves_exactly(hass: HomeAssistant):
    """Two lights share a name; only the one we targeted may be switched."""
    await _setup(hass)
    areas = ar.async_get(hass)
    kitchen = areas.async_create("Kitchen")
    bedroom = areas.async_create("Bedroom")
    registry = er.async_get(hass)

    a = registry.async_get_or_create("light", "demo", "a")
    b = registry.async_get_or_create("light", "demo", "b")
    registry.async_update_entity(a.entity_id, area_id=kitchen.id)
    registry.async_update_entity(b.entity_id, area_id=bedroom.id)
    # Deliberately identical friendly names.
    hass.states.async_set(a.entity_id, "off", {"friendly_name": "Ceiling Light"})
    hass.states.async_set(b.entity_id, "off", {"friendly_name": "Ceiling Light"})
    _expose(hass, a.entity_id)
    _expose(hass, b.entity_id)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    target_entity = next(e for e in catalog.entities if e.entity_id == b.entity_id)

    calls = async_mock_service(hass, "light", "turn_on")
    plan = Plan(
        Route.COMMAND,
        domain="light",
        action="turn_on",
        spec=spec_for("light", "turn_on"),
        target=Target(
            entity=target_entity, area_id=target_entity.area_id, domain="light"
        ),
    )
    await async_execute(hass, plan, _input(hass), catalog)

    assert len(calls) == 1
    assert calls[0].data[ATTR_ENTITY_ID] == [b.entity_id], (
        "the duplicate name must not win"
    )


async def test_friendly_text_rides_along_for_the_response(hass: HomeAssistant):
    """The id targets; the text is what the user hears."""
    await _setup(hass)
    registry = er.async_get(hass)
    entry = registry.async_get_or_create("switch", "demo", "coffee")
    hass.states.async_set(entry.entity_id, "off", {"friendly_name": "Coffee Maker"})
    _expose(hass, entry.entity_id)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    entity = catalog.entities[0]
    slots = build_slots(Target(entity=entity, domain="switch"))
    assert slots["name"]["value"] == entry.entity_id
    assert slots["name"]["text"] == "Coffee Maker"


async def test_brightness_percentage_reaches_the_service(hass: HomeAssistant):
    await _setup(hass)
    assert await async_setup_component(hass, "light", {})
    # Domain intents live on the integration's intent platform; the intent
    # component loads them lazily, so nudge it before asserting on them.
    from homeassistant.components.light import intent as light_intent

    await light_intent.async_setup_intents(hass)
    registry = er.async_get(hass)
    entry = registry.async_get_or_create("light", "demo", "lamp")
    hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Lamp"})
    _expose(hass, entry.entity_id)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    calls = async_mock_service(hass, "light", "turn_on")
    plan = Plan(
        Route.COMMAND,
        domain="light",
        action="set_brightness",
        spec=spec_for("light", "set_brightness"),
        target=Target(entity=catalog.entities[0], domain="light"),
        value=30.0,
        value_unit="%",
    )
    await async_execute(hass, plan, _input(hass), catalog)
    assert calls[0].data["brightness_pct"] == 30


async def test_whole_house_always_carries_a_domain(hass: HomeAssistant):
    """An intent with no constraint at all raises IntentHandleError.

    'turn off everything' names no domain, so the executor fans out over the
    domains the home actually has rather than sending an empty slot set.
    """
    await _setup(hass)
    registry = er.async_get(hass)
    light = registry.async_get_or_create("light", "demo", "l")
    switch = registry.async_get_or_create("switch", "demo", "s")
    hass.states.async_set(light.entity_id, "on", {"friendly_name": "L"})
    hass.states.async_set(switch.entity_id, "on", {"friendly_name": "S"})
    _expose(hass, light.entity_id)
    _expose(hass, switch.entity_id)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    light_calls = async_mock_service(hass, "light", "turn_off")
    switch_calls = async_mock_service(hass, "switch", "turn_off")

    plan = Plan(
        Route.COMMAND,
        action="turn_off",
        target=Target(whole_house=True),
    )
    response = await async_execute(hass, plan, _input(hass), catalog)
    assert len(light_calls) == 1
    assert len(switch_calls) == 1
    assert response.speech


async def test_empty_constraints_would_have_raised(hass: HomeAssistant):
    """Guard the assumption the whole-house fan-out is built on."""
    await _setup(hass)
    with pytest.raises(intent.IntentHandleError):
        await intent.async_handle(hass, "test", "HassTurnOff", {})


# --- targeting an area that holds nothing of the domain ----------------------
# Sending such an area is a guaranteed MatchFailedError, and the catalog says
# so before the call. What to do instead depends on the handler, because
# Home Assistant consults preferred_area_id only for single-target matches and
# duplicate names - never to widen a fan-out command.


def _entity(entity_id, name, area, domain):
    from custom_components.typesafe_conversation.entities import CatalogEntity

    return CatalogEntity(
        entity_id=entity_id,
        name=name,
        aliases=(),
        area_id=area,
        area_name=area,
        floor_name=None,
        domain=domain,
        device_class=None,
        supported_features=0,
    )


def _resolve(area, domain, action, entities):
    from custom_components.typesafe_conversation.router import _resolve_area

    return _resolve_area(
        area, domain, spec_for(domain, action), {e.entity_id: e for e in entities}
    )


def test_an_area_with_a_match_keeps_the_hard_constraint():
    from custom_components.typesafe_conversation.router import AreaOutcome

    speakers = [_entity("media_player.a", "A", "kitchen", "media_player")]
    assert (
        _resolve("kitchen", "media_player", "search_and_play", speakers).outcome
        is AreaOutcome.USE_AREA
    )


def test_a_single_target_handler_prefers_the_area_instead():
    """search_and_play picks one target, so widening is safe."""
    from custom_components.typesafe_conversation.router import AreaOutcome

    speakers = [
        _entity("media_player.a", "A", "kitchen", "media_player"),
        _entity("media_player.b", "B", "bedroom", "media_player"),
    ]
    got = _resolve("upstairs", "media_player", "search_and_play", speakers)
    assert got.outcome is AreaOutcome.PREFER_AREA


def test_the_only_candidate_in_the_home_is_used():
    from custom_components.typesafe_conversation.router import AreaOutcome

    one = [_entity("light.only", "Only", "kitchen", "light")]
    got = _resolve("garage", "light", "turn_on", one)
    assert got.outcome is AreaOutcome.USE_ONLY_ENTITY
    assert got.entity.entity_id == "light.only"


def test_a_fan_out_command_asks_rather_than_widening():
    """The regression this whole change must not introduce.

    "turn off the lights" in a room with no lights must not become "turn off
    every light in the house". preferred_area_id would not even help here -
    Home Assistant ignores it for fan-out matches - so it would widen to all.
    """
    from custom_components.typesafe_conversation.router import AreaOutcome

    many = [
        _entity("light.a", "A", "kitchen", "light"),
        _entity("light.b", "B", "bedroom", "light"),
    ]
    assert _resolve("garage", "light", "turn_off", many).outcome is AreaOutcome.CLARIFY


async def test_preferred_area_is_sent_instead_of_area(hass: HomeAssistant):
    """The slot the router chose must be the slot that goes out."""
    await _setup(hass)
    from custom_components.typesafe_conversation.router import Plan, Route, Target

    plan = Plan(
        Route.COMMAND,
        domain="media_player",
        action="search_and_play",
        spec=spec_for("media_player", "search_and_play"),
        target=Target(domain="media_player"),
        preferred_area_id="upstairs",
    )
    slots = build_slots(plan.target, preferred_area_id=plan.preferred_area_id)
    assert slots["preferred_area_id"]["value"] == "upstairs"
    assert "area" not in slots, "a hard area is what fails; it must not be sent"


async def test_an_area_with_no_speaker_still_matches_a_player(hass: HomeAssistant):
    """End to end on the matching step, which is what actually broke.

    A satellite in an area holding no media_player asked for music. The old
    code sent area=<that area> as a hard constraint, async_match_targets found
    nothing, and it died as MatchFailedError before any service ran. With the
    area demoted to a preference the same request resolves to a real player.
    """
    await _setup(hass)
    from homeassistant.components.media_player import MediaPlayerEntityFeature as F

    areas = ar.async_get(hass)
    bedroom = areas.async_create("Bedroom")
    upstairs = areas.async_create("Upstairs")  # the satellite's area, no speaker
    registry = er.async_get(hass)
    speaker = registry.async_get_or_create(
        "media_player", "demo", "spk", suggested_object_id="bedroom_speaker"
    )
    registry.async_update_entity(speaker.entity_id, area_id=bedroom.id)
    hass.states.async_set(
        speaker.entity_id,
        "idle",
        {
            "friendly_name": "Bedroom Speaker",
            "supported_features": int(F.SEARCH_MEDIA | F.PLAY_MEDIA),
        },
    )
    _expose(hass, speaker.entity_id)

    from custom_components.typesafe_conversation.router import Plan, Route, Target

    plan = Plan(
        Route.COMMAND,
        domain="media_player",
        action="search_and_play",
        spec=spec_for("media_player", "search_and_play"),
        target=Target(domain="media_player"),
        preferred_area_id=upstairs.id,
        text_slot=("search_query", "jazz music"),
    )
    slots = build_slots(plan.target, preferred_area_id=plan.preferred_area_id)
    assert "area" not in slots

    # Exactly the constraints MediaSearchAndPlayHandler builds.
    constraints = intent.MatchTargetsConstraints(
        name=None,
        area_name=slots.get("area", {}).get("value"),
        domains={"media_player"},
        assistant=conversation.DOMAIN,
        features=F.SEARCH_MEDIA | F.PLAY_MEDIA,
        single_target=True,
    )
    result = intent.async_match_targets(
        hass,
        constraints,
        intent.MatchTargetsPreferences(area_id=slots["preferred_area_id"]["value"]),
    )
    assert result.is_match, "the old hard-area slot failed here"
    assert [s.entity_id for s in result.states] == [speaker.entity_id]


async def test_the_old_hard_area_slot_would_have_failed(hass: HomeAssistant):
    """Guard the premise: prove the thing we replaced really did fail."""
    await _setup(hass)
    from homeassistant.components.media_player import MediaPlayerEntityFeature as F

    areas = ar.async_get(hass)
    bedroom = areas.async_create("Bedroom")
    upstairs = areas.async_create("Upstairs")
    registry = er.async_get(hass)
    speaker = registry.async_get_or_create("media_player", "demo", "spk")
    registry.async_update_entity(speaker.entity_id, area_id=bedroom.id)
    hass.states.async_set(
        speaker.entity_id,
        "idle",
        {
            "friendly_name": "Bedroom Speaker",
            "supported_features": int(F.SEARCH_MEDIA | F.PLAY_MEDIA),
        },
    )
    _expose(hass, speaker.entity_id)

    result = intent.async_match_targets(
        hass,
        intent.MatchTargetsConstraints(
            area_name=upstairs.id,
            domains={"media_player"},
            assistant=conversation.DOMAIN,
            features=F.SEARCH_MEDIA | F.PLAY_MEDIA,
            single_target=True,
        ),
        intent.MatchTargetsPreferences(),
    )
    assert not result.is_match, "if this ever matches, the fix is unnecessary"


def test_area_resolution_ignores_a_dead_entity():
    """An unavailable entity must not count as the area's only candidate."""
    from custom_components.typesafe_conversation.router import (
        AreaOutcome,
        _resolve_area,
    )

    one_dead = [_entity("light.only", "Only", "kitchen", "light")]
    by_id = {e.entity_id: e for e in one_dead}
    spec = spec_for("light", "turn_on")

    alive = _resolve_area("garage", "light", spec, by_id, frozenset())
    assert alive.outcome is AreaOutcome.USE_ONLY_ENTITY

    dead = _resolve_area("garage", "light", spec, by_id, frozenset({"light.only"}))
    assert dead.outcome is AreaOutcome.CLARIFY, (
        "with its only candidate dead there is nothing to promote"
    )


async def test_a_query_against_an_unavailable_entity_still_runs(
    hass: HomeAssistant,
):
    """Commands and queries diverge here, on purpose.

    "Is the speaker playing?" against a dead entity has a correct answer -
    unavailable - so the query path must still dispatch. Only commands are
    blocked, because they cannot succeed.
    """
    await _setup(hass)
    registry = er.async_get(hass)
    entry = registry.async_get_or_create("media_player", "demo", "spk")
    hass.states.async_set(
        entry.entity_id, "unavailable", {"friendly_name": "Kitchen Speaker"}
    )
    _expose(hass, entry.entity_id)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    entity = next(e for e in catalog.entities if e.entity_id == entry.entity_id)
    from custom_components.typesafe_conversation.executor import (
        async_execute_query,
    )
    from custom_components.typesafe_conversation.router import Plan, Route, Target

    plan = Plan(
        Route.QUERY,
        query_kind="device_state",
        target=Target(entity=entity, domain="media_player"),
    )
    response = await async_execute_query(hass, plan, _input(hass))
    assert response.response_type is not intent.IntentResponseType.ERROR


# --- relative changes start from what the devices report ----------------------


def _device(entity_id: str, area_id: str | None = "kitchen", domain: str = "light"):
    from custom_components.typesafe_conversation.entities import CatalogEntity

    return CatalogEntity(
        entity_id=entity_id,
        name=entity_id,
        aliases=(),
        area_id=area_id,
        area_name=area_id,
        floor_name=None,
        domain=domain,
        device_class=None,
        supported_features=0,
    )


def _relative(action, target, step, domain="light", **extra) -> Plan:
    return Plan(
        Route.COMMAND,
        domain=domain,
        action=action,
        spec=spec_for(domain, action),
        target=target,
        relative_step=step,
        **extra,
    )


async def _sent(hass, plan, entities) -> dict[str, int]:
    """Run the plan and return the level each device was sent, by entity id."""
    from types import SimpleNamespace
    from unittest.mock import patch

    sent: dict[str, int] = {}

    async def _record(hass, intent_type, slots, user_input):
        value_slot = plan.spec.value_slot
        sent[slots["name"]["value"]] = slots[value_slot]["value"]
        return intent.IntentResponse(language="en")

    with patch("custom_components.typesafe_conversation.executor._handle", _record):
        await async_execute(
            hass, plan, _input(hass), SimpleNamespace(entities=entities)
        )
    return sent


async def test_brighter_on_an_off_light_starts_from_zero(hass: HomeAssistant):
    lamp = _device("light.lamp")
    hass.states.async_set("light.lamp", "off")
    sent = await _sent(hass, _relative("brighter", Target(entity=lamp), 25), [lamp])
    assert sent == {"light.lamp": 25}, "not 75, from an assumed 50%"


async def test_dimmer_on_an_off_light_says_so(hass: HomeAssistant):
    from custom_components.typesafe_conversation.executor import ExecutionError

    lamp = _device("light.lamp")
    hass.states.async_set("light.lamp", "off")
    with pytest.raises(ExecutionError, match="already off"):
        await _sent(hass, _relative("dimmer", Target(entity=lamp), -25), [lamp])


@pytest.mark.parametrize(
    ("state", "attributes"),
    [
        ("unavailable", {}),
        ("on", {"brightness": float("nan")}),
        ("on", {"brightness": True}),
        ("on", {}),
    ],
)
async def test_an_unknown_level_is_never_guessed(
    hass: HomeAssistant, state, attributes
):
    from custom_components.typesafe_conversation.executor import ExecutionError

    lamp = _device("light.lamp")
    hass.states.async_set("light.lamp", state, attributes)
    with pytest.raises(ExecutionError, match="say a value"):
        await _sent(hass, _relative("dimmer", Target(entity=lamp), -25), [lamp])


async def test_dimming_a_room_leaves_its_off_lights_off(hass: HomeAssistant):
    """ "It's too bright in here": each light that is on dims from its own level."""
    a, b = _device("light.a"), _device("light.b")
    off, hall = _device("light.off"), _device("light.hall", "hall")
    hass.states.async_set("light.a", "on", {"brightness": 255})
    hass.states.async_set("light.b", "on", {"brightness": 127.5})
    hass.states.async_set("light.off", "off")
    hass.states.async_set("light.hall", "on", {"brightness": 255})

    sent = await _sent(
        hass,
        _relative("dimmer", Target(area_id="kitchen", domain="light"), -25),
        [a, b, off, hall],
    )

    assert sent == {"light.a": 75, "light.b": 25}, (
        "the off light and the hall untouched"
    )


async def test_brightening_a_dark_room_turns_its_lights_on(hass: HomeAssistant):
    a, b = _device("light.a"), _device("light.b")
    hass.states.async_set("light.a", "off")
    hass.states.async_set("light.b", "off")
    sent = await _sent(
        hass,
        _relative("brighter", Target(area_id="kitchen", domain="light"), 25),
        [a, b],
    )
    assert sent == {"light.a": 25, "light.b": 25}


async def test_dimming_a_dark_room_says_so(hass: HomeAssistant):
    from custom_components.typesafe_conversation.executor import ExecutionError

    a = _device("light.a")
    hass.states.async_set("light.a", "off")
    with pytest.raises(ExecutionError, match="already off"):
        await _sent(
            hass,
            _relative("dimmer", Target(area_id="kitchen", domain="light"), -25),
            [a],
        )


async def test_the_speakers_room_is_a_preference_not_a_filter(hass: HomeAssistant):
    """From a room with no thermostat, "warmer" reaches the only one there is."""
    stat = _device("climate.stat", "hall", "climate")
    hass.states.async_set("climate.stat", "heat", {"temperature": 20})
    plan = _relative(
        "warmer", Target(domain="climate"), 10, "climate", preferred_area_id="kitchen"
    )
    assert await _sent(hass, plan, [stat]) == {"climate.stat": 21.0}


async def test_an_unknown_thermostat_setpoint_is_not_assumed(hass: HomeAssistant):
    from custom_components.typesafe_conversation.executor import ExecutionError

    stat = _device("climate.stat", None, "climate")
    hass.states.async_set("climate.stat", "heat", {})
    with pytest.raises(ExecutionError, match="say a value"):
        await _sent(
            hass, _relative("warmer", Target(entity=stat), 10, "climate"), [stat]
        )
