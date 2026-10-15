"""Replay recorded Jev answers through the router.

Every fixture here is a real response from jev-1.13.0, and the main cases are
replayed from jev-1.14.0 too, against ``tests/fixtures/home.json``. If a
threshold in const.py changes, or the model does, these tests say exactly which
utterances start behaving differently.
"""

from __future__ import annotations

import pytest
from conftest import ANSWERS, ANSWERS_1_14, load_response

from custom_components.typesafe_conversation.extraction import extract
from custom_components.typesafe_conversation.router import Route, route
from custom_components.typesafe_conversation.system_one import ChoiceAnswer


def _route(name: str, entities_by_id, available_domains, answers=ANSWERS):
    response, payload = load_response(name, answers)
    utterance = payload["utterance"]
    extraction = extract(
        utterance,
        want_media="media_player" in available_domains,
        want_color="light" in available_domains,
    )
    plan = route(
        response,
        entities_by_id=entities_by_id,
        extraction=extraction,
        speaker_area_id=payload["spoken_from_area"],
        available_domains=available_domains,
    )
    return plan, utterance


# (fixture, expected route, expected domain, expected action)
CASES = [
    ("get_the_coffee_boiling", Route.COMMAND, "switch", "turn_on"),
    ("it_s_too_bright_in_here", Route.COMMAND, "light", "dimmer"),
    ("set_the_living_room_lights_to_30", Route.COMMAND, "light", "set_brightness"),
    ("make_the_lights_a_bit_warmer", Route.COMMAND, "light", "set_color"),
    ("set_the_thermostat_to_21_degrees", Route.COMMAND, "climate", "set_temperature"),
    ("turn_the_volume_down_a_bit", Route.COMMAND, "media_player", "quieter"),
    ("unlock_the_front_door", Route.CONFIRM, "lock", "unlock"),
    ("play_some_jazz_in_the_kitchen", Route.COMMAND, "media_player", "search_and_play"),
    ("goodnight", Route.COMMAND, "script", "run"),
    ("turn_off_all_the_lights_and_lock_the_front_door", Route.COMPOUND, None, None),
    ("dim_the_bedroom_lamp_and_start_the_dishwasher", Route.COMPOUND, None, None),
    ("is_the_garage_door_open", Route.QUERY, None, None),
    ("what_s_the_temperature_in_the_bedroom", Route.QUERY, None, None),
    ("how_many_lights_are_on", Route.QUERY, None, None),
    ("what_time_is_it", Route.QUERY, None, None),
    ("is_everything_locked_up", Route.QUERY, None, None),
    ("who_won_the_world_cup_in_1998", Route.INFORMATION, None, None),
    ("never_mind", Route.CANCEL, None, None),
    ("asdfgh", Route.FALLBACK, None, None),
]


MODELS = pytest.mark.parametrize(
    "answers", [ANSWERS, ANSWERS_1_14], ids=["jev-1.13", "jev-1.14"]
)


@MODELS
@pytest.mark.parametrize(("name", "expected", "domain", "action"), CASES)
def test_routes(
    name, expected, domain, action, answers, entities_by_id, available_domains
):
    plan, utterance = _route(name, entities_by_id, available_domains, answers)
    assert plan.route is expected, f"{utterance!r}: {plan.reason} / {plan.trace}"
    if domain is not None:
        assert plan.domain == domain, utterance
    if action is not None:
        assert plan.action == action, utterance


def test_coffee_maker_resolves_to_the_exact_entity(entities_by_id, available_domains):
    plan, _ = _route("get_the_coffee_boiling", entities_by_id, available_domains)
    assert plan.target.entity is not None
    assert plan.target.entity.entity_id == "switch.coffee_maker"
    # High confidence: no need to spell the target out loud.
    assert plan.name_target_in_speech is False


def test_absolute_brightness_carries_the_value(entities_by_id, available_domains):
    plan, _ = _route(
        "set_the_living_room_lights_to_30", entities_by_id, available_domains
    )
    assert plan.value == 30.0
    assert plan.value_unit == "%"
    assert plan.target.area_id == "living_room"


def test_thermostat_value_and_unit(entities_by_id, available_domains):
    plan, _ = _route(
        "set_the_thermostat_to_21_degrees", entities_by_id, available_domains
    )
    assert plan.value == 21.0
    # "degrees" with no letter: the executor resolves the unit from the entity,
    # never from the model.
    assert plan.value_unit == "deg"
    assert plan.spec is not None and plan.spec.single_target


def test_relative_change_becomes_a_signed_step(entities_by_id, available_domains):
    plan, _ = _route("turn_the_volume_down_a_bit", entities_by_id, available_domains)
    assert plan.relative_step == -10, "'a bit' is the slight magnitude"
    plan, _ = _route("it_s_too_bright_in_here", entities_by_id, available_domains)
    assert plan.relative_step is not None and plan.relative_step < 0


def test_warmer_light_becomes_a_colour_temperature(entities_by_id, available_domains):
    plan, _ = _route("make_the_lights_a_bit_warmer", entities_by_id, available_domains)
    assert plan.color_temp_kelvin == 2700
    assert plan.text_slot is None


def test_media_search_query_is_a_verbatim_span(entities_by_id, available_domains):
    plan, _ = _route("play_some_jazz_in_the_kitchen", entities_by_id, available_domains)
    assert plan.text_slot == ("search_query", "some jazz")

    # This fixture has no media_player in the kitchen - both live in the living
    # room - so a hard `area` slot would be a guaranteed MatchFailedError.
    # search_and_play is single-target, so the area becomes a preference and
    # Home Assistant picks a real player while still favouring the kitchen.
    assert plan.target.area_id is None
    assert plan.preferred_area_id == "kitchen"
    assert plan.trace["area_resolution"] == "prefer_area"


def test_unlocking_asks_before_acting(entities_by_id, available_domains):
    """A risky action needs more than ordinary confidence to happen silently."""
    plan, _ = _route("unlock_the_front_door", entities_by_id, available_domains)
    assert plan.route is Route.CONFIRM
    assert plan.trace["risky"] > 0.9


@MODELS
def test_ambiguous_target_asks_which_one(answers, entities_by_id, available_domains):
    plan, _ = _route(
        "turn_on_the_thing_in_the_corner", entities_by_id, available_domains, answers
    )
    assert plan.route in (Route.CLARIFY, Route.FALLBACK)
    if plan.route is Route.CLARIFY:
        assert len(plan.options) == 2


@MODELS
def test_whole_house_command_never_lacks_a_target(
    answers, entities_by_id, available_domains
):
    """'turn off everything' names no domain.

    An intent with no name/area/floor/domain constraint raises
    IntentHandleError, so the plan must say whole_house and let the executor
    fan out over the domains that exist.
    """
    plan, _ = _route("turn_off_everything", entities_by_id, available_domains, answers)
    if plan.route is Route.COMMAND:
        assert plan.target.whole_house is True
        assert plan.action == "turn_off"


def test_query_kinds(entities_by_id, available_domains):
    for name, kind in [
        ("is_the_garage_door_open", "device_state"),
        ("what_s_the_temperature_in_the_bedroom", "temperature"),
        ("how_many_lights_are_on", "count"),
        ("what_time_is_it", "time_or_date"),
        ("is_everything_locked_up", "needs_prose"),
    ]:
        plan, utterance = _route(name, entities_by_id, available_domains)
        assert plan.query_kind == kind, f"{utterance!r} -> {plan.query_kind}"


def test_unread_branches_are_ignored(entities_by_id, available_domains):
    """'is everything locked up' is a query.

    action_lock comes back "lock" at 0.99 - a confident answer to a question
    nobody asked. The router must not act on it. This is the speculative
    fan-out contract: wrong answers on unread branches are free.
    """
    response, _ = load_response("is_everything_locked_up")
    assert response.choice("action_lock").choice == "lock"
    plan, _ = _route("is_everything_locked_up", entities_by_id, available_domains)
    assert plan.route is Route.QUERY
    assert plan.action is None


# --- targets that exist but cannot act ---------------------------------------
# route() is pure, so all of this replays the *recorded* answers with a
# synthetic availability set. No new API calls, and the model's real
# probability distribution does the ranking.


def _route_with(
    name, entities_by_id, available_domains, *, dead=frozenset(), override_entity=None
):
    response, payload = load_response(name)
    if override_entity is not None:
        response.answers["target_entity"] = override_entity
    utterance = payload["utterance"]
    return route(
        response,
        entities_by_id=entities_by_id,
        extraction=extract(
            utterance,
            want_media="media_player" in available_domains,
            want_color="light" in available_domains,
        ),
        speaker_area_id=payload["spoken_from_area"],
        available_domains=available_domains,
        unavailable_ids=dead,
    )


def test_an_unavailable_target_is_named_not_dispatched(
    entities_by_id, available_domains
):
    """The live failure: dispatching produced a service error naming nothing."""
    plan = _route_with(
        "make_the_lights_a_bit_warmer",
        entities_by_id,
        available_domains,
        dead=frozenset({"light.kitchen_ceiling", "light.kitchen_under_cabinet"}),
    )
    assert plan.route is Route.UNAVAILABLE
    assert plan.speech == "Kitchen Ceiling is unavailable."
    assert plan.trace["unavailable_target"] == "light.kitchen_ceiling"


def test_a_weak_runner_up_is_not_promoted(entities_by_id, available_domains):
    """The only other light scores 0.01 - too weak to silently act on."""
    plan = _route_with(
        "make_the_lights_a_bit_warmer",
        entities_by_id,
        available_domains,
        dead=frozenset({"light.kitchen_ceiling"}),
    )
    assert plan.route is Route.UNAVAILABLE
    assert "substituted_for" not in plan.trace


def test_a_credible_alternative_is_used_and_named(entities_by_id, available_domains):
    """When the distribution offers a real second choice, use it - and say so."""
    plan = _route_with(
        "make_the_lights_a_bit_warmer",
        entities_by_id,
        available_domains,
        dead=frozenset({"light.kitchen_ceiling"}),
        override_entity=ChoiceAnswer(
            choice="light.kitchen_ceiling",
            probabilities={
                "light.kitchen_ceiling": 0.55,
                "light.kitchen_under_cabinet": 0.40,
                "no_single_entity": 0.05,
            },
            confidence=0.80,
        ),
    )
    assert plan.route is Route.COMMAND
    assert plan.target.entity.entity_id == "light.kitchen_under_cabinet"
    assert plan.trace["substituted_for"] == "light.kitchen_under_cabinet"
    assert plan.name_target_in_speech, "we used a different device; say which"


def test_availability_changes_nothing_when_everything_is_alive(
    entities_by_id, available_domains
):
    """No behaviour drift for the ordinary case."""
    before = _route_with("get_the_coffee_boiling", entities_by_id, available_domains)
    after = _route_with(
        "get_the_coffee_boiling",
        entities_by_id,
        available_domains,
        dead=frozenset({"light.bedroom_ceiling"}),  # unrelated entity
    )
    assert before.route is after.route is Route.COMMAND
    assert before.target.entity.entity_id == after.target.entity.entity_id
    assert "unavailable_target" not in after.trace


# --- the room the request was spoken in ------------------------------------
#
# The shape these cover: a request that names one thing but no device and no
# room - "play that album", "turn it on". scope=single says a single target is
# meant, target_entity cannot say which, target_area stays no_area, and
# here_relative is low because the wording has no locative and no device word
# for it to be relative about. Nothing in the answer points at a target; only
# the room the request was spoken in does.


def _answers(**overrides):
    """A minimal command answer set. Overrides replace whole answers."""

    def choice(pick, probabilities, confidence):
        return {
            "type": "choice",
            "choice": pick,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    answers = {
        "category": choice("command", {"command": 0.98, "information": 0.02}, 0.96),
        "compound": {"type": "noul", "noul": 0.05},
        "query_kind": choice("not_a_query", {"not_a_query": 1.0}, 1.0),
        "here_relative": {"type": "noul", "noul": 0.16},
        "risky": {"type": "noul", "noul": 0.05},
        "change_direction": choice("no_numeric", {"no_numeric": 1.0}, 1.0),
        "magnitude": choice("not_applicable", {"not_applicable": 1.0}, 1.0),
        "scope": choice("single", {"single": 0.88, "not_a_target": 0.12}, 0.85),
        "target_area": choice("no_area", {"no_area": 0.56, "bedroom": 0.44}, 0.49),
        "target_domain": choice("light", {"light": 0.9, "none": 0.1}, 0.85),
        "target_entity": choice(
            "no_single_entity",
            {"no_single_entity": 0.49, "light.office_desk": 0.40, "light.hall": 0.11},
            0.45,
        ),
        "action_light": choice("turn_on", {"turn_on": 0.9, "turn_off": 0.1}, 0.88),
    }
    answers.update(overrides)
    return answers


def _plan(entities_by_id, available_domains, *, spoken_from, **overrides):
    from custom_components.typesafe_conversation.system_one import (
        SystemOneResponse,
        _parse_answer,
    )

    body = _answers(**overrides)
    response = SystemOneResponse(
        model="jev-1.13.0",
        answers={k: _parse_answer(k, v) for k, v in body.items()},
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        raw={"answers": body},
    )
    return route(
        response,
        entities_by_id=entities_by_id,
        extraction=extract("turn it on", want_media=False, want_color=False),
        speaker_area_id=spoken_from,
        available_domains=available_domains,
    )


def test_a_room_holding_one_candidate_is_the_default_target(
    entities_by_id, available_domains
):
    """Nothing named, spoken in the office, which has exactly one light."""
    plan = _plan(entities_by_id, available_domains, spoken_from="office")

    assert plan.route is Route.COMMAND
    assert plan.target.area_id == "office"
    assert plan.trace["speaker_area_default"] == "office"


def test_a_room_holding_several_is_not_silently_acted_on(
    entities_by_id, available_domains
):
    """The bedroom has two lights and scope=single says one was meant.

    Widening to the area would switch on both, which is worse than admitting
    we do not know - so the default must not apply here.
    """
    plan = _plan(entities_by_id, available_domains, spoken_from="bedroom")

    assert plan.route is not Route.COMMAND
    assert "speaker_area_default" not in plan.trace


def test_the_default_needs_a_known_room(entities_by_id, available_domains):
    """A request with no satellite behind it has no room to fall back on."""
    plan = _plan(entities_by_id, available_domains, spoken_from=None)

    assert plan.route is not Route.COMMAND


# --- what a clarification may offer ----------------------------------------


def _entity_answer(probabilities, confidence):
    return {
        "type": "choice",
        "choice": "no_single_entity",
        "confidence": confidence,
        "probabilities": probabilities,
    }


def test_a_clarification_only_offers_the_domain_that_was_chosen(
    entities_by_id, available_domains
):
    """The regression: a request to play music was answered with a vacuum.

    target_entity ranks every exposed entity, so the runner-up is whatever
    sorted highest overall - the domain is already decided by this point and
    must be applied.
    """
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="bedroom",
        target_domain={
            "type": "choice",
            "choice": "media_player",
            "confidence": 0.6,
            "probabilities": {"media_player": 0.65, "none": 0.31},
        },
        action_media_player={
            "type": "choice",
            "choice": "search_and_play",
            "confidence": 0.82,
            "probabilities": {"search_and_play": 0.84, "play": 0.10},
        },
        target_entity=_entity_answer(
            {
                "no_single_entity": 0.40,
                "media_player.living_room_speaker": 0.22,
                "vacuum.downstairs": 0.20,
                "media_player.tv": 0.18,
            },
            0.45,
        ),
    )

    assert plan.route is Route.CLARIFY
    offered = {entity_id for entity_id, _ in plan.options}
    assert offered == {"media_player.living_room_speaker", "media_player.tv"}


def test_a_clear_leader_is_not_dressed_up_as_a_choice(
    entities_by_id, available_domains
):
    """0.40 against 0.04 is one candidate and some noise, not two options.

    Spoken from the living room, which holds two lights - so the speaker-area
    default does not apply and this really does reach the clarify decision.
    """
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="living_room",
        target_entity=_entity_answer(
            {
                "no_single_entity": 0.49,
                "light.office_desk": 0.40,
                "light.hall": 0.04,
                "light.bathroom": 0.03,
            },
            0.45,
        ),
    )

    assert plan.route is not Route.CLARIFY


def test_a_genuine_tie_is_still_asked_about(entities_by_id, available_domains):
    """Two lights the model cannot separate: the question is worth asking."""
    # Again from a room holding several, so the default cannot pre-empt it.
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="living_room",
        target_entity=_entity_answer(
            {
                "no_single_entity": 0.60,
                "light.office_desk": 0.14,
                "light.hall": 0.08,
                "light.bathroom": 0.01,
            },
            0.45,
        ),
    )

    assert plan.route is Route.CLARIFY
    assert {entity_id for entity_id, _ in plan.options} == {
        "light.office_desk",
        "light.hall",
    }


# --- whole house, or one device that serves it --------------------------------
#
# jev-1.14 scores a whole-house scope for a device that serves the whole house:
# the alarm, the water main. These shapes follow its answers; a whole-house
# target must only ever come from a request about every device of a kind.


def _choice(pick, probabilities, confidence):
    return {
        "type": "choice",
        "choice": pick,
        "confidence": confidence,
        "probabilities": probabilities,
    }


WHOLE_HOUSE = _choice("whole_house", {"whole_house": 0.69, "single": 0.30}, 0.61)


def test_a_solid_device_beats_a_whole_house_scope(entities_by_id, available_domains):
    """ "Shut off the water": every switch must not go off with the valve."""
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="kitchen",
        scope=WHOLE_HOUSE,
        target_domain=_choice("switch", {"switch": 0.96, "none": 0.04}, 0.95),
        target_entity=_choice(
            "switch.dishwasher",
            {"switch.dishwasher": 0.68, "no_single_entity": 0.32},
            0.67,
        ),
        action_switch=_choice("turn_off", {"turn_off": 0.89, "toggle": 0.06}, 0.85),
    )

    assert plan.route is Route.COMMAND
    assert plan.target.entity.entity_id == "switch.dishwasher"
    assert plan.target.whole_house is False


def test_a_routine_it_cannot_pin_down_is_not_run_everywhere(
    entities_by_id, available_domains
):
    """ "Arm the alarm" leaning to one routine must not run every script."""
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="kitchen",
        scope=_choice("whole_house", {"whole_house": 0.75, "single": 0.25}, 0.69),
        target_domain=_choice("script", {"script": 0.99, "none": 0.01}, 0.99),
        target_entity=_choice(
            "script.bedtime",
            {"script.bedtime": 0.51, "no_single_entity": 0.44},
            0.49,
        ),
        action_script=_choice("run", {"run": 0.65, "not_targeted": 0.35}, 0.30),
    )

    assert plan.route is not Route.COMMAND
    assert plan.target.whole_house is False


def test_scripts_never_fan_out_to_the_whole_house(entities_by_id, available_domains):
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="kitchen",
        scope=_choice("whole_house", {"whole_house": 0.95, "single": 0.05}, 0.94),
        target_domain=_choice("script", {"script": 0.99, "none": 0.01}, 0.99),
        target_entity=_choice(
            "no_single_entity",
            {"no_single_entity": 0.95, "script.bedtime": 0.05},
            0.94,
        ),
        action_script=_choice("run", {"run": 0.9, "not_targeted": 0.1}, 0.8),
    )

    assert plan.route is not Route.COMMAND
    assert plan.target.whole_house is False


def test_a_device_it_cannot_pin_down_is_not_everything(
    entities_by_id, available_domains
):
    """With no domain either, leaning to one switch must not turn off the house."""
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="kitchen",
        scope=WHOLE_HOUSE,
        target_domain=_choice("switch", {"switch": 0.5, "none": 0.5}, 0.3),
        target_entity=_choice(
            "switch.dishwasher",
            {"switch.dishwasher": 0.5, "no_single_entity": 0.45},
            0.45,
        ),
        action_switch=_choice("turn_off", {"turn_off": 0.89, "toggle": 0.06}, 0.85),
    )

    assert plan.route is not Route.COMMAND
    assert plan.target.whole_house is False


def test_a_whole_house_device_is_not_taken_from_the_speakers_room(
    entities_by_id, available_domains
):
    """The office's only lock is not the house lock that was half-named."""
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="office",
        scope=WHOLE_HOUSE,
        target_domain=_choice("lock", {"lock": 0.97, "none": 0.03}, 0.96),
        target_entity=_choice(
            "lock.front_door",
            {"lock.front_door": 0.5, "no_single_entity": 0.45},
            0.45,
        ),
        action_lock=_choice("lock", {"lock": 0.95, "unlock": 0.05}, 0.92),
    )

    assert plan.route is not Route.COMMAND
    assert plan.target.area_id != "office"
    assert plan.reason.startswith("whole_house cannot fan out")


def test_every_light_still_means_the_whole_house(entities_by_id, available_domains):
    """ "Turn off all the lights" names no one light, so it still fans out."""
    plan = _plan(
        entities_by_id,
        available_domains,
        spoken_from="kitchen",
        scope=_choice("whole_house", {"whole_house": 0.95, "single": 0.05}, 0.94),
        target_entity=_choice(
            "no_single_entity",
            {"no_single_entity": 0.97, "light.kitchen_ceiling": 0.03},
            0.96,
        ),
        action_light=_choice("turn_off", {"turn_off": 0.95, "turn_on": 0.05}, 0.92),
    )

    assert plan.route is Route.COMMAND
    assert plan.target.whole_house is True


@MODELS
def test_a_house_wide_count_ignores_the_room_it_was_asked_in(
    answers, entities_by_id, available_domains
):
    """jev-1.14 scores the kitchen as the area; the scope says the house."""
    response, payload = load_response("how_many_lights_are_on", answers)
    response.answers["target_area"] = ChoiceAnswer(
        choice="kitchen",
        probabilities={"kitchen": 0.67, "no_area": 0.33},
        confidence=0.63,
    )
    plan = route(
        response,
        entities_by_id=entities_by_id,
        extraction=extract(payload["utterance"], want_media=True, want_color=True),
        speaker_area_id=payload["spoken_from_area"],
        available_domains=available_domains,
    )

    assert plan.route is Route.QUERY
    assert plan.query_kind == "count"
    assert plan.target.area_id is None
