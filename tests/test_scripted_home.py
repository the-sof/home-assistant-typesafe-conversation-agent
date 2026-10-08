"""Routing against a script-heavy catalog.

`tests/fixtures/scripted_home.json` is a synthetic catalog built to hold the
structural properties that a light-heavy fixture does not exercise, because
each of them broke the router once:

* **eight `script` entities.** A script's action Choice offers only two options,
  `run` and `not_targeted`, and Jev derives confidence as
  ``(n * p_top - 1) / (n - 1)``. Two options at p=0.66 therefore score 0.31,
  where seven options at the same probability score 0.60. Gating the action on
  confidence made every script command fall back.
* **two climate zones.** `HassClimateSetTemperature` is `single_target=True`,
  so an unqualified "set the thermostat" has no single answer.
* **eleven entities in no area at all**, so area-based targeting cannot be
  assumed.
* **`todo` and `weather` domains**, which have no on/off semantics.
* **an area named like a floor, with no floor of that name**, which used to make
  a floor-scoped request dead-end.

The answers in `fixtures/scripted_answers/` are real jev-1.13.0 responses to
this catalog, recorded with `scripts/calibrate.py --record`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_components.typesafe_conversation.const import RISKY_ACTIONS
from custom_components.typesafe_conversation.entities import CatalogEntity
from custom_components.typesafe_conversation.extraction import extract
from custom_components.typesafe_conversation.router import Route, route
from custom_components.typesafe_conversation.system_one import (
    SystemOneResponse,
    _parse_answer,
)

FIXTURES = Path(__file__).parent / "fixtures"
ANSWERS = FIXTURES / "scripted_answers"


def _catalog():
    home = json.loads((FIXTURES / "scripted_home.json").read_text())
    names = {a["id"]: a["name"] for a in home["areas"]}
    floors = {a["id"]: a.get("floor") for a in home["areas"]}
    entities = tuple(
        CatalogEntity(
            entity_id=e["id"],
            name=e["name"],
            aliases=tuple(e["also"].split(", ")) if e.get("also") else (),
            area_id=e.get("area"),
            area_name=names.get(e.get("area")),
            floor_name=floors.get(e.get("area")),
            domain=e["domain"],
            device_class=(e.get("attrs") or {}).get("device_class"),
            supported_features=0,
        )
        for e in home["entities"]
    )
    catalog_floors = {a["id"]: a["floor"] for a in home["areas"] if a.get("floor")}
    return entities, catalog_floors


@pytest.fixture(name="scripted")
def scripted_fixture():
    entities, catalog_floors = _catalog()
    by_id = {e.entity_id: e for e in entities}
    domains = frozenset(e.domain for e in entities)

    def _route(slug: str):
        payload = json.loads((ANSWERS / f"{slug}.json").read_text())
        body = payload["response"]
        response = SystemOneResponse(
            model=body["model"],
            answers={k: _parse_answer(k, v) for k, v in body["answers"].items()},
            input_tokens=body.get("usage", {}).get("input_tokens", 0),
            output_tokens=0,
            latency_ms=0.0,
            raw=body,
        )
        utterance = payload["utterance"]
        return route(
            response,
            entities_by_id=by_id,
            extraction=extract(
                utterance,
                want_media="media_player" in domains,
                want_color="light" in domains,
            ),
            speaker_area_id=payload["spoken_from_area"],
            available_domains=domains,
            catalog_floors=catalog_floors,
        )

    return _route


DECIDED = (
    Route.COMMAND,
    Route.QUERY,
    Route.COMPOUND,
    Route.INFORMATION,
    Route.CONFIRM,
    Route.CLARIFY,
)


def test_the_fixture_still_has_the_properties_these_tests_rely_on():
    """Guard the fixture itself.

    Every test below depends on a structural property of the catalog rather
    than on a particular device, so a well-meaning edit that smooths the
    catalog out would silently stop exercising the bug it was built for.
    """
    entities, _ = _catalog()
    domains = [e.domain for e in entities]
    assert domains.count("script") >= 6, (
        "need a domain whose action Choice has 2 options"
    )
    ids = {e.entity_id for e in entities}
    assert {"script.security_disarm", "script.security_arm_home"} <= ids, (
        "the risky gate tests need a security script whose action is only 'run'"
    )
    # An area holding no media_player, so the area-resolution path is exercised.
    with_players = {e.area_id for e in entities if e.domain == "media_player"}
    all_areas = {e.area_id for e in entities if e.area_id}
    assert all_areas - with_players, (
        "need an area with no media_player for the widening test"
    )
    assert domains.count("climate") == 2, "need two zones for the single_target clash"
    assert sum(1 for e in entities if e.area_id is None) >= 10
    assert "todo" in domains and "weather" in domains

    home = json.loads((FIXTURES / "scripted_home.json").read_text())
    floor_like = [
        a for a in home["areas"] if a["name"].endswith("Level") and "floor" not in a
    ]
    assert floor_like, "need an area named like a floor but with no floor set"


@pytest.mark.parametrize("slug", sorted(p.stem for p in ANSWERS.glob("*.json")))
def test_nothing_dead_ends(slug, scripted):
    """Every recorded utterance reaches a decision, not the fallback ladder."""
    plan = scripted(slug)
    assert plan.route in DECIDED, f"{slug}: {plan.reason} / {plan.trace}"


def test_a_confident_routine_is_run_without_a_vote_on_the_action(scripted):
    """A routine can only be run; once the target is solid, run it.

    The action question still goes out, but for a single-action domain it can
    only lose a good answer, as it did when "wake me up at 5:15" picked the
    alarm script at 0.96 and then split "run" 45/55.
    """
    plan = scripted("start_the_bedtime_routine")
    assert plan.route is Route.COMMAND
    assert plan.domain == "script"
    assert plan.action == "run"
    assert plan.target.entity.entity_id == "script.routine_bedtime"
    assert plan.trace["action"] == "implied"


def test_list_items_come_from_a_span_not_the_model(scripted):
    plan = scripted("add_milk_to_the_shopping_list")
    assert plan.route is Route.COMMAND
    assert plan.domain == "todo"
    assert plan.action == "add_item"
    # "milk", not "add milk": the chunker strips the verb before Jev picks.
    assert plan.text_slot == ("item", "milk")
    assert plan.target.entity.entity_id == "todo.list_one"


def test_an_area_named_like_a_floor_still_resolves(scripted):
    """ "Upstairs" is an area here, and no floor carries that name.

    Jev reads the request as floor-scoped. Bailing out when the floor cannot be
    resolved dead-ended a perfectly good command; falling through to area
    targeting handles it.
    """
    plan = scripted("make_it_cooler_upstairs")
    assert plan.route is Route.COMMAND
    assert plan.domain == "climate"
    assert plan.action == "cooler"
    assert plan.relative_step is not None and plan.relative_step < 0


def test_a_door_counts_as_a_cover(scripted):
    """Home Assistant models doors, gates and windows as covers.

    The action question used to describe only blinds and garage doors, so a
    plain door came back `not_targeted` and the command fell back.
    """
    plan = scripted("open_the_side_door")
    assert plan.route in (Route.COMMAND, Route.CONFIRM)
    assert plan.domain == "cover"
    assert plan.action == "open"


def test_two_zones_are_not_silently_picked_between(scripted):
    """With no zone named, either ask or say which one was chosen."""
    plan = scripted("set_the_thermostat_to_70")
    assert plan.route in (Route.COMMAND, Route.CLARIFY)
    if plan.route is Route.COMMAND:
        assert plan.name_target_in_speech, "must name a target it is unsure of"
        assert plan.value == 70.0


def test_general_knowledge_reaches_the_llm(scripted):
    assert scripted("who_won_the_world_cup_in_1998").route is Route.INFORMATION
    # "what's the weather" sits near the category threshold; either the
    # information route or the fallback ladder ends up at the LLM.
    assert scripted("what_s_the_weather").route in (
        Route.INFORMATION,
        Route.FALLBACK,
    )


# --- the risky gate ----------------------------------------------------------
# A script's action is always "run", so the gate used to AND the risky question
# with an allowlist of risky *actions* and no script could ever reach it.
# Security actions are commonly implemented as scripts, so those ran
# unconfirmed.


@pytest.mark.parametrize(
    "slug",
    [
        "disarm_the_alarm",
        "open_the_driveway_gate",
        "unlock_the_side_door",
        "open_the_side_door",
    ],
)
def test_anything_that_reduces_security_asks_first(slug, scripted):
    plan = scripted(slug)
    assert plan.route is Route.CONFIRM, f"{slug} must not run unprompted"
    assert plan.trace["risky"] >= 0.5


def test_a_security_script_is_gated_even_though_its_action_is_run(scripted):
    """The exact hole: domain=script, action=run, but genuinely dangerous."""
    plan = scripted("disarm_the_alarm")
    assert plan.domain == "script"
    assert plan.action == "run", "a script's action carries no risk signal"
    assert plan.action not in RISKY_ACTIONS, (
        "if this ever becomes true the test has stopped proving anything"
    )
    # The question, not the action, is what catches it.
    assert plan.trace["risky"] >= 0.9
    assert plan.route is Route.CONFIRM


def test_arming_is_not_treated_as_risky(scripted):
    """Arming increases security, so it must not be gated.

    The mirror of the test above: the question has to separate the two, or the
    fix would just confirm everything.
    """
    plan = scripted("arm_the_alarm_in_home_mode")
    assert plan.route is Route.COMMAND
    assert plan.target.entity.entity_id == "script.security_arm_home"
    assert plan.trace["risky"] < 0.1


def test_ordinary_routines_are_not_gated(scripted):
    """Guard against the fix confirming everything.

    Twelve scripts in this catalog are ordinary routines. If the gate starts
    firing on those, confirmation fatigue makes it worthless.
    """
    benign = [
        "start_the_morning_routine",
        "run_the_evening_routine",
        "set_up_guest_mode",
        "start_movie_mode",
        "start_party_mode",
        "start_the_bedtime_routine",
        "switch_to_away_mode",
        "turn_on_the_study_lamp",
        "add_milk_to_the_shopping_list",
        "start_the_robot_cleaner",
    ]
    for slug in benign:
        plan = scripted(slug)
        assert plan.route is not Route.CONFIRM, f"{slug} should not need confirming"
        assert plan.trace["risky"] < 0.5, slug


def test_the_confirmation_names_the_script_readably(scripted):
    from custom_components.typesafe_conversation.agent import _confirm_question

    plan = scripted("disarm_the_alarm")
    question = _confirm_question(plan)
    assert question == "Do you want me to run Disarm the alarm?"
    assert "run the Disarm" not in question


def test_an_area_with_no_player_widens_instead_of_failing(scripted):
    """ "Play jazz in the workshop" - and the workshop has no speaker.

    Sending area=workshop as a hard constraint is a guaranteed
    MatchFailedError. search_and_play is single-target, so the area becomes a
    preference and Home Assistant picks a real player, still favouring the
    workshop if one ever appears there.
    """
    plan = scripted("play_jazz_in_the_workshop")
    assert plan.route is Route.COMMAND
    assert plan.domain == "media_player"
    assert plan.action == "search_and_play"
    assert plan.text_slot == ("search_query", "jazz")
    assert plan.target.area_id is None, "a hard area here is what fails"
    assert plan.preferred_area_id == "area_two"
    assert plan.trace["area_resolution"] == "prefer_area"
