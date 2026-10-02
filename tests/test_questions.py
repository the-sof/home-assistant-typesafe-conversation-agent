"""The question set: validity, caps, and what it costs."""

from __future__ import annotations

import pytest
from conftest import build_catalog

from custom_components.typesafe_conversation.const import MAX_CHOICE_OPTIONS
from custom_components.typesafe_conversation.entities import CatalogEntity
from custom_components.typesafe_conversation.extraction import extract
from custom_components.typesafe_conversation.questions import (
    Q_COLOR_PICK,
    Q_TARGET_DOMAIN,
    Q_TARGET_ENTITY,
    QuestionSetError,
    build_questions,
    estimate_tokens,
    validate_questions,
)


def _build(**kwargs):
    entities, areas = build_catalog()
    domains = tuple(sorted({e.domain for e in entities}))
    return build_questions(
        entities=entities,
        areas=areas,
        domains=domains,
        extraction=kwargs.pop(
            "extraction", extract("", want_media=False, want_color=False)
        ),
        **kwargs,
    )


def test_every_question_is_valid():
    validate_questions(_build())


def test_entity_options_are_ids_with_an_escape_hatch():
    questions = _build()
    entities, _ = build_catalog()
    criteria = questions["target_entity"]["criteria"]
    for entity in entities:
        assert entity.entity_id in criteria
    assert "no_single_entity" in criteria, "the model must be able to say 'none'"
    # Null descriptions: the detail already lives in `home.entities`.
    assert criteria[entities[0].entity_id] is None


def test_inline_descriptions_are_opt_in_and_cost_more():
    lean = estimate_tokens(_build())
    rich = estimate_tokens(_build(inline_descriptions=True))
    assert rich > lean


def test_one_action_question_per_present_domain():
    questions = _build()
    entities, _ = build_catalog()
    domains = {e.domain for e in entities}
    assert "action_light" in questions
    assert "action_lock" in questions
    # A home with no humidifier pays nothing for humidifier questions.
    assert "humidifier" not in domains
    assert "action_humidifier" not in questions
    for key, question in questions.items():
        if key.startswith("action_"):
            assert "not_targeted" in question["criteria"], key


def test_conditional_questions_only_appear_when_earned():
    assert "value_pick" not in _build()
    with_value = _build(
        extraction=extract("set it to 30%", want_media=True, want_color=True)
    )
    assert "value_pick" in with_value
    assert "30%" in with_value["value_pick"]["criteria"]


def _lights(count: int) -> tuple[CatalogEntity, ...]:
    return tuple(
        CatalogEntity(
            entity_id=f"light.l{i}",
            name=f"Light {i}",
            aliases=(),
            area_id=None,
            area_name=None,
            floor_name=None,
            domain="light",
            device_class=None,
            supported_features=0,
        )
        for i in range(count)
    )


def test_a_home_over_the_cap_leaves_the_entity_for_a_second_request():
    """Ollama caps a Choice at 26 options; a 30-entity home can't be offered whole.

    The entity question is left for the second stage, while the questions that
    narrow it down - domain, area - are still asked in the first.
    """
    built = build_questions(
        entities=_lights(30),
        areas=(),
        domains=("light",),
        extraction=extract("", want_media=False, want_color=False),
        max_options=26,
    )
    assert Q_TARGET_ENTITY not in built
    assert Q_TARGET_DOMAIN in built


def test_a_home_within_the_cap_is_asked_in_one_go():
    built = build_questions(
        entities=_lights(25),
        areas=(),
        domains=("light",),
        extraction=extract("", want_media=False, want_color=False),
        max_options=26,
    )
    assert len(built[Q_TARGET_ENTITY]["criteria"]) == 26, "25 lights + the escape"


def test_choice_cap_is_enforced():
    oversized = {
        "x": {
            "type": "choice",
            "instructions": "pick one",
            "criteria": {f"o{i}": None for i in range(27)},
        }
    }
    with pytest.raises(QuestionSetError, match="exceeds this server's cap of 26"):
        validate_questions(oversized, max_options=26)
    validate_questions(oversized, max_options=MAX_CHOICE_OPTIONS)


def test_a_small_cap_offers_only_the_colours_that_were_said():
    """Every colour is 58 options; a 26-option server only needs those named."""
    extraction = extract("make it red", want_media=False, want_color=True)
    built = build_questions(
        entities=_lights(3),
        areas=(),
        domains=("light",),
        extraction=extraction,
        max_options=26,
    )
    options = set(built[Q_COLOR_PICK]["criteria"])
    assert "red" in options
    assert "blue" not in options
    assert {"warm_white", "daylight"} <= options, "descriptive requests still map"
    assert len(options) <= 26

    roomy = build_questions(
        entities=_lights(3),
        areas=(),
        domains=("light",),
        extraction=extraction,
    )
    assert "blue" in roomy[Q_COLOR_PICK]["criteria"], "TypeSafe still gets them all"


def test_the_whole_request_fits_the_context_budget():
    """Jev allows 64k per request and 32k for state plus the longest question."""
    from conftest import load_home

    questions = _build()
    state_tokens = estimate_tokens({"home": load_home()})
    longest = max(estimate_tokens(q) for q in questions.values())
    total = state_tokens + estimate_tokens(questions)
    assert state_tokens + longest < 30_000
    assert total < 60_000


def test_validator_rejects_a_malformed_question():
    with pytest.raises(QuestionSetError, match="instructions"):
        validate_questions({"x": {"type": "noul", "instructions": ""}})
    with pytest.raises(QuestionSetError, match="2-10 levels"):
        validate_questions(
            {"x": {"type": "score", "instructions": "how much", "criteria": ["one"]}}
        )
    with pytest.raises(QuestionSetError, match="unknown question type"):
        validate_questions({"x": {"type": "vibe", "instructions": "hmm"}})


def test_every_offered_action_has_a_spec():
    """The two tables must not drift.

    An action offered to Jev that no spec can route is a dead end: the model
    picks it, the router finds nothing, and the request falls back for no
    visible reason.
    """
    from custom_components.typesafe_conversation.actions import spec_for
    from custom_components.typesafe_conversation.questions import _ACTION_CRITERIA

    missing = [
        (domain, action)
        for domain, actions in _ACTION_CRITERIA.items()
        for action in actions
        if spec_for(domain, action) is None
    ]
    assert not missing


def test_no_media_action_routes_through_a_power_intent():
    """Media players routinely expose no power at all.

    A software endpoint - a streaming or cast player - commonly supports STOP
    and PAUSE but not TURN_OFF, so HassTurnOff is rejected by the entity. The
    rejection is easy to miss because Home Assistant counts the matched area
    as a success, so the response still reports action_done.
    """
    from custom_components.typesafe_conversation.actions import ACTIONS
    from custom_components.typesafe_conversation.questions import _ACTION_CRITERIA

    power = {"HassTurnOn", "HassTurnOff"}
    routed = {
        action: spec.intent_type
        for (domain, action), spec in ACTIONS.items()
        if domain == "media_player"
    }
    offered = _ACTION_CRITERIA["media_player"]
    assert not {a for a in offered if routed.get(a) in power}
    # Home Assistant has no stop-media intent, so both words mean pause.
    assert routed["stop"] == routed["pause"] == "HassMediaPause"
    # ... and only one of them is offered, or they would split the vote.
    assert "stop" not in _ACTION_CRITERIA["media_player"]
