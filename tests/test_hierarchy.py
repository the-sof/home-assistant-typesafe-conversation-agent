"""The second stage for homes larger than the server's option cap.

Pure functions, like route(): answers and catalogue in, plan out.
"""

from __future__ import annotations

import pytest

from custom_components.typesafe_conversation import questions as Q
from custom_components.typesafe_conversation.entities import CatalogEntity
from custom_components.typesafe_conversation.hierarchy import (
    entity_stage_questions,
    merge_entity_stage,
    needs_entity_stage,
    plan_entity_stage,
)
from custom_components.typesafe_conversation.system_one import (
    ChoiceAnswer,
    SystemOneResponse,
)

CAP = 26


def _entity(entity_id: str, area: str | None) -> CatalogEntity:
    return CatalogEntity(
        entity_id=entity_id,
        name=entity_id.split(".")[1].replace("_", " ").title(),
        aliases=(),
        area_id=area,
        area_name=area,
        floor_name=None,
        domain=entity_id.split(".")[0],
        device_class=None,
        supported_features=0,
    )


# 45 entities, 29 of them lights: over a 26-option cap either way.
HOME = (
    _entity("light.bedroom_ceiling", "bedroom"),
    _entity("light.bedside_lamp", "bedroom"),
    _entity("light.living_room_ceiling", "living_room"),
    _entity("light.floor_lamp", "living_room"),
    _entity("switch.bedroom_fan", "bedroom"),
    *(_entity(f"light.hall_{i}", "hall") for i in range(25)),
    *(_entity(f"switch.garage_{i}", "garage") for i in range(15)),
)


def _choice(probabilities: dict[str, float]) -> ChoiceAnswer:
    top = max(probabilities, key=probabilities.__getitem__)
    return ChoiceAnswer(top, probabilities, 0.9)


def _response(**answers: ChoiceAnswer) -> SystemOneResponse:
    defaults = {
        Q.Q_CATEGORY: _choice({"command": 0.97, "query": 0.03}),
        Q.Q_SCOPE: _choice({"single": 0.9, "area": 0.1}),
        Q.Q_TARGET_DOMAIN: _choice({"light": 0.93, "switch": 0.04, "none": 0.03}),
        Q.Q_TARGET_AREA: _choice(
            {"bedroom": 0.55, "living_room": 0.30, "hall": 0.05, "no_area": 0.10}
        ),
    }
    return SystemOneResponse(
        model="nimble",
        answers={**defaults, **answers},
        input_tokens=100,
        output_tokens=5,
        latency_ms=40.0,
    )


def test_one_specific_device_needs_the_second_stage():
    assert needs_entity_stage(_response())


@pytest.mark.parametrize("scope", ["whole_house", "floor"])
def test_a_solid_group_command_never_needs_it(scope):
    """'Turn off all the lights' acts on many; there is no one entity to pick."""
    solid = ChoiceAnswer(scope, {scope: 0.95, "single": 0.05}, 0.94)
    assert not needs_entity_stage(_response(**{Q.Q_SCOPE: solid}))


def test_a_shaky_group_scope_still_asks_for_the_entity():
    """'Turn off the floor lamp' can pull scope toward 'floor'. route() only
    trusts a solid scope and otherwise checks the entity, so it must be asked."""
    shaky = ChoiceAnswer("floor", {"floor": 0.5, "single": 0.45, "area": 0.05}, 0.4)
    assert needs_entity_stage(_response(**{Q.Q_SCOPE: shaky}))


def test_an_area_scope_still_asks_for_the_entity():
    """route() checks a solid entity before the area, as in a one-call home."""
    area = ChoiceAnswer("area", {"area": 0.9, "single": 0.1}, 0.88)
    assert needs_entity_stage(_response(**{Q.Q_SCOPE: area}))


def test_information_requests_never_need_it():
    response = _response(**{Q.Q_CATEGORY: _choice({"information": 0.9})})
    assert not needs_entity_stage(response)


def test_a_home_that_fit_already_has_its_answer():
    entity = _choice({"light.bedside_lamp": 0.9, Q.NO_SINGLE_ENTITY: 0.1})
    assert not needs_entity_stage(_response(**{Q.Q_TARGET_ENTITY: entity}))


def test_the_worked_example_keeps_the_two_best_paths():
    """'Turn off the lamp' from the bedroom: 45 devices narrow to four."""
    plan = plan_entity_stage(_response(), HOME, max_options=CAP)

    assert plan.ask_room is None
    assert [(p.domain, p.area_id) for p in plan.stage.paths] == [
        ("light", "bedroom"),
        ("light", "living_room"),
    ]
    assert {e.entity_id for e in plan.stage.candidates} == {
        "light.bedroom_ceiling",
        "light.bedside_lamp",
        "light.living_room_ceiling",
        "light.floor_lamp",
    }
    question = entity_stage_questions(plan.stage)[Q.Q_TARGET_ENTITY]
    assert len(question["criteria"]) == 5, "four candidates and the escape"


def test_the_merge_scores_by_path_and_picks_the_lamp():
    first = _response()
    plan = plan_entity_stage(first, HOME, max_options=CAP)
    second = SystemOneResponse(
        model="nimble",
        answers={
            Q.Q_TARGET_ENTITY: _choice(
                {
                    "light.bedside_lamp": 0.78,
                    "light.floor_lamp": 0.15,
                    "light.bedroom_ceiling": 0.04,
                    "light.living_room_ceiling": 0.02,
                    Q.NO_SINGLE_ENTITY: 0.01,
                }
            )
        },
        input_tokens=20,
        output_tokens=1,
        latency_ms=10.0,
    )

    merged = merge_entity_stage(first, second, plan.stage)
    entity = merged.choice(Q.Q_TARGET_ENTITY)

    assert entity.choice == "light.bedside_lamp"
    assert sum(entity.probabilities.values()) == pytest.approx(1.0)
    # The bedroom path outweighs the living room's, so the lamp's lead widens.
    assert entity.probabilities["light.bedside_lamp"] > 0.78
    n, top = len(entity.probabilities), entity.probabilities["light.bedside_lamp"]
    assert entity.confidence == pytest.approx((n * top - 1) / (n - 1))
    # Everything else from the first call survives untouched.
    assert merged.choice(Q.Q_CATEGORY) is first.choice(Q.Q_CATEGORY)
    assert merged.input_tokens == 120
    assert merged.latency_ms == 50.0


def test_none_of_these_still_wins_when_stage_two_says_so():
    first = _response()
    plan = plan_entity_stage(first, HOME, max_options=CAP)
    second = SystemOneResponse(
        model="nimble",
        answers={
            Q.Q_TARGET_ENTITY: _choice(
                {
                    "light.bedside_lamp": 0.05,
                    "light.floor_lamp": 0.03,
                    "light.bedroom_ceiling": 0.01,
                    "light.living_room_ceiling": 0.01,
                    Q.NO_SINGLE_ENTITY: 0.90,
                }
            )
        },
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
    )
    merged = merge_entity_stage(first, second, plan.stage)
    assert merged.choice(Q.Q_TARGET_ENTITY).choice == Q.NO_SINGLE_ENTITY


def test_no_room_and_too_many_of_one_kind_asks_which_room():
    """No room said, and 29 lights is more than one question can offer."""
    unplaced = _response(**{Q.Q_TARGET_AREA: _choice({"no_area": 0.9, "hall": 0.1})})
    plan = plan_entity_stage(unplaced, HOME, max_options=CAP)

    assert plan.stage is None
    assert plan.ask_room is not None
    assert plan.ask_room.domain == "light"


def test_a_named_room_that_is_still_too_big_falls_back_instead():
    """25 lights in one room won't fit a 10-option cap; asking 'which room?'
    would be absurd when the room was named, so let the fallback ladder try."""
    hallway = _response(**{Q.Q_TARGET_AREA: _choice({"hall": 0.95, "no_area": 0.05})})
    plan = plan_entity_stage(hallway, HOME, max_options=10)

    assert plan.stage is None
    assert plan.ask_room is None


def test_a_path_that_would_overflow_the_question_is_skipped():
    """The beam keeps adding paths only while they still fit beside the escape."""
    response = _response(
        **{
            Q.Q_TARGET_AREA: _choice(
                {"bedroom": 0.6, "hall": 0.3, "living_room": 0.05, "no_area": 0.05}
            )
        }
    )
    # A 15-option cap leaves room for 14 candidates: the bedroom's 2 lights fit,
    # adding the hall's 25 would not, so the beam passes over it to the next.
    plan = plan_entity_stage(response, HOME, max_options=15)

    assert [p.area_id for p in plan.stage.paths] == ["bedroom", "living_room"]
    assert len(plan.stage.candidates) == 4
