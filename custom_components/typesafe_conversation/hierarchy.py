"""Choosing one entity when a home has more than the server can offer at once.

Servers cap the options in a single Choice: TypeSafe at 255, Ollama at 26. A
home with more exposed entities than that cannot be offered whole in one
``target_entity`` question, so the first request leaves it out and this module
plans a second, narrower one.

It follows TypeSafe's hierarchical-classification cookbook: classify level by
level, keep the ``BEAM_WIDTH`` most probable paths, and score a leaf by the
product of the probabilities along its path. The first request already asks
the upper levels - ``target_domain`` and ``target_area`` - so the second only
has to choose among the entities on the surviving paths.

Every function here is pure, like ``route()``: answers and catalogue in, plan
out. That is what lets the recorded fixtures test it without a network.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from . import questions as Q
from .const import BEAM_WIDTH, MIN_MARGIN, T_SCOPE, T_SCOPE_WHOLE_HOUSE
from .entities import CatalogEntity
from .system_one import ChoiceAnswer, SystemOneResponse

_GROUP_SCOPES = {"whole_house": T_SCOPE_WHOLE_HOUSE, "floor": T_SCOPE}
"""Scopes where route() acts on many entities without consulting one.

Only when the answer is solid, with the same gates route() uses: a shaky
"floor" falls through to the entity branch there, so the entity must have been
asked. "area" is absent on purpose - route() checks a solid entity before the
area, as it does for a home small enough to ask in one call.
"""

_ENTITY_CATEGORIES = frozenset({"command", "query"})


@dataclass(frozen=True, slots=True)
class Path:
    """One domain-and-area branch of the hierarchy."""

    domain: str
    area_id: str | None
    """None when the request names no room: every area of the domain."""

    p_domain: float
    p_area: float

    @property
    def probability(self) -> float:
        return self.p_domain * self.p_area

    def holds(self, entity: CatalogEntity) -> bool:
        return entity.domain == self.domain and (
            self.area_id is None or entity.area_id == self.area_id
        )


@dataclass(frozen=True, slots=True)
class EntityStage:
    """The second request: which paths survived, and who is on them."""

    paths: tuple[Path, ...]
    candidates: tuple[CatalogEntity, ...]


@dataclass(frozen=True, slots=True)
class AskForRoom:
    """No room was named and one domain alone is too large to offer.

    Guessing would mean acting on a device the user may not have meant, so the
    agent asks which room instead.
    """

    domain: str
    size: int


@dataclass(frozen=True, slots=True)
class StagePlan:
    """What, if anything, the second stage should do."""

    stage: EntityStage | None = None
    ask_room: AskForRoom | None = None
    trace: dict[str, Any] = field(default_factory=dict)


def needs_entity_stage(response: SystemOneResponse) -> bool:
    """Whether this request is about one specific entity we never asked about."""
    if response.choice(Q.Q_TARGET_ENTITY) is not None:
        return False
    category = response.choice(Q.Q_CATEGORY)
    if category is None or category.choice not in _ENTITY_CATEGORIES:
        return False
    scope = response.choice(Q.Q_SCOPE)
    if scope is None or scope.choice not in _GROUP_SCOPES:
        return True
    solid = (
        scope.confidence >= _GROUP_SCOPES[scope.choice] and scope.margin >= MIN_MARGIN
    )
    return not solid


def plan_entity_stage(
    response: SystemOneResponse,
    entities: Iterable[CatalogEntity],
    *,
    max_options: int,
    beam_width: int = BEAM_WIDTH,
) -> StagePlan:
    """Pick the paths and candidates for the second request.

    Paths are ranked by ``p_domain * p_area`` and kept while their combined
    entities still fit in one question beside the ``no_single_entity`` escape.
    """
    entities = tuple(entities)
    room = max_options - 1
    ranked = sorted(
        (p for p in _paths(response) if any(p.holds(e) for e in entities)),
        key=lambda p: -p.probability,
    )
    if not ranked:
        return StagePlan(trace={"entity_stage": "no matching path"})

    best = ranked[0]
    best_size = sum(1 for e in entities if best.holds(e))
    if best_size > room:
        trace = {
            "entity_stage": "best path too large",
            "path": _describe(best),
            "size": best_size,
        }
        if best.area_id is None:
            return StagePlan(ask_room=AskForRoom(best.domain, best_size), trace=trace)
        return StagePlan(trace=trace)

    kept: list[Path] = []
    candidates: dict[str, CatalogEntity] = {}
    for path in ranked:
        if len(kept) == beam_width:
            break
        members = {e.entity_id: e for e in entities if path.holds(e)}
        if len({**candidates, **members}) > room:
            continue
        kept.append(path)
        candidates.update(members)

    return StagePlan(
        stage=EntityStage(paths=tuple(kept), candidates=tuple(candidates.values())),
        trace={
            "entity_stage": "asked",
            "paths": [_describe(p) for p in kept],
            "candidates": len(candidates),
        },
    )


def entity_stage_questions(
    stage: EntityStage, *, inline_descriptions: bool = False
) -> dict[str, Any]:
    return {
        Q.Q_TARGET_ENTITY: Q.target_entity_question(
            stage.candidates, inline_descriptions
        )
    }


def merge_entity_stage(
    first: SystemOneResponse, second: SystemOneResponse, stage: EntityStage
) -> SystemOneResponse:
    """Fold the second answer back into the first, as one ``target_entity``.

    Each candidate is weighted by its path's probability times its own answer,
    the cookbook's path score. Every path here is the same depth (domain, area,
    entity), so this ranks exactly as the cookbook's depth-normalised geometric
    mean would, while still summing to a distribution ``route()`` can read
    confidence and margin from.
    """
    answer = second.choice(Q.Q_TARGET_ENTITY)
    if answer is None:
        return first

    weights: dict[str, float] = {}
    for entity in stage.candidates:
        path = next(p for p in stage.paths if p.holds(entity))
        weights[entity.entity_id] = path.probability * answer.probabilities.get(
            entity.entity_id, 0.0
        )
    # The escape option sits on the same scale as a candidate whose path took
    # all the kept probability, so stage two's "none of these" keeps its say.
    weights[Q.NO_SINGLE_ENTITY] = sum(p.probability for p in stage.paths) * (
        answer.probabilities.get(Q.NO_SINGLE_ENTITY, 0.0)
    )

    total = sum(weights.values()) or 1.0
    probabilities = {key: weight / total for key, weight in weights.items()}
    choice = max(probabilities, key=probabilities.__getitem__)
    merged = ChoiceAnswer(
        choice=choice,
        probabilities=probabilities,
        confidence=_confidence(probabilities[choice], len(probabilities)),
    )

    return SystemOneResponse(
        model=first.model,
        answers={**first.answers, Q.Q_TARGET_ENTITY: merged},
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        latency_ms=first.latency_ms + second.latency_ms,
        raw={**first.raw, "entity_stage": second.raw},
    )


def _paths(response: SystemOneResponse) -> list[Path]:
    domains = _distribution(response.choice(Q.Q_TARGET_DOMAIN), exclude=Q.NO_DOMAIN)
    areas = _distribution(response.choice(Q.Q_TARGET_AREA))
    if not areas:
        areas = {Q.NO_AREA: 1.0}
    return [
        Path(
            domain=domain,
            area_id=None if area == Q.NO_AREA else area,
            p_domain=p_domain,
            p_area=p_area,
        )
        for domain, p_domain in domains.items()
        for area, p_area in areas.items()
    ]


def _distribution(
    answer: ChoiceAnswer | None, exclude: str | None = None
) -> dict[str, float]:
    if answer is None:
        return {}
    return {k: v for k, v in answer.probabilities.items() if k != exclude}


def _confidence(p_top: float, options: int) -> float:
    """Jev's own definition, so route()'s thresholds read it the same way."""
    if options < 2:
        return 1.0
    return max(0.0, (options * p_top - 1) / (options - 1))


def _describe(path: Path) -> str:
    return f"{path.domain}@{path.area_id or 'any'} ({path.probability:.2f})"


__all__ = [
    "AskForRoom",
    "EntityStage",
    "Path",
    "StagePlan",
    "entity_stage_questions",
    "merge_entity_stage",
    "needs_entity_stage",
    "plan_entity_stage",
]
