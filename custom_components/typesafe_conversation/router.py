"""Turn Jev's answers into a plan, using nothing but the answers.

``route`` is a pure function: answers in, plan out. No hass, no network, no
clock. That is what makes the routing logic testable as a table of fixtures,
which matters because the thresholds here are the part most likely to need
tuning.

Two axes decide whether a branch is trustworthy, because they fail differently:

* ``confidence`` - how peaked the distribution is overall.
* ``margin`` - p(top) - p(second). A distribution can look confident while the
  top two options remain effectively tied.

A branch is "solid" only when both clear their bar. Nouls carry no confidence,
so they get a dead band instead: between the low and high thresholds we treat
the answer as "no" and log it, because a Noul near 0.5 means the model splits
yes/no, not that the condition half-holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from . import questions as Q
from .actions import INCREASING_ACTIONS, ActionSpec, spec_for
from .const import (
    CONF_ACT_EXPLICIT,
    CONF_ACT_TERSE,
    CONF_CLARIFY_FLOOR,
    LOGGER,
    MAGNITUDE_STEPS,
    MIN_MARGIN,
    NOUL_COMPOUND_HIGH,
    NOUL_COMPOUND_LOW,
    NOUL_HERE_RELATIVE,
    NOUL_RISKY,
    NOUL_RISKY_EXPLICIT,
    RISKY_ACTIONS,
    T_ACTION_PROBABILITY,
    T_AREA,
    T_CATEGORY_CANCEL,
    T_CATEGORY_COMPOUND,
    T_CATEGORY_INFORMATION,
    T_CATEGORY_QUERY,
    T_CATEGORY_UNCLEAR,
    T_DOMAIN,
    T_ENTITY,
    T_LLM_LEAN,
    T_QUERY_KIND,
    T_RISKY_ACTION,
    T_RISKY_TARGET,
    T_SCOPE,
    T_SCOPE_WHOLE_HOUSE,
)
from .entities import CatalogEntity
from .extraction import COLOR_TEMP_PRESETS, Extraction
from .system_one import ChoiceAnswer, SystemOneResponse


class Route(StrEnum):
    """What the agent decided to do with the utterance."""

    COMMAND = "command"
    QUERY = "query"
    COMPOUND = "compound"
    INFORMATION = "information"
    CANCEL = "cancel"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"
    """The target exists but cannot act. Terminal, never the fallback ladder:
    hassil would resolve the same dead entity and the LLM would apologise
    vaguely, when the useful answer is to name what is unavailable."""


@dataclass(slots=True)
class Target:
    """Who the action applies to, as intent slots would express it."""

    entity: CatalogEntity | None = None
    area_id: str | None = None
    floor_name: str | None = None
    domain: str | None = None
    whole_house: bool = False

    @property
    def described(self) -> str:
        if self.entity is not None:
            return self.entity.name
        if self.whole_house:
            return "everything"
        return "them"


@dataclass(slots=True)
class Plan:
    """The router's decision. ``execute`` turns this into intent calls."""

    route: Route
    reason: str = ""

    # command / confirm
    domain: str | None = None
    action: str | None = None
    spec: ActionSpec | None = None
    target: Target = field(default_factory=Target)
    value: float | None = None
    value_unit: str | None = None
    relative_step: int | None = None
    text_slot: tuple[str, str] | None = None
    """(slot name, value) for the free-text slots we can fill."""
    color_temp_kelvin: int | None = None

    # query
    query_kind: str | None = None

    # clarify / confirm
    options: tuple[tuple[str, str], ...] = ()
    """(entity_id, friendly name) pairs for a disambiguation question."""
    speech: str | None = None

    preferred_area_id: str | None = None
    """Sent instead of a hard area when that area holds nothing to act on.

    Home Assistant consults a preference only for single-target matches and
    for disambiguating duplicate names, so this can never widen a fan-out
    command - see _resolve_area."""

    # presentation
    script_service: str | None = None
    script_data: dict[str, Any] | None = None
    """A script's field values, filled by the language model and checked against
    its selectors. Set, the script is called directly with them instead of
    through the generic turn-on intent, which carries no data."""

    name_target_in_speech: bool = False
    """True in the middle confidence band: act, but say what we acted on."""

    # diagnostics, always populated
    trace: dict[str, Any] = field(default_factory=dict)


def _solid(answer: ChoiceAnswer | None, threshold: float) -> bool:
    """Confident enough *and* not a near-tie."""
    return (
        answer is not None
        and answer.confidence >= threshold
        and answer.margin >= MIN_MARGIN
    )


_ONLY_ACTION: dict[str, str] = {"script": "run", "scene": "activate", "button": "press"}
"""Domains with a single possible action, which a solid target implies."""


def route(
    response: SystemOneResponse,
    *,
    entities_by_id: dict[str, CatalogEntity],
    extraction: Extraction,
    speaker_area_id: str | None,
    available_domains: frozenset[str],
    always_confirm_risky: bool = True,
    catalog_floors: dict[str, str] | None = None,
    unavailable_ids: frozenset[str] = frozenset(),
) -> Plan:
    """Decide what to do. Pure - depends only on its arguments."""
    category = response.choice(Q.Q_CATEGORY)
    compound = response.noul(Q.Q_COMPOUND) or 0.0
    trace: dict[str, Any] = {
        "category": _describe(category),
        "compound": round(compound, 3),
        "latency_ms": round(response.latency_ms),
        "input_tokens": response.input_tokens,
    }

    if category is None:
        return Plan(Route.FALLBACK, reason="no category answer", trace=trace)

    # -- 0. the user is calling it off ---------------------------------------
    if category.choice == "cancel" and category.confidence >= T_CATEGORY_CANCEL:
        return Plan(Route.CANCEL, reason="category=cancel", trace=trace)

    # -- 1. more than one thing was asked for --------------------------------
    if (
        compound >= NOUL_COMPOUND_HIGH
        and category.choice == "command"
        and category.confidence >= T_CATEGORY_COMPOUND
    ):
        return Plan(Route.COMPOUND, reason=f"compound={compound:.2f}", trace=trace)
    if NOUL_COMPOUND_LOW <= compound < NOUL_COMPOUND_HIGH:
        # Splitting costs an LLM round trip plus N more Jev calls, and a bad
        # split mangles a request that would have worked as one.
        LOGGER.debug(
            "Compound noul %.2f is in the dead band; treating as a single command",
            compound,
        )

    # -- 2. nothing usable ----------------------------------------------------
    if category.choice == "unclear" and category.confidence >= T_CATEGORY_UNCLEAR:
        return Plan(Route.FALLBACK, reason="category=unclear", trace=trace)

    # -- 3. general knowledge -------------------------------------------------
    if category.choice == "information" and _solid(category, T_CATEGORY_INFORMATION):
        return Plan(Route.INFORMATION, reason="category=information", trace=trace)

    query_kind = response.choice(Q.Q_QUERY_KIND)
    trace["query_kind"] = _describe(query_kind)

    # -- 4. a question about the home ----------------------------------------
    if category.choice == "query" and category.confidence >= T_CATEGORY_QUERY:
        kind = query_kind.choice if _solid(query_kind, T_QUERY_KIND) else "needs_prose"
        plan = _plan_query(
            response,
            kind=kind,
            entities_by_id=entities_by_id,
            speaker_area_id=speaker_area_id,
            trace=trace,
        )
        return plan

    # -- 5. a command ---------------------------------------------------------
    if category.choice == "command":
        return _plan_command(
            response,
            entities_by_id=entities_by_id,
            extraction=extraction,
            speaker_area_id=speaker_area_id,
            available_domains=available_domains,
            always_confirm_risky=always_confirm_risky,
            catalog_floors=catalog_floors or {},
            unavailable_ids=unavailable_ids,
            trace=trace,
        )

    # Category was something we handle but not confidently enough to act on.
    return Plan(
        Route.FALLBACK,
        reason=f"category={category.choice} conf={category.confidence:.2f}",
        trace=trace,
    )


def _plan_query(
    response: SystemOneResponse,
    *,
    kind: str,
    entities_by_id: dict[str, CatalogEntity],
    speaker_area_id: str | None,
    trace: dict[str, Any],
) -> Plan:
    target_entity = response.choice(Q.Q_TARGET_ENTITY)
    target_area = response.choice(Q.Q_TARGET_AREA)
    target_domain = response.choice(Q.Q_TARGET_DOMAIN)
    trace.update(
        target_entity=_describe(target_entity),
        target_area=_describe(target_area),
        target_domain=_describe(target_domain),
    )

    target = Target()
    if (
        _solid(target_entity, T_ENTITY)
        and target_entity is not None
        and target_entity.choice != Q.NO_SINGLE_ENTITY
    ):
        target.entity = entities_by_id.get(target_entity.choice)
    if (
        _solid(target_area, T_AREA)
        and target_area is not None
        and target_area.choice != Q.NO_AREA
    ):
        target.area_id = target_area.choice
    elif kind == "temperature" and speaker_area_id:
        target.area_id = speaker_area_id
    if (
        _solid(target_domain, T_DOMAIN)
        and target_domain is not None
        and target_domain.choice != Q.NO_DOMAIN
    ):
        target.domain = target_domain.choice

    return Plan(
        Route.QUERY,
        reason=f"query/{kind}",
        query_kind=kind,
        target=target,
        trace=trace,
    )


def _plan_command(  # noqa: C901 - one decision tree, kept in one place on purpose
    response: SystemOneResponse,
    *,
    entities_by_id: dict[str, CatalogEntity],
    extraction: Extraction,
    speaker_area_id: str | None,
    available_domains: frozenset[str],
    always_confirm_risky: bool,
    catalog_floors: dict[str, str],
    unavailable_ids: frozenset[str],
    trace: dict[str, Any],
) -> Plan:
    scope = response.choice(Q.Q_SCOPE)
    target_entity = response.choice(Q.Q_TARGET_ENTITY)
    target_area = response.choice(Q.Q_TARGET_AREA)
    target_domain = response.choice(Q.Q_TARGET_DOMAIN)
    here = response.noul(Q.Q_HERE_RELATIVE) or 0.0
    risky = response.noul(Q.Q_RISKY) or 0.0
    trace.update(
        scope=_describe(scope),
        target_entity=_describe(target_entity),
        target_area=_describe(target_area),
        target_domain=_describe(target_domain),
        here_relative=round(here, 3),
        risky=round(risky, 3),
    )

    entity = (
        entities_by_id.get(target_entity.choice)
        if target_entity is not None and target_entity.choice != Q.NO_SINGLE_ENTITY
        else None
    )

    # -- 5a. which kind of thing ---------------------------------------------
    domain: str | None = None
    if _solid(target_domain, T_DOMAIN) and target_domain.choice != Q.NO_DOMAIN:
        domain = target_domain.choice
    # An entity is a stronger constraint than a domain guess: it implies one.
    # SIM102 suppressed: merging these makes a five-line boolean that reads worse.
    if entity is not None and _solid(target_entity, T_ENTITY):  # noqa: SIM102
        if domain is None or (
            target_domain is not None
            and target_entity.confidence > target_domain.confidence
        ):
            domain = entity.domain

    if domain is None or domain not in available_domains:
        # "turn off everything" legitimately names no domain. Whole-house with
        # no domain would raise IntentHandleError ("cannot target all
        # devices"), so the executor fans out over the controllable domains
        # instead of guessing one.
        if (
            scope is not None
            and scope.choice == "whole_house"
            and _solid(scope, T_SCOPE_WHOLE_HOUSE)
        ):
            return Plan(
                Route.COMMAND,
                reason="whole house, no single domain",
                action=_whole_house_action(response, available_domains),
                target=Target(whole_house=True),
                trace=trace,
            )
        return Plan(
            Route.FALLBACK,
            reason=f"no usable domain ({_describe(target_domain)})",
            trace=trace,
        )

    # -- 5b. what to do, read from that domain's branch only ------------------
    if (
        (implied := _ONLY_ACTION.get(domain)) is not None
        and entity is not None
        and entity.domain == domain
        and _solid(target_entity, T_ENTITY)
    ):
        # A routine, scene or button can only be run. Once the model is sure
        # which one, a second vote on "what to do" can only lose it: "wake me
        # up at 5:15" names script.set_alarm at 0.96 yet splits the action
        # question 55/45, because waking up doesn't sound like running a routine.
        action = implied
        # Its certainty is the target's: there was nothing else to decide.
        action_probability = action_confidence = target_entity.confidence
        trace["action"] = "implied"
    else:
        action_answer = response.choice(Q.action_question_id(domain))
        trace["action"] = _describe(action_answer)
        if action_answer is None:
            return Plan(
                Route.FALLBACK, reason=f"no action answer for {domain}", trace=trace
            )
        action_probability = action_answer.probabilities.get(action_answer.choice, 0.0)
        action_confidence = action_answer.confidence
        if (
            action_answer.choice == Q.NOT_TARGETED
            or action_probability < T_ACTION_PROBABILITY
        ):
            return Plan(
                Route.FALLBACK,
                reason=(
                    f"action_{domain}={action_answer.choice} p={action_probability:.2f}"
                ),
                trace=trace,
            )
        action = action_answer.choice

    spec = spec_for(domain, action)
    if spec is None:
        return Plan(
            Route.FALLBACK, reason=f"no intent for {domain}.{action}", trace=trace
        )

    # -- 5c. who it applies to ------------------------------------------------
    target = Target(domain=domain)
    chosen_area: str | None = None
    preferred_area: str | None = None
    if (
        scope is not None
        and scope.choice == "whole_house"
        and _solid(scope, T_SCOPE_WHOLE_HOUSE)
    ):
        target.whole_house = True
    elif (
        scope is not None
        and scope.choice == "floor"
        and _solid(scope, T_SCOPE)
        and (floor := _resolve_floor(entity, target_area, catalog_floors)) is not None
    ):
        target.floor_name = floor
    elif entity is not None and _solid(target_entity, T_ENTITY):
        target.entity = entity
        target.area_id = entity.area_id
    elif scope is not None and scope.choice == "single" and _solid(scope, T_SCOPE):
        # The user asked for one specific thing and we could not work out
        # which. Widening to the whole area would act on devices they did not
        # mention, so ask instead - unless the room the request was spoken in
        # holds exactly one of these, in which case there is nothing to widen.
        if (
            here_area := _lone_candidate_area(
                speaker_area_id, domain, entities_by_id, unavailable_ids
            )
        ) is not None:
            trace["speaker_area_default"] = here_area
            chosen_area = here_area
        else:
            return _clarify_or_fall_back(
                domain,
                action,
                target_entity,
                entities_by_id,
                trace,
                reason="scope=single but no confident entity",
            )
    elif (
        _solid(target_area, T_AREA)
        and target_area is not None
        and target_area.choice != Q.NO_AREA
    ):
        chosen_area = target_area.choice
    elif here >= NOUL_HERE_RELATIVE and speaker_area_id:
        chosen_area = speaker_area_id
    elif spec.name_only:
        # This handler has no area slot, so without an entity there is nothing
        # we can legally send.
        return Plan(
            Route.FALLBACK,
            reason=f"{spec.intent_type} needs a named entity",
            trace=trace,
        )
    elif (
        here_area := _lone_candidate_area(
            speaker_area_id, domain, entities_by_id, unavailable_ids
        )
    ) is not None:
        trace["speaker_area_default"] = here_area
        chosen_area = here_area
    else:
        return _clarify_or_fall_back(
            domain,
            action,
            target_entity,
            entities_by_id,
            trace,
            reason="no usable target",
        )

    if target.entity is not None and target.entity.entity_id in unavailable_ids:
        dead = target.entity
        stand_in = _substitute_unavailable(
            dead, target_entity, entities_by_id, unavailable_ids
        )
        trace["unavailable_target"] = dead.entity_id
        if stand_in is None:
            # Nothing else can serve. Say which thing is down - dispatching
            # would surface a service-layer error that names nothing useful.
            return Plan(
                Route.UNAVAILABLE,
                reason=f"{dead.entity_id} is unavailable",
                domain=domain,
                action=action,
                target=target,
                speech=f"{dead.name} is unavailable.",
                trace=trace,
            )
        trace["substituted_for"] = stand_in.entity_id
        target.entity = stand_in
        target.area_id = stand_in.area_id

    if chosen_area is not None:
        resolution = _resolve_area(
            chosen_area, domain, spec, entities_by_id, unavailable_ids
        )
        trace["area_resolution"] = resolution.outcome.value
        match resolution.outcome:
            case AreaOutcome.USE_AREA:
                target.area_id = chosen_area
            case AreaOutcome.PREFER_AREA:
                # No hard area: it holds nothing of this domain, so sending it
                # would fail. The handler picks one target and still prefers
                # this room.
                preferred_area = chosen_area
            case AreaOutcome.USE_ONLY_ENTITY:
                target.entity = resolution.entity
                target.area_id = resolution.entity.area_id
            case AreaOutcome.CLARIFY:
                return _clarify_or_fall_back(
                    domain,
                    action,
                    target_entity,
                    entities_by_id,
                    trace,
                    reason=f"no {domain} in area {chosen_area}",
                )

    plan = Plan(
        Route.COMMAND,
        reason=f"{domain}.{action}",
        domain=domain,
        action=action,
        preferred_area_id=preferred_area,
        spec=spec,
        target=target,
        trace=trace,
    )

    # -- 5d. arguments --------------------------------------------------------
    _apply_arguments(plan, response, extraction, trace)

    # -- 5e. do not quietly unlock the house ---------------------------------
    # The risky question is authoritative. An explicit unlock/open/disarm only
    # lowers the bar; requiring one would exempt scripts, whose action is
    # always "run", and security actions are commonly scripts.
    risky_threshold = NOUL_RISKY_EXPLICIT if action in RISKY_ACTIONS else NOUL_RISKY
    if risky >= risky_threshold:
        target_conf = (
            target_entity.confidence
            if target_entity is not None and target.entity is not None
            else (target_area.confidence if target_area is not None else 0.0)
        )
        if (
            always_confirm_risky
            or action_confidence < T_RISKY_ACTION
            or target_conf < T_RISKY_TARGET
        ):
            plan.route = Route.CONFIRM
            plan.reason = f"risky {action}, asking first"
            return plan

    # -- presentation ---------------------------------------------------------
    deciding = min(
        action_probability,
        target_entity.confidence
        if target.entity is not None and target_entity is not None
        else 1.0,
    )
    plan.name_target_in_speech = (
        deciding < CONF_ACT_TERSE
        or trace.get("area_resolution") == AreaOutcome.USE_ONLY_ENTITY.value
        # The user named one thing and we used another; say which.
        or "substituted_for" in trace
    )
    trace["deciding_confidence"] = round(deciding, 3)
    if deciding < CONF_ACT_EXPLICIT:
        LOGGER.debug(
            "Acting at %.2f confidence, below the explicit band - naming the target",
            deciding,
        )
    return plan


def _resolve_floor(
    entity: CatalogEntity | None,
    target_area: ChoiceAnswer | None,
    catalog_floors: dict[str, str],
) -> str | None:
    """Work out which floor a floor-scoped request means.

    Returns None when we cannot tell, and the caller then falls through to area
    targeting rather than giving up: a home can name an area "Upstairs" without
    ever defining a floor, and "make it cooler upstairs" should still work.
    """
    if entity is not None and entity.floor_name:
        return entity.floor_name
    if target_area is not None and target_area.choice in catalog_floors:
        return catalog_floors[target_area.choice]
    return None


class AreaOutcome(StrEnum):
    """What to do with an area the request resolved to."""

    USE_AREA = "use_area"
    """The area holds something of this domain; send it as a hard constraint."""

    PREFER_AREA = "prefer_area"
    """It holds nothing, but the handler picks a single target - let Home
    Assistant widen to the home while still preferring this room."""

    USE_ONLY_ENTITY = "use_only_entity"
    """It holds nothing, and the home has exactly one candidate. Use it, and
    say which one out loud since the user did not name it."""

    CLARIFY = "clarify"
    """It holds nothing and several candidates exist. Widening a fan-out
    command here would act on devices the user never mentioned."""


@dataclass(slots=True)
class AreaResolution:
    """How to target a command that resolved to an area."""

    outcome: AreaOutcome
    entity: CatalogEntity | None = None


def _resolve_area(
    area_id: str,
    domain: str,
    spec: ActionSpec,
    entities_by_id: dict[str, CatalogEntity],
    unavailable_ids: frozenset[str] = frozenset(),
) -> AreaResolution:
    """Decide how to target an area, given what the catalog says is in it.

    Sending an area that holds no entity of the target domain is a guaranteed
    MatchFailedError, and the catalog is right here - so decide now rather
    than spend a round trip finding out.
    """
    # An unavailable entity is not a candidate: counting it would send a
    # request that cannot succeed, or name it as the only option.
    in_area = [
        e
        for e in entities_by_id.values()
        if e.area_id == area_id
        and e.domain == domain
        and e.entity_id not in unavailable_ids
    ]
    if in_area:
        return AreaResolution(AreaOutcome.USE_AREA)

    if spec.single_target:
        # The handler picks exactly one target, and without a preference a
        # multi-candidate match fails outright with MULTIPLE_TARGETS.
        return AreaResolution(AreaOutcome.PREFER_AREA)

    candidates = [
        e
        for e in entities_by_id.values()
        if e.domain == domain and e.entity_id not in unavailable_ids
    ]
    if len(candidates) == 1:
        return AreaResolution(AreaOutcome.USE_ONLY_ENTITY, candidates[0])

    return AreaResolution(AreaOutcome.CLARIFY)


def _lone_candidate_area(
    speaker_area_id: str | None,
    domain: str,
    entities_by_id: dict[str, CatalogEntity],
    unavailable_ids: frozenset[str],
) -> str | None:
    """The room the request was spoken in, when it can only mean one device.

    A request that names neither a device nor a room, spoken to a satellite
    standing in a room, means that room. The here-relative noul cannot carry
    this on its own: it asks whether the wording depends on where the speaker
    is standing, which scores low when the request names no device either -
    there is no locative and no device word for it to be relative about.

    Restricted to rooms holding exactly one candidate, where "target the area"
    and "target that one device" are the same instruction. Where the room holds
    several, the caller's scope=single still means the user meant one of them,
    and acting on all of them is not an improvement over asking.
    """
    if not speaker_area_id:
        return None
    in_area = [
        entity
        for entity in entities_by_id.values()
        if entity.area_id == speaker_area_id
        and entity.domain == domain
        and entity.entity_id not in unavailable_ids
    ]
    return speaker_area_id if len(in_area) == 1 else None


def _substitute_unavailable(
    chosen: CatalogEntity,
    target_entity: ChoiceAnswer | None,
    entities_by_id: dict[str, CatalogEntity],
    unavailable_ids: frozenset[str],
) -> CatalogEntity | None:
    """Find a working stand-in for an entity that cannot act.

    The answer is a full distribution, not just a winner, so when the top
    choice is dead the runners-up are already ranked by how well they fit the
    request. Walk them for the best same-domain candidate that is actually
    available, above a floor so a stray 0.01 is never promoted.
    """
    if target_entity is None:
        return None
    ranked = sorted(target_entity.probabilities.items(), key=lambda kv: -kv[1])
    for entity_id, probability in ranked:
        if probability < CONF_CLARIFY_FLOOR:
            break
        candidate = entities_by_id.get(entity_id)
        if (
            candidate is not None
            and candidate.entity_id != chosen.entity_id
            and candidate.domain == chosen.domain
            and candidate.entity_id not in unavailable_ids
        ):
            return candidate
    return None


def _clarify_or_fall_back(
    domain: str,
    action: str,
    target_entity: ChoiceAnswer | None,
    entities_by_id: dict[str, CatalogEntity],
    trace: dict[str, Any],
    *,
    reason: str,
) -> Plan:
    """Ask which device was meant, if we have two plausible candidates.

    Clarification is only ever used for *target* ambiguity. The user knows
    which lamp they meant, and the question is short and natural. We never ask
    it about the category or the action - "did you want to turn something on?"
    is a worse experience than simply letting another matcher try.
    """
    if target_entity is not None and target_entity.confidence >= CONF_CLARIFY_FLOOR:
        # Only candidates that could actually serve this command. The
        # distribution covers every exposed entity, so without the domain
        # filter a request to play music can be answered with a vacuum cleaner.
        ranked = [
            (entity_id, probability)
            for entity_id, probability in sorted(
                target_entity.probabilities.items(), key=lambda kv: -kv[1]
            )
            if entity_id != Q.NO_SINGLE_ENTITY
            and entity_id in entities_by_id
            and entities_by_id[entity_id].domain == domain
        ]
        # ... and only when the top two are genuinely too close to call. A
        # clear leader with a 0.03 behind it is one candidate and some noise:
        # reading the noise out as a real alternative invites the wrong answer.
        if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] < MIN_MARGIN:
            return Plan(
                Route.CLARIFY,
                reason=reason,
                domain=domain,
                action=action,
                options=tuple(
                    (entity_id, entities_by_id[entity_id].name)
                    for entity_id, _ in ranked[:2]
                ),
                trace=trace,
            )
    return Plan(Route.FALLBACK, reason=reason, trace=trace)


def _whole_house_action(
    response: SystemOneResponse, available_domains: frozenset[str]
) -> str | None:
    """Pick the action for a whole-house command that names no domain.

    We read the action branch of every domain present and take the one the
    model was most sure about, as long as it is an on/off style action - those
    are the only ones that make sense applied to the whole house.
    """
    best: tuple[float, str] | None = None
    for domain in available_domains:
        answer = response.choice(Q.action_question_id(domain))
        if answer is None or answer.choice in (Q.NOT_TARGETED,):
            continue
        if answer.choice not in ("turn_on", "turn_off", "close", "lock"):
            continue
        if best is None or answer.confidence > best[0]:
            best = (answer.confidence, answer.choice)
    return best[1] if best else None


def _apply_arguments(
    plan: Plan,
    response: SystemOneResponse,
    extraction: Extraction,
    trace: dict[str, Any],
) -> None:
    """Fill the numeric and text slots the chosen action needs."""
    spec = plan.spec
    if spec is None:
        return

    if spec.needs_text == "color":
        color = response.choice(Q.Q_COLOR_PICK)
        trace["color_pick"] = _describe(color)
        if color is not None and color.choice != Q.NO_VALUE:
            if (kelvin := COLOR_TEMP_PRESETS.get(color.choice)) is not None:
                plan.color_temp_kelvin = kelvin
            else:
                plan.text_slot = ("color", color.choice)
        return

    if spec.needs_text == "item":
        span = response.choice(Q.Q_LIST_ITEM_SPAN)
        trace["list_item_span"] = _describe(span)
        if span is not None and span.choice != Q.NO_VALUE and span.confidence >= 0.5:
            plan.text_slot = ("item", span.choice)
        else:
            plan.route = Route.FALLBACK
            plan.reason = "could not tell what item was meant"
        return

    if spec.needs_text == "search_query":
        span = response.choice(Q.Q_MEDIA_SPAN)
        trace["media_span"] = _describe(span)
        if span is not None and span.choice != Q.NO_VALUE and span.confidence >= 0.55:
            plan.text_slot = ("search_query", span.choice)
        else:
            # hassil has good patterns for this and, because we advertise
            # CONTROL, the pipeline never let it try. Let it.
            plan.route = Route.FALLBACK
            plan.reason = "media search query not confident"
        return

    if spec.value_slot is None:
        return

    if spec.relative:
        magnitude = response.choice(Q.Q_MAGNITUDE)
        trace["magnitude"] = _describe(magnitude)
        step = MAGNITUDE_STEPS.get(
            magnitude.choice if magnitude is not None else "moderate", 25
        )
        sign = 1 if plan.action in INCREASING_ACTIONS else -1
        plan.relative_step = sign * step
        return

    value_pick = response.choice(Q.Q_VALUE_PICK)
    trace["value_pick"] = _describe(value_pick)
    if value_pick is None or value_pick.choice == Q.NO_VALUE:
        # Absolute action with no value to apply: let hassil or the LLM try.
        plan.route = Route.FALLBACK
        plan.reason = f"{plan.action} needs a value and none was picked"
        return
    candidate = extraction.by_span(value_pick.choice)
    if candidate is None:
        plan.route = Route.FALLBACK
        plan.reason = f"picked span {value_pick.choice!r} is not a known candidate"
        return
    plan.value = candidate.value
    plan.value_unit = candidate.unit


def _describe(answer: ChoiceAnswer | None) -> dict[str, Any] | None:
    """Compact record of one answer, for the debug log and the HA trace."""
    if answer is None:
        return None
    top = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])[:3]
    return {
        "choice": answer.choice,
        "confidence": round(answer.confidence, 3),
        "margin": round(answer.margin, 3),
        "top": {name: round(p, 3) for name, p in top},
    }


def should_try_llm_answer(response: SystemOneResponse) -> bool:
    """Whether a freeform LLM answer is worth trying as a fallback."""
    category = response.choice(Q.Q_CATEGORY)
    if category is None:
        return False
    lean = category.probabilities.get("information", 0.0) + category.probabilities.get(
        "query", 0.0
    )
    return lean >= T_LLM_LEAN


__all__ = ["Plan", "Route", "Target", "route", "should_try_llm_answer"]
