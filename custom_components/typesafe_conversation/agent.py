"""The request pipeline: one Jev call, then code decides.

Everything the router might need is asked in a single API call, including the
branches that will turn out to be irrelevant. Measured against jev-1.13.0, each
extra question costs about 97 input tokens and almost no extra latency, so
asking a question we discard is close to free - and it saves a round trip on
the requests where it turns out to matter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from homeassistant.components import conversation
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import intent
from homeassistant.util import dt as dt_util

from . import hierarchy
from . import questions as Q
from .const import (
    CATALOG_SUMMARY_MAX_ENTITIES,
    DEFAULT_ALWAYS_CONFIRM_RISKY,
    LOGGER,
    MAX_HISTORY_TURNS,
)
from .entities import EntityCatalog
from .executor import (
    ExecutionError,
    async_execute,
    async_execute_cancel,
    async_execute_query,
    describe_action,
)
from .extraction import extract
from .llm_backend import LLMBackend, LLMBackendError
from .router import Plan, Route, route, should_try_llm_answer
from .system_one import (
    SystemOneClient,
    SystemOneError,
    SystemOneRequestError,
    SystemOneResponse,
)


def build_request(
    hass: HomeAssistant,
    catalog: EntityCatalog,
    text: str,
    *,
    speaker_area_id: str | None,
    max_options: int,
    structural: dict[str, Any] | None = None,
    inline_descriptions: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The state and questions for one utterance - the first request of a turn.

    Shared with setup, which times one of these to set the request timeout, so
    the request it measures is exactly the shape a real command sends.
    """
    extraction = extract(
        text,
        want_media="media_player" in catalog.domains,
        want_color="light" in catalog.domains,
    )
    if structural is None:
        structural = Q.build_questions(
            entities=catalog.entities,
            areas=catalog.areas,
            domains=catalog.domains,
            extraction=extract("", want_media=False, want_color=False),
            inline_descriptions=inline_descriptions,
            max_options=max_options,
        )
    questions = dict(structural)
    # Conditional questions depend on the utterance, not the catalog, so they
    # are built fresh and never cached.
    if extraction.values:
        questions[Q.Q_VALUE_PICK] = Q._value_pick_question(extraction)
    if extraction.colors_mentioned:
        questions[Q.Q_COLOR_PICK] = Q._color_pick_question(extraction, max_options)
    if extraction.media_chunks:
        questions[Q.Q_MEDIA_SPAN] = Q._media_span_question(extraction)
    Q.validate_questions(questions, max_options)

    state = {
        "request": {
            "text": text,
            "language": hass.config.language,
            "spoken_from_area": speaker_area_id,
            "local_time": dt_util.now().strftime("%Y-%m-%dT%H:%M"),
            "weekday": dt_util.now().strftime("%A"),
        },
        "home": catalog.snapshot(catalog.entities),
    }
    return state, questions


class _AskForRoom(Exception):
    """No room was named and one kind of device alone is too many to offer."""


@dataclass(slots=True)
class AgentSettings:
    """Per-conversation settings, merged from the entry and its subentry."""

    inline_entity_descriptions: bool = False
    always_confirm_risky: bool = DEFAULT_ALWAYS_CONFIRM_RISKY
    bypass_local_intents: bool = False


class TypeSafeAgent:
    """Runs one utterance through Jev and carries out the result."""

    def __init__(
        self,
        hass: HomeAssistant,
        catalog: EntityCatalog,
        jev: SystemOneClient,
        llm: LLMBackend | None,
        settings: AgentSettings,
        traces: Any = None,
    ) -> None:
        self.hass = hass
        self.catalog = catalog
        self.jev = jev
        self.llm = llm
        self.settings = settings
        self._traces = traces
        self._questions_cache: tuple[int, bool, dict[str, Any]] | None = None
        self.continue_conversation = False
        """Set per request. Belongs to ConversationResult, not IntentResponse,
        so the entity reads it back after async_process returns."""

    # -- the entry point ------------------------------------------------------

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        self.continue_conversation = False
        text = user_input.text.strip()
        if not text:
            # No point spending a request on an empty utterance.
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.NO_INTENT_MATCH,
                "Sorry, I didn't catch that.",
            )

        speaker_area_id = self._speaker_area(user_input)
        try:
            response, entities = await self._ask(text, speaker_area_id, chat_log)
        except _AskForRoom:
            return self._speech(user_input, "Which room?", continue_conversation=True)
        except SystemOneRequestError as err:
            # The server refused the request and would refuse it again: too
            # many options, a prompt over the model's context, a missing model.
            # Already logged verbatim; keep it in the diagnostics too, since it
            # usually names the exact setting to change.
            self._trace_failure(text, user_input, err)
            return await self._fallback(user_input, chat_log, None)
        except SystemOneError as err:
            LOGGER.warning(
                "System One unavailable (%s); using the fallback ladder", err
            )
            self._trace_failure(text, user_input, err)
            return await self._fallback(user_input, chat_log, None)

        plan = route(
            response,
            entities_by_id={e.entity_id: e for e in entities},
            extraction=extract(
                text,
                want_media="media_player" in self.catalog.domains,
                want_color="light" in self.catalog.domains,
            ),
            speaker_area_id=speaker_area_id,
            available_domains=frozenset(self.catalog.domains),
            always_confirm_risky=self.settings.always_confirm_risky,
            catalog_floors={
                a.area_id: a.floor_name for a in self.catalog.areas if a.floor_name
            },
            unavailable_ids=self._unavailable_ids(),
        )
        record = {
            "utterance": text,
            # Where the request came from. Standard ConversationInput fields,
            # so these are populated whatever the front end: a satellite fills
            # both, typed input leaves both None. Recording them is what turns
            # "something keeps arming the alarm" into a one-step answer.
            "device_id": user_input.device_id,
            "satellite_id": user_input.satellite_id,
            "from_satellite": user_input.satellite_id is not None,
            "route": plan.route.value,
            "reason": plan.reason,
            "domain": plan.domain,
            "action": plan.action,
            "target": plan.target.entity.entity_id
            if plan.target.entity is not None
            else (
                f"area:{plan.target.area_id}"
                if plan.target.area_id
                else ("whole_house" if plan.target.whole_house else None)
            ),
            "value": plan.value,
            "relative_step": plan.relative_step,
            "text_slot": plan.text_slot,
            **plan.trace,
            **(
                {"entity_stage": stage}
                if (stage := response.raw.get("entity_stage_plan"))
                else {}
            ),
        }
        LOGGER.debug(
            "Routed %r from %s -> %s (%s) in %sms, %s input tokens",
            text,
            user_input.satellite_id or user_input.device_id or "text input",
            plan.route.value,
            plan.reason,
            plan.trace.get("latency_ms"),
            plan.trace.get("input_tokens"),
        )
        for key, value in plan.trace.items():
            if isinstance(value, dict) and "choice" in value:
                LOGGER.debug(
                    "  %-16s %-28s conf %.2f margin %.2f  %s",
                    key,
                    value["choice"],
                    value["confidence"],
                    value["margin"],
                    value["top"],
                )
        if self._traces is not None:
            self._traces.append(record)
        conversation.async_conversation_trace_append(
            conversation.ConversationTraceEventType.AGENT_DETAIL, record
        )
        return await self._carry_out(plan, response, user_input, chat_log)

    def _trace_failure(
        self,
        text: str,
        user_input: conversation.ConversationInput,
        err: Exception,
    ) -> None:
        """Keep a failed request in the diagnostics, with the server's reason."""
        if self._traces is None:
            return
        self._traces.append(
            {
                "utterance": text,
                "device_id": user_input.device_id,
                "satellite_id": user_input.satellite_id,
                "from_satellite": user_input.satellite_id is not None,
                "route": "error",
                "reason": str(err),
            }
        )

    # -- the System One call --------------------------------------------------

    async def _ask(
        self,
        text: str,
        speaker_area_id: str | None,
        chat_log: conversation.ChatLog | None,
    ) -> tuple[SystemOneResponse, tuple]:
        """Ask everything in one call, plus a second when the home is too big.

        The whole catalogue always goes in the state. Only the ``target_entity``
        question is bounded by the server's option cap; when the home exceeds
        it, the first call leaves that question out and ``hierarchy`` plans a
        second one over just the likeliest devices.
        """
        entities = self.catalog.entities
        max_options = self.jev.profile.max_options
        state, questions = build_request(
            self.hass,
            self.catalog,
            text,
            speaker_area_id=speaker_area_id,
            max_options=max_options,
            structural=self._structural_questions(entities),
        )
        if chat_log is not None and (history := self._history(chat_log)):
            state["conversation"] = history

        response = await self.jev.async_ask(state, questions)
        if not hierarchy.needs_entity_stage(response):
            return response, entities

        plan = hierarchy.plan_entity_stage(response, entities, max_options=max_options)
        LOGGER.debug("Entity stage for %r: %s", text, plan.trace)
        if plan.ask_room is not None:
            raise _AskForRoom
        if plan.stage is None:
            return response, entities

        # Only the candidates go in the second state: the question is which of
        # these few devices, and a small prompt is what keeps it quick.
        second_state: dict[str, Any] = {
            "request": state["request"],
            "home": self.catalog.snapshot(plan.stage.candidates),
        }
        if "conversation" in state:
            second_state["conversation"] = state["conversation"]
        second_questions = hierarchy.entity_stage_questions(
            plan.stage,
            inline_descriptions=self.settings.inline_entity_descriptions,
        )
        Q.validate_questions(second_questions, max_options)
        second = await self.jev.async_ask(second_state, second_questions)
        merged = hierarchy.merge_entity_stage(response, second, plan.stage)
        merged.raw["entity_stage_plan"] = plan.trace
        return merged, entities

    def _structural_questions(self, entities: tuple) -> dict[str, Any]:
        """Cache the catalog-derived questions against the catalog generation.

        They are a pure function of the catalog, so rebuilding twenty question
        dicts on every utterance would be wasted work.
        """
        generation = self.catalog.generation
        inline = self.settings.inline_entity_descriptions
        if (
            self._questions_cache is not None
            and self._questions_cache[0] == generation
            and self._questions_cache[1] == inline
        ):
            return self._questions_cache[2]

        built = Q.build_questions(
            entities=entities,
            areas=self.catalog.areas,
            domains=self.catalog.domains,
            extraction=extract("", want_media=False, want_color=False),
            inline_descriptions=inline,
            max_options=self.jev.profile.max_options,
        )
        self._questions_cache = (generation, inline, built)
        return built

    # -- acting on the plan ---------------------------------------------------

    async def _carry_out(
        self,
        plan: Plan,
        response: SystemOneResponse,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        match plan.route:
            case Route.CANCEL:
                return await async_execute_cancel(self.hass, user_input)

            case Route.COMPOUND:
                return await self._handle_compound(user_input, chat_log)

            case Route.INFORMATION:
                return await self._answer_freeform(user_input, chat_log)

            case Route.QUERY:
                # hassil's GetState speech is well phrased and localized, and
                # because we advertise CONTROL the pipeline withheld exactly
                # this intent from it. Give it the first try - it costs ~5ms.
                if (
                    local := await conversation.async_handle_intents(
                        self.hass, user_input, chat_log
                    )
                ) is not None:
                    return local
                if plan.query_kind == "needs_prose":
                    return await self._answer_freeform(user_input, chat_log)
                try:
                    return await async_execute_query(self.hass, plan, user_input)
                except (ExecutionError, intent.IntentError) as err:
                    LOGGER.debug("Query execution failed (%s)", err)
                    return await self._answer_freeform(user_input, chat_log)

            case Route.CLARIFY:
                names = " or ".join(name for _, name in plan.options)
                return self._speech(
                    user_input, f"Did you mean the {names}?", continue_conversation=True
                )

            case Route.UNAVAILABLE:
                # Terminal. The fallback ladder would resolve the same dead
                # entity via hassil, then have the LLM apologise vaguely.
                return self._speech(user_input, plan.speech or "That is unavailable.")

            case Route.CONFIRM:
                return self._speech(
                    user_input, _confirm_question(plan), continue_conversation=True
                )

            case Route.COMMAND:
                return await self._run_command(plan, user_input, chat_log)

        return await self._fallback(user_input, chat_log, response)

    async def _run_command(
        self,
        plan: Plan,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        try:
            response = await async_execute(self.hass, plan, user_input, self.catalog)
        except intent.MatchFailedError as err:
            LOGGER.debug("No match for %s (%s); falling back", plan.reason, err)
            return await self._fallback(user_input, chat_log, None)
        except (ExecutionError, intent.IntentError) as err:
            LOGGER.warning("Could not carry out %s: %s", plan.reason, err)
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"Sorry, I couldn't do that. {err}",
            )

        if failed := _wholly_failed(response):
            # Home Assistant records the *area* it matched in success_results,
            # so a response in which every entity refused the call still comes
            # back as action_done with no error. Reading that as a win would
            # have us cheerfully announce something that did not happen.
            LOGGER.warning(
                "%s reached no entity: %s rejected the call",
                plan.reason,
                ", ".join(failed),
            )
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"{_join(failed)} could not do that.",
            )

        if not response.speech:
            # A successful command must always say something. Nothing else
            # will: the intent handlers set targets and states but no speech,
            # so without this the pipeline skips TTS and the user cannot tell
            # a command that worked from one that hung. name_target_in_speech
            # only decides how specific to be - in the middle confidence band
            # we name the target so a wrong guess can be corrected at once.
            response.async_set_speech(describe_action(plan, response))
        return response

    async def _handle_compound(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        if self.llm is None:
            return await self._fallback(user_input, chat_log, None)

        parts = await self.llm.split_compound(user_input.text)
        if not parts:
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                "I couldn't split that into separate requests safely, so nothing "
                "was done. Could you ask for one thing at a time?",
            )
        if len(parts) == 1:
            # Not actually compound: one more pass without the compound branch.
            return await self._rerun_single(parts[0], user_input, chat_log)

        speaker_area_id = self._speaker_area(user_input)
        results = await asyncio.gather(
            *(self._ask(part, speaker_area_id, None) for part in parts),
            return_exceptions=True,
        )

        # Resolve every part before acting on any of them. Nothing physical can
        # be undone, so an unclear second part must not leave the first one done.
        plans: list[tuple[str, Plan]] = []
        unclear: list[str] = []
        for part, result in zip(parts, results, strict=True):
            if isinstance(result, BaseException):
                unclear.append(part)
                continue
            sub_response, entities = result
            plan = route(
                sub_response,
                entities_by_id={e.entity_id: e for e in entities},
                extraction=extract(
                    part,
                    want_media="media_player" in self.catalog.domains,
                    want_color="light" in self.catalog.domains,
                ),
                speaker_area_id=speaker_area_id,
                available_domains=frozenset(self.catalog.domains),
                always_confirm_risky=self.settings.always_confirm_risky,
                unavailable_ids=self._unavailable_ids(),
            )
            # A clarifying question or a confirmation mid-way would leave the
            # rest hanging, so anything but a plain command or query is unclear.
            if plan.route not in (Route.COMMAND, Route.QUERY):
                unclear.append(part)
                continue
            plans.append((part, plan))
        if unclear:
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"I didn't do anything, because I wasn't sure about {_join(unclear)}. "
                "Could you say that part again?",
            )

        # In the order the user said them: "turn on the AC and set it to 20" only
        # works one way round. Stop at the first failure, for the same reason.
        done: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        for index, (part, plan) in enumerate(plans):
            sub_input = _with_text(user_input, part)
            try:
                if plan.route is Route.QUERY:
                    result = await async_execute_query(self.hass, plan, sub_input)
                else:
                    result = await async_execute(
                        self.hass, plan, sub_input, self.catalog
                    )
                if _step_failed(result):
                    raise ExecutionError("The step did not reach its target")
            except (ExecutionError, intent.IntentError) as err:
                LOGGER.debug("Sub-command %r failed: %s", part, err)
                failed.append(part)
                skipped = [text for text, _ in plans[index + 1 :]]
                break
            done.append(plan.target.described)

        return self._compound_response(user_input, done, failed, skipped)

    def _compound_response(
        self,
        user_input: conversation.ConversationInput,
        done: list[str],
        failed: list[str],
        skipped: list[str],
    ) -> intent.IntentResponse:
        # The user needs to know precisely what did and did not happen.
        stopped = f", so I stopped before {_join(skipped)}" if skipped else ""
        if not done:
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"Sorry, I couldn't {_join(failed)}{stopped}.",
            )
        speech = f"Done: {_join(done)}."
        if failed:
            speech += f" But I couldn't {_join(failed)}{stopped}."
        return self._speech(user_input, speech)

    async def _rerun_single(
        self,
        text: str,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        try:
            response, entities = await self._ask(
                text, self._speaker_area(user_input), chat_log
            )
        except _AskForRoom:
            return self._speech(user_input, "Which room?", continue_conversation=True)
        except SystemOneError:
            return await self._fallback(user_input, chat_log, None)
        plan = route(
            response,
            entities_by_id={e.entity_id: e for e in entities},
            extraction=extract(
                text,
                want_media="media_player" in self.catalog.domains,
                want_color="light" in self.catalog.domains,
            ),
            speaker_area_id=self._speaker_area(user_input),
            available_domains=frozenset(self.catalog.domains),
            always_confirm_risky=self.settings.always_confirm_risky,
            catalog_floors={
                a.area_id: a.floor_name for a in self.catalog.areas if a.floor_name
            },
            unavailable_ids=self._unavailable_ids(),
        )
        if plan.route is Route.COMPOUND:
            plan.route = Route.FALLBACK
        return await self._carry_out(plan, response, user_input, chat_log)

    # -- fallbacks ------------------------------------------------------------

    async def _fallback(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        response: SystemOneResponse | None,
    ) -> intent.IntentResponse:
        """hassil, then the LLM, then admit defeat.

        No intent_filter: because we advertise CONTROL, the pipeline's
        prefer-local pass withheld HassGetState and HassMediaSearchAndPlay from
        the matcher. This is a genuinely new attempt, not a repeat of one.
        """
        if (
            local := await conversation.async_handle_intents(
                self.hass, user_input, chat_log
            )
        ) is not None:
            LOGGER.debug("Fallback: handled locally by the sentence matcher")
            return local

        if self.llm is not None and (
            response is None or should_try_llm_answer(response)
        ):
            try:
                return await self._answer_freeform(user_input, chat_log)
            except LLMBackendError as err:
                LOGGER.warning("LLM fallback failed: %s", err)

        return self._error(
            user_input,
            intent.IntentResponseErrorCode.NO_INTENT_MATCH,
            "Sorry, I'm not sure what you'd like me to do.",
        )

    async def _answer_freeform(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> intent.IntentResponse:
        if self.llm is None:
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.NO_INTENT_MATCH,
                "I can only control the home right now.",
            )
        now = dt_util.now()
        entities = self.catalog.entities[:CATALOG_SUMMARY_MAX_ENTITIES]
        try:
            answer = await self.llm.answer_freeform(
                user_input.text,
                self._history_pairs(chat_log),
                home_state=self.catalog.summarize(entities),
                local_time=now.strftime("%H:%M"),
                weekday=now.strftime("%A"),
                speaker_area=self._speaker_area_name(user_input),
            )
        except LLMBackendError as err:
            LOGGER.warning("LLM could not answer: %s", err)
            return self._error(
                user_input,
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I can't answer that right now.",
            )
        return self._speech(user_input, answer or "I'm not sure.")

    # -- small helpers --------------------------------------------------------

    def _unavailable_ids(self) -> frozenset[str]:
        """Entities that exist but cannot act, read fresh.

        The catalog deliberately caches structure and never state, and route()
        is pure, so availability is computed here and passed in. `unknown` is
        not included: that entity is alive, its value simply is not known yet.
        """
        return frozenset(
            entity.entity_id
            for entity in self.catalog.entities
            if (state := self.hass.states.get(entity.entity_id)) is None
            or state.state == STATE_UNAVAILABLE
        )

    def _speaker_area(self, user_input: conversation.ConversationInput) -> str | None:
        if user_input.device_id is None:
            return None
        from homeassistant.helpers import device_registry as dr

        device = dr.async_get(self.hass).async_get(user_input.device_id)
        return device.area_id if device else None

    def _speaker_area_name(
        self, user_input: conversation.ConversationInput
    ) -> str | None:
        area_id = self._speaker_area(user_input)
        if area_id is None:
            return None
        area = ar.async_get(self.hass).async_get_area(area_id)
        return area.name if area else None

    def _history(self, chat_log: conversation.ChatLog) -> list[dict[str, str]]:
        return [
            {"user": user_text, "assistant": assistant_text}
            for user_text, assistant_text in self._history_pairs(chat_log)
        ]

    def _history_pairs(self, chat_log: conversation.ChatLog) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        pending: str | None = None
        for content in chat_log.content:
            if isinstance(content, conversation.UserContent):
                pending = content.content
            elif isinstance(content, conversation.AssistantContent) and pending:
                pairs.append((pending, content.content or ""))
                pending = None
        # Drop the turn currently in flight, then keep the last few.
        return pairs[-MAX_HISTORY_TURNS:]

    def _speech(
        self,
        user_input: conversation.ConversationInput,
        text: str,
        continue_conversation: bool = False,
    ) -> intent.IntentResponse:
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        if continue_conversation:
            self.continue_conversation = True
        return response

    def _error(
        self,
        user_input: conversation.ConversationInput,
        code: intent.IntentResponseErrorCode,
        message: str,
    ) -> intent.IntentResponse:
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(code, message)
        return response


def _confirm_question(plan: Plan) -> str:
    """Phrase the confirmation.

    A script carries its meaning in its name, not its action, so "run the
    Disarm the alarm?" reads badly - name it directly instead.
    """
    target = plan.target.described
    if plan.domain in ("script", "scene"):
        return f"Do you want me to run {target}?"
    verb = (plan.action or "do that").replace("_", " ")
    return f"Do you want me to {verb} the {target}?"


def _wholly_failed(response: intent.IntentResponse) -> list[str]:
    """Names of the entities that refused, when *none* accepted.

    async_handle_states puts the matched area in success_results whatever
    becomes of the entities inside it, so response_type stays action_done and
    error_code stays unset even when every service call was rejected. The only
    honest signal is that failed_results holds entities and success_results
    holds none. A partial success is left alone - something did happen.
    """
    if not response.failed_results:
        return []
    if any(
        target.type == intent.IntentResponseTargetType.ENTITY
        for target in response.success_results
    ):
        return []
    return [
        target.name
        for target in response.failed_results
        if target.type == intent.IntentResponseTargetType.ENTITY
    ]


def _with_text(
    user_input: conversation.ConversationInput, text: str
) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        satellite_id=user_input.satellite_id,
        language=user_input.language,
        agent_id=user_input.agent_id,
        extra_system_prompt=user_input.extra_system_prompt,
    )


def _step_failed(result: intent.IntentResponse) -> bool:
    """An error response, or one where every entity refused."""
    return result.response_type is intent.IntentResponseType.ERROR or bool(
        _wholly_failed(result)
    )


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


__all__ = ["AgentSettings", "TypeSafeAgent", "build_request"]
