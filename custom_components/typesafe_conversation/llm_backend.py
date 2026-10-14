"""The things we still need a text-generating model for.

Everything about controlling the home is decided by Jev and executed by code.
The LLM is used only where a string genuinely has to be produced:

1. splitting a compound request into atomic commands,
2. answering a general-knowledge or prose question, and
3. filling in the fields of a script Jev already chose.

It is never given a tool that acts on the home and never told it can control
anything, so a slow or failing LLM can delay an answer but can never block or
corrupt a device action. When web search is turned on, the answer path may be
given read-only lookup tools (see tools.py) - nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp

from .const import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    ANSWER_TIMEOUT,
    BACKEND_OLLAMA,
    BACKEND_OPENAI_COMPAT,
    CHECK_MAX_TOKENS,
    FILL_MAX_TOKENS,
    LLM_KEEP_ALIVE,
    LOGGER,
    MAX_SUB_COMMANDS,
    MAX_TOOL_ROUNDS,
    PROMPT_LOG_CHARS,
    SPLIT_MAX_TOKENS,
    SPLIT_TIMEOUT,
    TOOL_TIMEOUT,
)

if TYPE_CHECKING:
    from .tools import AnswerTool

SPLIT_SYSTEM_PROMPT = """\
You split a smart-home voice command into atomic commands.

Rules:
- Output ONLY a JSON array of strings. No prose, no explanation, no markdown, \
no code fences.
- Each string must be a complete, self-contained command that names its own \
target.
- Distribute shared subjects and verbs: "turn off the lights and the fan" -> \
["turn off the lights", "turn off the fan"]
- Keep the user's original wording and their original order.
- Never invent a device, room, value or action that is not in the input.
- Do not split one action applied to several devices: "turn off all the \
lights" is ONE command.
- If the input is a single command, return a one-element array.

Input: turn off the kitchen lights and lock the front door
Output: ["turn off the kitchen lights", "lock the front door"]

Input: dim the living room lights to 30% and start the coffee maker
Output: ["dim the living room lights to 30%", "start the coffee maker"]

Input: turn off all of the lights in the house
Output: ["turn off all of the lights in the house"]

Input: set the thermostat to 21, close the blinds, and play some jazz
Output: ["set the thermostat to 21", "close the blinds", "play some jazz"]"""

ANSWER_SYSTEM_PROMPT = """\
You are the voice assistant for a Home Assistant smart home.

- Reply in plain spoken text. No markdown, no bullet lists, no emoji, no \
headings.
- Reply in the language the user spoke in.
- Be brief: one or two sentences, under 40 words, as if speaking aloud.
- You cannot control any device in this mode. Never claim to have turned \
anything on or off, and never promise to do something.
- Use the home state below when the question is about the home. If the answer \
is not in it, or you are not confident, say you do not know rather than \
guessing.
- Answer general-knowledge questions truthfully and briefly from your own \
knowledge.{lookup}

Current time: {local_time} on {weekday}. The speaker is in {speaker_area}.

Home state:
{home_state}"""

FILL_SYSTEM_PROMPT = """\
You fill in the details of a Home Assistant script the user asked to run.

Script: {title}
What it does: {description}

Reply with one JSON object matching this JSON schema, nothing else:
{schema}

- Give a field only when the request says it, or when the field's description \
tells you how to work it out (for example from the room the user is in).
- Leave a field out rather than guess. Never invent a time, number or name.
- If some values are already settled, give a settled field again only when the \
latest reply changes it. Never fill a settled field from a field's default.
- Times are 24-hour HH:MM:SS. "5.15 a.m." is 05:15:00, "quarter to seven in the \
evening" is 18:45:00, "noon" is 12:00:00, "midnight" is 00:00:00, "half four" is \
4:30 and "quarter past 8" is 8:15.
- A time said without AM/PM or a part of the day is whichever of the two comes \
next from now. At 3:44 PM, "4:30" is 16:30:00. At 10 PM, "6" is 06:00:00. At \
11 PM, "12:30" is 00:30:00.
- For a date or date-and-time field, the same rule sets the date: at 3:44 PM, \
"4:30" is today at 16:30; at 10 PM, "6" is tomorrow at 06:00. Never give a moment \
that has already passed unless the user asked for one.
- A weekday is its next occurrence; today counts only while the time asked for is \
still ahead.

The user is in: {speaker_area}
It is now {local_time}, {weekday}.
"""

ANSWER_LOOKUP_RULE = """
- You have tools that look up current information. Use them for anything \
current, live or that you are unsure of, such as news, schedules, scores, prices, \
weather or opening hours; answer stable general knowledge directly. Never say you \
cannot check or do not know something without looking it up first. If a lookup \
fails, say you could not check. Never read out web addresses or list your \
sources."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMBackendError(Exception):
    """Any failure talking to the configured LLM."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call the model asked for, in the same shape whichever server."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Turn:
    """One reply that may ask for tools."""

    text: str
    calls: list[ToolCall]
    assistant: dict[str, Any]
    """The reply as a message, to send back alongside the tools' results."""
    metrics: dict[str, Any]


class LLMBackend(ABC):
    """Splitting, answering and filling. Nothing else is ever asked of the LLM."""

    name: str

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str | None = None,
        answer_timeout: float = ANSWER_TIMEOUT,
    ) -> None:
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._answer_timeout = answer_timeout
        self._tools_supported = True
        """Cleared when the model refuses a request with tools, so later answers
        skip the failing round trip until the entry reloads."""

    async def async_warm_up(self) -> None:  # noqa: B027 - optional hook
        """Load the model ahead of need. Only Ollama does anything with this."""

    async def async_check_model(self) -> None:
        """Send the model one tiny request, so setup finds out if it is refused.

        A listed model can still be one the key may not use, and a variant
        such as ``vendor/model:online`` is never listed. Raises LLMBackendError
        when the server refuses it.
        """
        await self._chat(
            [{"role": "user", "content": "Reply with OK."}],
            max_tokens=CHECK_MAX_TOKENS,
            temperature=0,
            timeout=self._answer_timeout,
        )

    @abstractmethod
    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat completion.

        Returns the assistant's text and a normalized metrics dict. The two
        providers report different things - Ollama gives timings and token
        counts, an OpenAI-compatible endpoint gives token counts only - so
        each backend normalizes into shared key names and the caller never
        has to branch. Every backend supplies ``elapsed_s``, measured here
        rather than taken from the provider: it is what the user actually
        waits, and it is the only figure that also covers connection setup
        and a slow link.
        """

    @abstractmethod
    async def _chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        final: bool = False,
    ) -> _Turn:
        """Send a chat completion that offers tools.

        With ``final`` the model may not call any more of them and has to
        answer with what it has.
        """

    @staticmethod
    @abstractmethod
    def _tool_result(call: ToolCall, content: str) -> dict[str, Any]:
        """A tool's result, as the message this server expects."""

    async def split_compound(self, utterance: str) -> list[str]:
        """Break a compound request into atomic commands.

        Never raises. If the model is slow, unreachable, returns something that
        is not an array of commands, or more of them than we accept, the result
        is empty and nothing runs: running the whole sentence as one command, or
        only its first few parts, would act on a guess at what was meant.
        """
        messages = [
            {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ]
        try:
            raw, metrics = await self._chat(
                messages,
                max_tokens=SPLIT_MAX_TOKENS,
                temperature=0.0,
                timeout=SPLIT_TIMEOUT,
            )
        except LLMBackendError as err:
            LOGGER.warning("Could not split a compound request (%s)", err)
            return []
        _log_exchange("split", self.name, self._model, messages, raw, metrics)

        parts = _parse_string_array(raw)
        if not parts:
            LOGGER.warning("LLM split returned no usable array: %r", raw[:200])
            return []
        if len(parts) > MAX_SUB_COMMANDS:
            LOGGER.warning(
                "LLM split produced %s parts, more than the %s accepted",
                len(parts),
                MAX_SUB_COMMANDS,
            )
            return []
        LOGGER.debug("Split %r into %s", utterance, parts)
        return parts

    async def answer_freeform(
        self,
        utterance: str,
        history: list[tuple[str, str]],
        *,
        home_state: str,
        local_time: str,
        weekday: str,
        speaker_area: str | None,
        tools: list[AnswerTool] | None = None,
    ) -> str:
        """Answer a general or prose question in natural language.

        With ``tools``, the model may look things up first. A model that does
        not take tools still answers, without them.
        """
        tools = tools if tools and self._tools_supported else []
        system = ANSWER_SYSTEM_PROMPT.format(
            local_time=local_time,
            weekday=weekday,
            speaker_area=speaker_area or "an unknown room",
            home_state=home_state,
            lookup=ANSWER_LOOKUP_RULE if tools else "",
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for user_text, assistant_text in history:
            messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": utterance})

        if tools:
            return (await self._answer_with_tools(messages, tools)).strip()
        text, metrics = await self._chat(
            messages,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=ANSWER_TEMPERATURE,
            timeout=self._answer_timeout,
        )
        _log_exchange("answer", self.name, self._model, messages, text, metrics)
        return text.strip()

    async def _answer_with_tools(
        self, messages: list[dict[str, Any]], tools: list[AnswerTool]
    ) -> str:
        """Let the model look things up, a bounded number of times, then answer.

        Each round runs every call the model asked for at once and hands the
        results back. After MAX_TOOL_ROUNDS the model answers without tools,
        and the whole exchange shares one deadline so a voice answer cannot
        stall on a slow search.
        """
        by_name = {tool.name: tool for tool in tools}
        specs = [tool.spec() for tool in tools]
        deadline = self._answer_timeout + MAX_TOOL_ROUNDS * TOOL_TIMEOUT
        try:
            async with asyncio.timeout(deadline):
                for round_no in range(MAX_TOOL_ROUNDS + 1):
                    final = round_no == MAX_TOOL_ROUNDS
                    try:
                        turn = await self._chat_tools(
                            messages,
                            specs,
                            max_tokens=ANSWER_MAX_TOKENS,
                            temperature=ANSWER_TEMPERATURE,
                            timeout=self._answer_timeout,
                            final=final,
                        )
                    except LLMBackendError as err:
                        if round_no or str(err) not in ("HTTP 400", "HTTP 404"):
                            raise
                        # The model or server does not take tools. Answer
                        # without them, now and until the entry reloads.
                        LOGGER.warning(
                            "%s refused a request with tools (%s); answering "
                            "without web search",
                            self._model,
                            err,
                        )
                        self._tools_supported = False
                        return await self._answer_plain(messages)
                    _log_exchange(
                        "answer",
                        self.name,
                        self._model,
                        messages,
                        turn.text,
                        turn.metrics,
                    )
                    if not turn.calls or final:
                        return turn.text
                    results = await asyncio.gather(
                        *(_run(by_name, call) for call in turn.calls)
                    )
                    LOGGER.debug(
                        "Answer round %s looked up %s",
                        round_no + 1,
                        ", ".join(
                            f"{call.name} ({len(result)} chars)"
                            for call, result in zip(turn.calls, results, strict=True)
                        ),
                    )
                    messages.append(turn.assistant)
                    messages.extend(
                        self._tool_result(call, result)
                        for call, result in zip(turn.calls, results, strict=True)
                    )
        except TimeoutError as err:
            raise LLMBackendError(f"Timed out after {deadline:.0f}s") from err
        return ""

    async def _answer_plain(self, messages: list[dict[str, Any]]) -> str:
        """The answer without tools, and without the rule that offers them."""
        system = messages[0]["content"].replace(ANSWER_LOOKUP_RULE, "")
        messages = [{"role": "system", "content": system}, *messages[1:]]
        text, metrics = await self._chat(
            messages,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=ANSWER_TEMPERATURE,
            timeout=self._answer_timeout,
        )
        _log_exchange("answer", self.name, self._model, messages, text, metrics)
        return text

    async def fill_fields(
        self,
        utterance: str,
        *,
        title: str,
        description: str,
        schema: dict[str, Any],
        speaker_area: str | None,
        local_time: str,
        weekday: str,
        earlier: str | None = None,
        known: dict[str, Any] | None = None,
        asking: str | None = None,
    ) -> dict[str, Any]:
        """Fill in the fields of a script Jev already chose, as one JSON object.

        Raises LLMBackendError when the model cannot be reached. A reply that is
        not an object comes back empty, which the caller treats as nothing said.
        """
        system = FILL_SYSTEM_PROMPT.format(
            title=title,
            description=description or "No description.",
            schema=json.dumps(schema, separators=(",", ":")),
            speaker_area=speaker_area or "an unknown room",
            local_time=local_time,
            weekday=weekday,
        )
        lines = []
        if earlier:
            lines.append(f"Earlier: {earlier}")
        if known:
            lines.append(f"Already settled: {json.dumps(known)}")
        if asking:
            lines.append(f"You asked for: {asking}")
        request = "\n".join([*lines, f"Now: {utterance}"]) if lines else utterance
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request},
        ]
        raw, metrics = await self._chat(
            messages,
            max_tokens=FILL_MAX_TOKENS,
            temperature=0.0,
            timeout=self._answer_timeout,
            schema=schema,
        )
        _log_exchange("fill", self.name, self._model, messages, raw, metrics)
        return _parse_object(raw)


class OllamaBackend(LLMBackend):
    """Ollama's native chat endpoint."""

    name = BACKEND_OLLAMA

    keep_loaded = False
    """Opt-in. When set, every request asks Ollama to keep the model loaded, and
    the integration loads it at startup and pings it to keep it resident. Off by
    default: whether a model sits in memory is the server owner's call."""

    async def _send(
        self, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """POST to /api/chat. Returns the reply message and metrics."""
        payload = {"model": self._model, "stream": False, **payload}
        if self.keep_loaded:
            # Only on request. Otherwise the server's own idle setting decides.
            payload["keep_alive"] = LLM_KEEP_ALIVE
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        started = time.monotonic()
        data = await _post_json(
            self._session, f"{self._base_url}/api/chat", payload, headers, timeout
        )
        elapsed = time.monotonic() - started
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise LLMBackendError("Unexpected response shape from Ollama")
        return message, _ollama_metrics(data, elapsed)

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if schema is not None:
            # Ollama constrains the reply to this JSON schema.
            payload["format"] = schema
        message, metrics = await self._send(payload, timeout)
        return str(message.get("content") or ""), metrics

    async def _chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        final: bool = False,
    ) -> _Turn:
        payload: dict[str, Any] = {
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if not final:
            # Ollama has no tool_choice; leaving the tools out is how to say no.
            payload["tools"] = tools
        message, metrics = await self._send(payload, timeout)
        raw_calls = message.get("tool_calls") or []
        calls = [] if final else _calls(raw_calls)
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        return _Turn(str(message.get("content") or ""), calls, assistant, metrics)

    @staticmethod
    def _tool_result(call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "content": content, "tool_name": call.name}

    async def async_warm_up(self) -> None:
        """Load the model before it is needed, for users who opted in.

        A cold load of a large model can outlast the answer timeout, so paying
        for it in the background keeps the first general question fast.
        """
        if not self.keep_loaded:
            return
        try:
            _text, metrics = await self._chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
                timeout=max(self._answer_timeout, 120.0),
            )
            LOGGER.debug(
                "Ollama warm-up took %.2fs (load %.2fs)",
                metrics.get("elapsed_s", 0.0),
                metrics.get("load_s", 0.0),
            )
        except LLMBackendError as err:
            LOGGER.debug("Ollama warm-up failed: %s", err)


class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible /v1/chat/completions endpoint, such as OpenRouter."""

    name = BACKEND_OPENAI_COMPAT

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str | None = None,
        referer: str | None = None,
        title: str | None = None,
        answer_timeout: float = ANSWER_TIMEOUT,
    ) -> None:
        super().__init__(session, base_url, model, api_key, answer_timeout)
        self._referer = referer
        self._title = title

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "fields", "schema": schema, "strict": False},
            }
        message, metrics = await self._send(payload, timeout)
        return str(message.get("content") or ""), metrics

    async def _chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
        final: bool = False,
    ) -> _Turn:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Earlier turns may hold tool calls, so the tools stay declared;
            # on the last round the model may no longer call them.
            "tools": tools,
            "tool_choice": "none" if final else "auto",
        }
        message, metrics = await self._send(payload, timeout)
        raw_calls = message.get("tool_calls") or []
        calls = [] if final else _calls(raw_calls)
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content"),
        }
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        return _Turn(str(message.get("content") or ""), calls, assistant, metrics)

    @staticmethod
    def _tool_result(call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    async def _send(
        self, payload: dict[str, Any], timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """POST to /v1/chat/completions. Returns the reply message and metrics."""
        payload = {"model": self._model, "stream": False, **payload}
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # OpenRouter attributes traffic with these; harmless elsewhere.
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        started = time.monotonic()
        url = f"{self._base_url}/v1/chat/completions"
        try:
            data = await _post_json(self._session, url, payload, headers, timeout)
        except LLMBackendError as err:
            if "response_format" not in payload or str(err) != "HTTP 400":
                raise
            # Not every server or model takes a schema. The prompt carries it
            # too, so ask once more for plain JSON.
            del payload["response_format"]
            data = await _post_json(self._session, url, payload, headers, timeout)
        elapsed = time.monotonic() - started
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as err:
            raise LLMBackendError("Unexpected response shape from the server") from err
        if not isinstance(message, dict):
            raise LLMBackendError("Unexpected response shape from the server")
        return message, _openai_metrics(data, elapsed)


def _calls(raw: Any) -> list[ToolCall]:
    """The tool calls in a reply, normalised.

    OpenAI-style servers send arguments as a JSON string, Ollama as an object;
    either may be malformed, in which case the call goes ahead with none and
    the tool reports what it is missing.
    """
    calls: list[ToolCall] = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError, TypeError:
                args = {}
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=str(function["name"]),
                args=args if isinstance(args, dict) else {},
            )
        )
    return calls


async def _run(by_name: dict[str, AnswerTool], call: ToolCall) -> str:
    if (tool := by_name.get(call.name)) is None:
        return f"There is no tool named {call.name}."
    return await tool.run(call.args)


# --- metrics ------------------------------------------------------------
# Each backend normalizes its provider's response into these shared keys, so
# the log helper never branches on which provider answered. A key that the
# provider does not report is simply absent.
#
#   elapsed_s          always - measured client-side
#   prompt_tokens      Ollama prompt_eval_count   | OpenAI usage.prompt_tokens
#   completion_tokens  Ollama eval_count          | OpenAI usage.completion_tokens
#   load_s             Ollama only - a cold model load
#   prompt_s, eval_s   Ollama only - enables tokens/sec
#   total_s            Ollama only - the server's own view of the request


def _ns_to_s(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds."""
    if isinstance(value, (int, float)) and value > 0:
        return value / 1_000_000_000
    return None


def _ollama_metrics(data: dict[str, Any], elapsed: float) -> dict[str, Any]:
    out: dict[str, Any] = {"elapsed_s": elapsed}
    for key, name in (
        ("prompt_eval_count", "prompt_tokens"),
        ("eval_count", "completion_tokens"),
    ):
        if isinstance(data.get(key), int):
            out[name] = data[key]
    for key, name in (
        ("load_duration", "load_s"),
        ("prompt_eval_duration", "prompt_s"),
        ("eval_duration", "eval_s"),
        ("total_duration", "total_s"),
    ):
        if (seconds := _ns_to_s(data.get(key))) is not None:
            out[name] = seconds
    return out


def _openai_metrics(data: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """An OpenAI-compatible endpoint reports tokens but no timing.

    There is no local model to load, so there is no cold-load equivalent
    either; elapsed_s is the only timing available and it is ours.
    """
    out: dict[str, Any] = {"elapsed_s": elapsed}
    usage = data.get("usage")
    if isinstance(usage, dict):
        for key, name in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
        ):
            if isinstance(usage.get(key), int):
                out[name] = usage[key]
    return out


def _log_exchange(
    operation: str,
    backend: str,
    model: str,
    messages: list[dict[str, Any]],
    reply: str,
    metrics: dict[str, Any],
) -> None:
    """Record one LLM exchange at DEBUG.

    The system prompt embeds the home catalog - entity names, areas and
    current states - and home-assistant.log is what people paste into issue
    reports, so it is truncated here. The reply is short enough
    (ANSWER_MAX_TOKENS is 180) to log whole.
    """
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    lines = [
        f"llm {backend} {model} {operation}",
        f"  messages  : {len(messages)} (system {len(system)} chars, truncated below)",
        f"  system    : {system[:PROMPT_LOG_CHARS]!r}"
        + ("..." if len(system) > PROMPT_LOG_CHARS else ""),
        f"  reply     : {reply!r} ({len(reply)} chars)",
        f"  elapsed   : {metrics['elapsed_s']:.2f}s",
    ]
    if (load := metrics.get("load_s")) is not None:
        lines.append(f"  load      : {load:.2f}s  <- cold model load")
    for label, tokens_key, seconds_key in (
        ("prompt", "prompt_tokens", "prompt_s"),
        ("eval  ", "completion_tokens", "eval_s"),
    ):
        tokens = metrics.get(tokens_key)
        if tokens is None:
            continue
        seconds = metrics.get(seconds_key)
        if seconds:
            lines.append(
                f"  {label}    : {tokens} tok in {seconds:.2f}s "
                f"({tokens / seconds:.1f} tok/s)"
            )
        else:
            # An OpenAI-compatible endpoint reports no per-phase timing.
            lines.append(f"  {label}    : {tokens} tok")
    if (total := metrics.get("total_s")) is not None:
        lines.append(f"  total     : {total:.2f}s (server-side)")
    LOGGER.debug("\n".join(lines))


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(timeout):
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=False,
            ) as response:
                if 300 <= response.status < 400:
                    raise LLMBackendError(
                        f"The server redirected the request (HTTP {response.status}). "
                        "Check the URL, for example http:// against https://."
                    )
                if response.status >= 400:
                    # The body can echo the prompt, which holds the home's
                    # state, so it stays out of the error and the warning log.
                    body = await response.text()
                    LOGGER.debug(
                        "HTTP %s from %s: %s", response.status, url, body[:300]
                    )
                    raise LLMBackendError(f"HTTP {response.status}")
                return await response.json()
    except LLMBackendError:
        raise
    except TimeoutError as err:
        raise LLMBackendError(f"Timed out after {timeout}s") from err
    except aiohttp.ClientError as err:
        raise LLMBackendError(
            f"Could not reach the server ({type(err).__name__})"
        ) from err
    except json.JSONDecodeError as err:
        raise LLMBackendError(f"Response was not JSON: {err}") from err


def _parse_object(raw: str) -> dict[str, Any]:
    """The JSON object in a reply, or an empty one."""
    found = _OBJECT_RE.search(raw)
    for candidate in (*_candidates(raw), found.group(0) if found else ""):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError, TypeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _parse_string_array(raw: str) -> list[str]:
    """Get a list of strings out of whatever the model actually returned.

    Models wrap JSON in fences, prefix it with "Output:", or bury it in a
    sentence. Try the strict reading first, then progressively looser ones.
    """
    for candidate in _candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError, TypeError:
            continue
        if isinstance(parsed, list) and parsed:
            # One unusable item means the split as a whole can't be trusted.
            if all(isinstance(p, str) and p.strip() for p in parsed):
                return [p.strip() for p in parsed]
            return []
    return []


def _candidates(raw: str):
    text = raw.strip()
    yield text
    if (fenced := _FENCE_RE.search(text)) is not None:
        yield fenced.group(1)
    if (array := _ARRAY_RE.search(text)) is not None:
        yield array.group(0)


class LLMAuthError(LLMBackendError):
    """The server refused the API key."""


_URL_SUFFIXES = (
    "/v1/chat/completions",
    "/chat/completions",
    "/api/chat",
    "/api/tags",
    "/v1/models",
    "/v1",
)


def normalise_llm_url(backend: str, url: str) -> str:
    """The base address, from whatever was pasted.

    Each backend adds its own path (``/api/chat`` for Ollama,
    ``/v1/chat/completions`` otherwise), so a full endpoint or a trailing
    ``/v1`` - which is how OpenRouter documents its address - is trimmed back.
    """
    url = url.strip().rstrip("/")
    for suffix in _URL_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    if backend == BACKEND_OLLAMA and url.endswith("/api"):
        url = url[: -len("/api")]
    return url


async def async_list_models(
    session: aiohttp.ClientSession, backend: str, base_url: str, api_key: str | None
) -> list[str] | None:
    """The models the server offers, or None when it does not list them.

    Raises LLMAuthError for a refused key and LLMBackendError when the server
    cannot be reached, which is what the setup form needs to tell apart.
    """
    path = "/api/tags" if backend == BACKEND_OLLAMA else "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with session.get(
            base_url + path,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
            allow_redirects=False,
        ) as response:
            if response.status in (401, 403):
                raise LLMAuthError(f"HTTP {response.status}")
            if response.status == 404:
                return None
            if response.status >= 300:
                raise LLMBackendError(f"HTTP {response.status}")
            payload = await response.json(content_type=None)
    except LLMBackendError:
        raise
    except (aiohttp.ClientError, TimeoutError, ValueError) as err:
        raise LLMBackendError(
            f"Could not reach the server ({type(err).__name__})"
        ) from err
    if not isinstance(payload, dict):
        return None
    if backend == BACKEND_OLLAMA:
        items, key = payload.get("models"), "name"
    else:
        items, key = payload.get("data"), "id"
    if not isinstance(items, list):
        return None
    return sorted(
        {i[key] for i in items if isinstance(i, dict) and isinstance(i.get(key), str)}
    )


def same_model(a: str, b: str) -> bool:
    """Ollama lists ``qwen3:latest`` for a model asked for as ``qwen3``."""

    def bare(name: str) -> str:
        name = name.strip()
        return name[: -len(":latest")] if name.endswith(":latest") else name

    return bare(a) == bare(b)


def offers_model(backend: str, model: str, models: list[str]) -> bool:
    """Whether the server's list covers the model asked for.

    OpenRouter lists ``vendor/model`` but also serves variants of it such as
    ``vendor/model:online`` that it never lists; setup then sends the model a
    test request, which a misspelt variant fails. On Ollama the part after the
    colon is a tag naming a different download, so it must be listed itself.
    """
    if any(same_model(model, m) for m in models):
        return True
    base, sep, _variant = model.strip().rpartition(":")
    return (
        backend == BACKEND_OPENAI_COMPAT
        and bool(sep and base)
        and any(same_model(base, m) for m in models)
    )


def create_backend(
    session: aiohttp.ClientSession, settings: dict[str, Any]
) -> LLMBackend | None:
    """Build the configured backend, or None when no LLM is set up.

    Without an LLM the agent still handles every command and query; it just
    cannot split compound requests or answer general questions.
    """
    from .const import (
        CONF_LLM_API_KEY,
        CONF_LLM_BACKEND,
        CONF_LLM_BASE_URL,
        CONF_LLM_KEEP_LOADED,
        CONF_LLM_MODEL,
        CONF_LLM_REFERER,
        CONF_LLM_TIMEOUT,
        CONF_LLM_TITLE,
        DEFAULT_LLM_REFERER,
        DEFAULT_LLM_TITLE,
    )

    backend = settings.get(CONF_LLM_BACKEND)
    base_url = settings.get(CONF_LLM_BASE_URL)
    model = settings.get(CONF_LLM_MODEL)
    if not backend or not base_url or not model:
        return None
    # Entries saved before the URL was normalised may hold a pasted endpoint.
    base_url = normalise_llm_url(backend, base_url)

    timeout = float(settings.get(CONF_LLM_TIMEOUT) or ANSWER_TIMEOUT)

    if backend == BACKEND_OLLAMA:
        ollama = OllamaBackend(
            session, base_url, model, settings.get(CONF_LLM_API_KEY), timeout
        )
        ollama.keep_loaded = bool(settings.get(CONF_LLM_KEEP_LOADED))
        return ollama
    if backend == BACKEND_OPENAI_COMPAT:
        return OpenAICompatBackend(
            session,
            base_url,
            model,
            settings.get(CONF_LLM_API_KEY),
            settings.get(CONF_LLM_REFERER) or DEFAULT_LLM_REFERER,
            settings.get(CONF_LLM_TITLE) or DEFAULT_LLM_TITLE,
            timeout,
        )
    LOGGER.error("Unknown LLM backend %r", backend)
    return None


__all__ = [
    "LLMBackend",
    "LLMBackendError",
    "OllamaBackend",
    "OpenAICompatBackend",
    "create_backend",
]
