"""The two things we still need a text-generating model for.

Everything about controlling the home is decided by Jev and executed by code.
The LLM is used only where a string genuinely has to be produced:

1. splitting a compound request into atomic commands, and
2. answering a general-knowledge or prose question.

It is never given tools and never told it can control anything, so a slow or
failing LLM can delay an answer but can never block or corrupt a device action.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from .const import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    ANSWER_TIMEOUT,
    BACKEND_OLLAMA,
    BACKEND_OPENAI_COMPAT,
    LLM_KEEP_ALIVE,
    LOGGER,
    MAX_SUB_COMMANDS,
    PROMPT_LOG_CHARS,
    SPLIT_MAX_TOKENS,
    SPLIT_TIMEOUT,
)

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
- Be brief: one or two sentences, under 40 words, as if speaking aloud.
- You cannot control any device in this mode. Never claim to have turned \
anything on or off, and never promise to do something.
- Use the home state below when the question is about the home. If the answer \
is not in it, or you are not confident, say you do not know rather than \
guessing.
- Answer general-knowledge questions truthfully and briefly from your own \
knowledge.

Current time: {local_time} on {weekday}. The speaker is in {speaker_area}.

Home state:
{home_state}"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class LLMBackendError(Exception):
    """Any failure talking to the configured LLM."""


class LLMBackend(ABC):
    """Two operations. Nothing else is ever asked of the LLM."""

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

    async def async_warm_up(self) -> None:  # noqa: B027 - optional hook
        """Load the model ahead of need. Only Ollama does anything with this."""

    @abstractmethod
    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
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

    async def split_compound(self, utterance: str) -> list[str]:
        """Break a compound request into atomic commands.

        Never raises. If the model is slow, unreachable, or returns something
        that is not a JSON array, we fall back to treating the utterance as a
        single command - which is exactly what would have happened without the
        compound question. A failure here must not cost the user their command.
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
            return [utterance]
        _log_exchange("split", self.name, self._model, messages, raw, metrics)

        parts = _parse_string_array(raw)
        if not parts:
            LOGGER.warning("LLM split returned no usable array: %r", raw[:200])
            return [utterance]
        if len(parts) > MAX_SUB_COMMANDS:
            LOGGER.warning(
                "LLM split produced %s parts; keeping the first %s",
                len(parts),
                MAX_SUB_COMMANDS,
            )
            parts = parts[:MAX_SUB_COMMANDS]
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
    ) -> str:
        """Answer a general or prose question in natural language."""
        system = ANSWER_SYSTEM_PROMPT.format(
            local_time=local_time,
            weekday=weekday,
            speaker_area=speaker_area or "an unknown room",
            home_state=home_state,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for user_text, assistant_text in history:
            messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": utterance})

        text, metrics = await self._chat(
            messages,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=ANSWER_TEMPERATURE,
            timeout=self._answer_timeout,
        )
        _log_exchange("answer", self.name, self._model, messages, text, metrics)
        return text.strip()


class OllamaBackend(LLMBackend):
    """Ollama's native chat endpoint."""

    name = BACKEND_OLLAMA

    keep_loaded = False
    """Opt-in. When set, every request asks Ollama to keep the model loaded, and
    the integration loads it at startup and pings it to keep it resident. Off by
    default: whether a model sits in memory is the server owner's call."""

    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
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
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as err:
            raise LLMBackendError(f"Unexpected Ollama response: {data}") from err
        return text, _ollama_metrics(data, elapsed)

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
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # OpenRouter attributes traffic with these; harmless elsewhere.
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        started = time.monotonic()
        data = await _post_json(
            self._session,
            f"{self._base_url}/v1/chat/completions",
            payload,
            headers,
            timeout,
        )
        elapsed = time.monotonic() - started
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as err:
            raise LLMBackendError(f"Unexpected response: {data}") from err
        return text, _openai_metrics(data, elapsed)


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
    messages: list[dict[str, str]],
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
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise LLMBackendError(f"HTTP {response.status}: {body[:300]}")
                return await response.json()
    except LLMBackendError:
        raise
    except TimeoutError as err:
        raise LLMBackendError(f"Timed out after {timeout}s") from err
    except aiohttp.ClientError as err:
        raise LLMBackendError(str(err)) from err
    except json.JSONDecodeError as err:
        raise LLMBackendError(f"Response was not JSON: {err}") from err


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
        if isinstance(parsed, list):
            items = [p.strip() for p in parsed if isinstance(p, str) and p.strip()]
            if items:
                return items
    return []


def _candidates(raw: str):
    text = raw.strip()
    yield text
    if (fenced := _FENCE_RE.search(text)) is not None:
        yield fenced.group(1)
    if (array := _ARRAY_RE.search(text)) is not None:
        yield array.group(0)


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
