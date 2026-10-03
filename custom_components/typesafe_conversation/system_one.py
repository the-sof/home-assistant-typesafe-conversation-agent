"""Async client for the System One API.

TypeSafe's hosted Jev is the default, but any server that speaks the same API
works: Ollama serves local decision models on it, for instance. Servers differ
in limits the API does not report, so ``async_probe`` measures them once at
setup and the result travels as a ``ServerProfile``.

Deliberately not the ``typesafe-sdk`` package: it depends on ``httpx2``, which
Home Assistant does not ship, and the endpoint is a single POST. Using the
aiohttp session HA already manages keeps the integration dependency-free.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import aiohttp

from .const import (
    API_BACKOFF,
    API_MAX_RETRIES,
    API_TIMEOUT,
    CIRCUIT_FAILURE_THRESHOLD,
    CIRCUIT_RESET_SECONDS,
    DEFAULT_BASE_URL,
    LOGGER,
    MAX_CHOICE_OPTIONS,
    MEASURE_TIMEOUT,
    MODELS_PATH,
    PROBE_MAX_OPTIONS,
    PROBE_TIMEOUT,
    SYSTEM_ONE_PATH,
    TIMEOUT_MARGIN,
)

_REJECTED = frozenset({400, 413, 422})
"""Statuses meaning "this request breaks a rule", as opposed to an outage."""


class SystemOneError(Exception):
    """Any failure talking to the System One API."""


class SystemOneAuthError(SystemOneError):
    """The API key is missing, invalid, or lacks access."""


class SystemOneRequestError(SystemOneError):
    """The server refused this request, and retrying will not change that.

    Too many options, a prompt larger than the model's context, a model that is
    not a decision model or is not installed. ``str(err)`` carries the server's
    own explanation, which is usually exactly what the user needs to fix it.
    """


class SystemOneUnavailableError(SystemOneError):
    """Jev is rate limited, overloaded, unreachable, or the breaker is open."""


@dataclass(slots=True)
class ChoiceAnswer:
    """One Choice answer."""

    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def margin(self) -> float:
        """p(top) - p(second).

        Confidence and margin fail differently: a distribution can be peaked
        enough to look confident while two options remain effectively tied.
        """
        if len(self.probabilities) < 2:
            return 1.0
        top, second = sorted(self.probabilities.values(), reverse=True)[:2]
        return top - second


@dataclass(slots=True)
class ScoreAnswer:
    """One Score answer."""

    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


@dataclass(slots=True)
class NoulAnswer:
    """One Noul answer. Has no confidence - the value carries the uncertainty."""

    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(slots=True)
class SystemOneResponse:
    """A parsed System One response."""

    model: str
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def choice(self, key: str) -> ChoiceAnswer | None:
        answer = self.answers.get(key)
        return answer if isinstance(answer, ChoiceAnswer) else None

    def score(self, key: str) -> ScoreAnswer | None:
        answer = self.answers.get(key)
        return answer if isinstance(answer, ScoreAnswer) else None

    def noul(self, key: str) -> float | None:
        answer = self.answers.get(key)
        return answer.noul if isinstance(answer, NoulAnswer) else None


@dataclass(slots=True, frozen=True)
class ServerProfile:
    """What one server and model can take, as measured at setup.

    The defaults describe TypeSafe's hosted API, which is what every entry
    created before discovery existed points at.
    """

    max_options: int = MAX_CHOICE_OPTIONS
    """Most options one Choice may offer, escape options included."""

    timeout: float = API_TIMEOUT
    cold_load_s: float | None = None
    """How long the first answer took when the model had to be loaded."""

    typical_s: float | None = None
    """One request shaped like a real one - this home, the full question set -
    timed at setup. What a voice command actually costs on this server."""

    num_ctx: int | None = None
    """Context window per rendered prompt, when the server reports it."""

    base_url: str | None = None
    model: str | None = None
    """What was probed. A change of either makes the profile stale."""

    def matches(self, base_url: str, model: str) -> bool:
        return self.base_url == normalise_base_url(
            base_url
        ) and self.model == normalise_model(model)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_typical(self, typical_s: float) -> ServerProfile:
        """Fold in the timed request, and derive the default timeout from it.

        The model may have been unloaded since the last request - Ollama does
        that after its own idle period, which is left to the server - so the
        timeout allows for a cold load on top of a typical request.
        """
        budget = (self.cold_load_s or 0.0) + typical_s
        return replace(
            self,
            typical_s=round(typical_s, 2),
            timeout=max(API_TIMEOUT, math.ceil(TIMEOUT_MARGIN * budget)),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ServerProfile:
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def normalise_base_url(url: str | None) -> str:
    return (url or DEFAULT_BASE_URL).strip().rstrip("/")


def normalise_model(name: str) -> str:
    """Ollama lists ``nimble:latest`` for a model requested as ``nimble``."""
    name = name.strip()
    return name[: -len(":latest")] if name.endswith(":latest") else name


def _parse_answers(payload: dict[str, Any]) -> dict[str, Answer]:
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        raise SystemOneError("Malformed answers from the server")
    return {key: _parse_answer(key, value) for key, value in answers.items()}


def _parse_answer(key: str, payload: Any) -> Answer:
    """One answer, checked before the router can act on it.

    A NaN confidence or a choice missing from its own distribution would
    otherwise reach the thresholds as if it were a real judgement.
    """
    try:
        return _parse_typed_answer(payload)
    except (KeyError, TypeError, AttributeError, ValueError) as err:
        raise SystemOneError(f"Malformed answer for {key!r}") from err


def _parse_typed_answer(payload: dict[str, Any]) -> Answer:
    kind = payload.get("type")
    if kind == "choice":
        probabilities = _distribution(payload["probabilities"])
        choice = payload["choice"]
        if not isinstance(choice, str) or choice not in probabilities:
            raise ValueError("choice is not in its distribution")
        return ChoiceAnswer(
            choice=choice,
            probabilities=probabilities,
            confidence=_number(payload["confidence"], probability=True),
        )
    if kind == "score":
        return ScoreAnswer(
            score=_number(payload["score"]),
            legend=dict(payload.get("legend", {})),
            probabilities=_distribution(payload["probabilities"]),
            confidence=_number(payload["confidence"], probability=True),
        )
    if kind == "noul":
        return NoulAnswer(noul=_number(payload["noul"], probability=True))
    raise ValueError(f"unknown answer type {kind!r}")


def _distribution(raw: dict[str, Any]) -> dict[str, float]:
    return {k: _number(v, probability=True) for k, v in raw.items()}


def _number(value: Any, *, probability: bool = False) -> float:
    """A finite real, and between 0 and 1 for a probability. Never a bool."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (probability and not 0 <= value <= 1)
    ):
        raise ValueError(f"not a valid number: {value!r}")
    return float(value)


class SystemOneClient:
    """Talks to ``POST /v1/systemone``, with retries and a circuit breaker."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_key: str | None,
        model: str,
        *,
        base_url: str | None = None,
        profile: ServerProfile | None = None,
    ) -> None:
        self._session = session
        self._api_key = api_key or None
        self._model = model
        self._base = normalise_base_url(base_url)
        self.profile = profile or ServerProfile()
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        """A local server usually takes no key, so send one only if we have it."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def circuit_open(self) -> bool:
        """True while we are deliberately not calling the API."""
        if self._open_until and time.monotonic() >= self._open_until:
            # Half-open: let the next request through and see what happens.
            self._open_until = 0.0
            self._consecutive_failures = 0
        return bool(self._open_until)

    async def async_validate(self) -> list[str]:
        """Check the endpoint and key, and return the models it offers.

        TypeSafe answers ``{"models": [{"name": ...}]}``; Ollama and other
        OpenAI-style servers answer ``{"data": [{"id": ...}]}``. Both are read.
        """
        try:
            async with self._session.get(
                self._base + MODELS_PATH,
                headers=self._headers(),
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT),
            ) as response:
                _refuse_redirect(response)
                if response.status in (401, 403):
                    raise SystemOneAuthError("The API key was rejected")
                response.raise_for_status()
                payload = await response.json()
        except SystemOneError:
            raise
        except aiohttp.ClientError as err:
            raise SystemOneUnavailableError(str(err)) from err
        except TimeoutError as err:
            raise SystemOneUnavailableError("Timed out reaching the server") from err
        if not isinstance(payload, dict):
            raise SystemOneUnavailableError("Unexpected model list from the server")
        names = [
            m.get("name") for m in payload.get("models") or [] if isinstance(m, dict)
        ]
        names += [m.get("id") for m in payload.get("data") or [] if isinstance(m, dict)]
        return list(
            dict.fromkeys(normalise_model(n) for n in names if isinstance(n, str) and n)
        )

    async def async_decision_models(self, names: list[str]) -> list[str]:
        """Narrow a model list to decision models, where the server can say.

        Ollama lists every model it has, chat models included, and reports a
        ``decision`` capability through ``/api/show``. That endpoint is Ollama's
        own, so a server without it simply keeps the full list.
        """
        kept: list[str] = []
        for name in names:
            info = await self._show(name)
            if info is None:
                return names
            capabilities = info.get("capabilities")
            if not isinstance(capabilities, list) or "decision" in capabilities:
                kept.append(name)
        return kept or names

    async def async_probe(self) -> ServerProfile:
        """Measure what this server and model can take.

        Every step uses a one-word state, so accepted probes cost almost
        nothing. A probe that breaks a limit is rejected before any model loads,
        so the option cap is found by a binary search over rejections.

        Timing comes separately, from ``async_time_request``: a one-word state
        says nothing about how long a real request takes.

        Raises ``SystemOneRequestError`` with the server's own message when the
        model cannot answer at all - not installed, or not a decision model.
        """
        status, cold, detail = await self._probe(3)
        if status != 200:
            raise _probe_failure(status, detail)

        max_options = PROBE_MAX_OPTIONS
        status, _, detail = await self._probe(PROBE_MAX_OPTIONS)
        if status != 200:
            if status not in _REJECTED:
                raise _probe_failure(status, detail)
            accepted, rejected = 3, PROBE_MAX_OPTIONS
            while rejected - accepted > 1:
                middle = (accepted + rejected) // 2
                status, _, detail = await self._probe(middle)
                if status == 200:
                    accepted = middle
                elif status in _REJECTED:
                    rejected = middle
                else:
                    raise _probe_failure(status, detail)
            max_options = accepted

        num_ctx = None
        if (info := await self._show(self._model)) is not None:
            num_ctx = _num_ctx(info.get("parameters"))

        profile = ServerProfile(
            max_options=max_options,
            cold_load_s=round(cold, 2),
            num_ctx=num_ctx,
            base_url=self._base,
            model=normalise_model(self._model),
        )
        LOGGER.debug("Probed %s for %s: %s", self._base, self._model, profile)
        return profile

    async def async_time_request(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> float:
        """Send one real-sized request, and return how long it took.

        The answer is discarded. The state is the home as it is now, which the
        server has not seen before, so its prompt cache cannot flatter the
        figure the way repeating an identical request would.
        """
        body = {"model": self._model, "state": state, "questions": questions}
        started = time.monotonic()
        try:
            async with self._session.post(
                self._base + SYSTEM_ONE_PATH,
                json=body,
                headers=self._headers(),
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=MEASURE_TIMEOUT),
            ) as response:
                _refuse_redirect(response)
                if response.status != 200:
                    raise _probe_failure(response.status, await _error_detail(response))
        except SystemOneError:
            raise
        except aiohttp.ClientError as err:
            raise SystemOneUnavailableError(str(err)) from err
        except TimeoutError as err:
            raise SystemOneUnavailableError(
                f"A typical request took over {MEASURE_TIMEOUT:.0f}s"
            ) from err
        return time.monotonic() - started

    async def _probe(self, options: int) -> tuple[int, float, str]:
        body: dict[str, Any] = {
            "model": self._model,
            "state": "x",
            "questions": {
                "probe": {
                    "type": "choice",
                    "instructions": "Pick any option.",
                    "criteria": {f"o{i}": None for i in range(options)},
                }
            },
        }
        started = time.monotonic()
        try:
            async with self._session.post(
                self._base + SYSTEM_ONE_PATH,
                json=body,
                headers=self._headers(),
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT),
            ) as response:
                _refuse_redirect(response)
                detail = "" if response.status == 200 else await _error_detail(response)
                return response.status, time.monotonic() - started, detail
        except aiohttp.ClientError as err:
            raise SystemOneUnavailableError(str(err)) from err
        except TimeoutError as err:
            raise SystemOneUnavailableError(
                f"No answer within {PROBE_TIMEOUT:.0f}s"
            ) from err

    async def _show(self, model: str) -> dict[str, Any] | None:
        """Ollama's model metadata, or None from any server that lacks it."""
        try:
            async with self._session.post(
                self._base + "/api/show",
                json={"model": model},
                headers=self._headers(),
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT),
            ) as response:
                if response.status != 200:
                    return None
                payload = await response.json(content_type=None)
        except aiohttp.ClientError, TimeoutError, ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    async def async_ask(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> SystemOneResponse:
        """Send one state and every question, and return the typed answers."""
        if self.circuit_open:
            raise SystemOneUnavailableError("System One circuit breaker is open")

        # No keep_alive: how long a model stays loaded is the server owner's
        # call. Someone sharing the machine may well want it unloaded.
        body: dict[str, Any] = {
            "state": state,
            "model": self._model,
            "questions": questions,
        }
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(API_MAX_RETRIES):
            try:
                payload = await self._post(body)
            except SystemOneUnavailableError as err:
                last_error = err
                if attempt == API_MAX_RETRIES - 1:
                    break
                delay = (
                    err.retry_after
                    if isinstance(err, _RetryableError) and err.retry_after
                    else API_BACKOFF[attempt]
                )
                LOGGER.debug(
                    "Request attempt %s/%s failed (%s); retrying in %.2fs",
                    attempt + 1,
                    API_MAX_RETRIES,
                    err,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            except SystemOneError:
                # Auth and validation errors are not worth retrying.
                self._record_failure()
                raise

            try:
                answers = _parse_answers(payload)
            except SystemOneError:
                # A malformed answer is a failure like any other: retrying the
                # same request would not fix it, and the breaker should count it.
                self._record_failure()
                raise
            self._consecutive_failures = 0
            latency_ms = (time.monotonic() - started) * 1000
            usage = payload.get("usage", {})
            response = SystemOneResponse(
                model=payload.get("model", self._model),
                answers=answers,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                latency_ms=latency_ms,
                raw=payload,
            )
            LOGGER.debug(
                "%s answered %s questions in %.0fms (%s input tokens)",
                response.model,
                len(response.answers),
                latency_ms,
                response.input_tokens,
            )
            return response

        self._record_failure()
        raise SystemOneUnavailableError(
            str(last_error) if last_error else "System One unavailable"
        )

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._session.post(
                self._base + SYSTEM_ONE_PATH,
                json=body,
                headers=self._headers(),
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=self.profile.timeout),
            ) as response:
                _refuse_redirect(response)
                if response.status in (401, 403):
                    raise SystemOneAuthError("The API key was rejected")
                if response.status in _REJECTED or response.status == 404:
                    # Deterministic: the same request would fail the same way,
                    # so retrying only burns time. The server's message names
                    # the problem - too many options, a prompt larger than the
                    # model's context, a missing model - so log it verbatim.
                    detail = await _error_detail(response)
                    LOGGER.warning(
                        "%s rejected the request (%s): %s",
                        self._base,
                        response.status,
                        detail,
                    )
                    raise SystemOneRequestError(detail)
                if response.status in (429, 529):
                    raise _RetryableError(
                        f"The API returned {response.status}",
                        retry_after=_parse_retry_after(
                            response.headers.get("retry-after")
                        ),
                    )
                if response.status >= 500:
                    raise _RetryableError(f"The API returned {response.status}")
                response.raise_for_status()
                return await response.json()
        except SystemOneError:
            raise
        except aiohttp.ClientError as err:
            raise _RetryableError(str(err)) from err
        except TimeoutError as err:
            raise _RetryableError("Timed out talking to the System One API") from err

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
            self._open_until = time.monotonic() + CIRCUIT_RESET_SECONDS
            LOGGER.warning(
                "The API failed %s times in a row; pausing calls for %.0fs and "
                "serving the fallback agent instead",
                self._consecutive_failures,
                CIRCUIT_RESET_SECONDS,
            )


class _RetryableError(SystemOneUnavailableError):
    """A failure worth another attempt."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


async def _error_detail(response: aiohttp.ClientResponse) -> str:
    """The server's explanation, from ``{"error": ...}`` or plain text."""
    text = (await response.text()).strip()
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            if isinstance(value := payload.get(key), str) and value:
                text = value
                break
    status = f"HTTP {response.status}"
    return f"{text[:300]} ({status})" if text else status


def _refuse_redirect(response: aiohttp.ClientResponse) -> None:
    """Never follow a redirect with the API key attached.

    The commonest cause is an ``http://`` URL for a server that only answers on
    ``https://``, so say that rather than failing obscurely.
    """
    if 300 <= response.status < 400:
        raise SystemOneRequestError(
            f"The server redirected the request (HTTP {response.status}). "
            "Check the server URL, for example http:// against https://."
        )


def _probe_failure(status: int, detail: str) -> SystemOneError:
    if status in (401, 403):
        return SystemOneAuthError("The API key was rejected")
    if status in _REJECTED or status == 404:
        return SystemOneRequestError(detail)
    return SystemOneUnavailableError(detail)


def _num_ctx(parameters: Any) -> int | None:
    """Read ``num_ctx`` out of Ollama's plain-text ``parameters`` block."""
    if not isinstance(parameters, str):
        return None
    match = re.search(r"^\s*num_ctx\s+(\d+)\s*$", parameters, re.MULTILINE)
    return int(match.group(1)) if match else None


__all__ = [
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "ServerProfile",
    "SystemOneAuthError",
    "SystemOneClient",
    "SystemOneError",
    "SystemOneRequestError",
    "SystemOneResponse",
    "SystemOneUnavailableError",
    "normalise_base_url",
    "normalise_model",
]
