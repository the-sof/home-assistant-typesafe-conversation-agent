"""Constants for the TypeSafe Conversation integration.

The integration talks to TypeSafe's System One API, which serves a family of
models. Jev is the one available today and the default; nothing outside
DEFAULT_MODEL assumes it, so pointing CONF_MODEL at a later model is a config
change rather than a code change.
"""

from __future__ import annotations

import logging
from typing import Final

DOMAIN: Final = "typesafe_conversation"

CONVERSATION_DOMAIN: Final = "conversation"
"""Home Assistant's conversation domain, which is the assistant entities are
exposed to.

Imported as a constant rather than read off the ``conversation`` module: this
package has its own ``conversation.py`` platform, and forwarding the platform
setup rebinds that name on the package, so ``conversation.DOMAIN`` inside this
integration can silently become our own domain instead."""
LOGGER: Final = logging.getLogger(__package__)

# --- TypeSafe System One ------------------------------------------------------
DEFAULT_BASE_URL: Final = "https://api.typesafe.ai"
"""Any server speaking the System One API works; this is TypeSafe's hosted one."""
SYSTEM_ONE_PATH: Final = "/v1/systemone"
MODELS_PATH: Final = "/v1/models"
TYPESAFE_API_URL: Final = DEFAULT_BASE_URL + SYSTEM_ONE_PATH
TYPESAFE_MODELS_URL: Final = DEFAULT_BASE_URL + MODELS_PATH
TYPESAFE_CONSOLE_URL: Final = "https://console.typesafe.ai/"
DEFAULT_MODEL: Final = "jev-latest"
"""Jev is the only System One model today. An alias, so it follows releases."""
API_TIMEOUT: Final = 6.0
API_MAX_RETRIES: Final = 3
API_BACKOFF: Final = (0.25, 0.75, 2.0)

# --- Endpoint discovery -------------------------------------------------------
# The System One API does not report its limits, so they are probed once at
# setup and cached on the entry. A probe that breaks a limit is rejected before
# the server loads the model, so finding a cap costs milliseconds, not loads.
PROBE_MAX_OPTIONS: Final = 255
"""The most options worth asking for. TypeSafe accepts this many."""
PROBE_TIMEOUT: Final = 180.0
"""A probe may have to wait for a cold model load."""
MEASURE_TIMEOUT: Final = 600.0
"""The timed, real-sized request at setup. A CPU-only server can take minutes."""
MEASURE_UTTERANCE: Final = "what time is it"
"""Asked once at setup to time a request shaped like a real one. Never acted on."""
TIMEOUT_MARGIN: Final = 1.5
"""Default request timeout: this many times (cold load + a typical request)."""
BEAM_WIDTH: Final = 2
"""Domain/area paths kept when a home is too large for one entity question."""

# Circuit breaker: after this many consecutive failures, stop calling the API for
# CIRCUIT_RESET_SECONDS and serve the fallback ladder instead.
CIRCUIT_FAILURE_THRESHOLD: Final = 3
CIRCUIT_RESET_SECONDS: Final = 60.0

# --- Config keys -------------------------------------------------------------
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_BASE_URL: Final = "base_url"
CONF_API_TIMEOUT: Final = "api_timeout"
CONF_SERVER_PROFILE: Final = "server_profile"
CONF_LLM_BACKEND: Final = "llm_backend"
CONF_LLM_BASE_URL: Final = "llm_base_url"
CONF_LLM_MODEL: Final = "llm_model"
CONF_LLM_API_KEY: Final = "llm_api_key"
CONF_LLM_REFERER: Final = "llm_referer"
CONF_LLM_TITLE: Final = "llm_title"
CONF_BYPASS_LOCAL_INTENTS: Final = "bypass_local_intents"
CONF_INLINE_ENTITY_DESCRIPTIONS: Final = "inline_entity_descriptions"
CONF_ALWAYS_CONFIRM_RISKY: Final = "always_confirm_risky"
CONF_LLM_TIMEOUT: Final = "llm_timeout"
CONF_LLM_KEEP_LOADED: Final = "llm_keep_loaded"

DEFAULT_ALWAYS_CONFIRM_RISKY: Final = True
"""Ask before unlocking or opening the house, however sure the model is.

The model answers "unlock the front door" at confidence 1.0, so the confidence gate
below would let it through silently. One extra turn is cheap; an unlock the
user did not intend is not. Turn this off to get the pure confidence gate."""

BACKEND_OLLAMA: Final = "ollama"
BACKEND_OPENAI_COMPAT: Final = "openai_compatible"

DEFAULT_OLLAMA_URL: Final = "http://localhost:11434"
DEFAULT_OPENAI_COMPAT_URL: Final = "https://openrouter.ai/api"
DEFAULT_LLM_REFERER: Final = (
    "https://github.com/the-sof/home-assistant-typesafe-conversation-agent"
)
DEFAULT_LLM_TITLE: Final = "HA TypeSafe Conversation"

# --- LLM behaviour -----------------------------------------------------------
SPLIT_TIMEOUT: Final = 4.0
LLM_KEEP_ALIVE: Final = "30m"
LLM_WARMUP_INTERVAL_SECONDS: Final = 20 * 60
"""Only for users who opt in to keeping the Ollama model loaded. By default the
integration never asks a server to keep a model in memory."""
ANSWER_TIMEOUT: Final = 30.0
"""Seconds to wait for a freeform answer.

A large local model can take well over the old 20s, especially on the first
call after a restart. Raise it with CONF_LLM_TIMEOUT, or point the LLM at a
smaller model - this path is only used for prose, so it does not need to be
the same model you would pick for reasoning."""
SPLIT_MAX_TOKENS: Final = 200
ANSWER_MAX_TOKENS: Final = 180
FILL_MAX_TOKENS: Final = 300
"""Room for a script's field values as JSON; they are short."""
PENDING_FILL_SECONDS: Final = 90
"""How long the agent waits for the answer to "What time should I set it for?"."""
ANSWER_TEMPERATURE: Final = 0.3
MAX_SUB_COMMANDS: Final = 6

PROMPT_LOG_CHARS: Final = 200
"""How much of a system prompt to write to the debug log.

The freeform prompt embeds the home catalog - entity names, areas and current
states - and home-assistant.log is what gets pasted into issue reports."""

# --- Catalog -----------------------------------------------------------------
MAX_CHOICE_OPTIONS: Final = 255
"""Options per Choice for entries set up before endpoint discovery existed.

Those entries all point at TypeSafe, which accepts 255. Newer entries use the
cap discovered for their own server instead."""
MAX_HISTORY_TURNS: Final = 2
CATALOG_SUMMARY_MAX_ENTITIES: Final = 120
TRACE_HISTORY: Final = 20
"""How many recent request traces to keep for the diagnostics download."""

# --- Routing thresholds ------------------------------------------------------
# Every value below is a first guess calibrated from the TypeSafe demo's
# described behaviour. Re-fit them with scripts/calibrate.py against a live key
# on a real home before relying on them.
MIN_MARGIN: Final = 0.15
"""Minimum p(top) - p(second) for a Choice to count as solid."""

CONF_ACT_TERSE: Final = 0.75
"""At or above this, act and acknowledge briefly."""

CONF_ACT_EXPLICIT: Final = 0.50
"""At or above this, act but name the target explicitly in the speech."""

CONF_CLARIFY_FLOOR: Final = 0.30
"""Below this a target is not worth clarifying; fall back instead."""

T_CATEGORY_CANCEL: Final = 0.60
T_CATEGORY_INFORMATION: Final = 0.55
T_CATEGORY_UNCLEAR: Final = 0.55
T_CATEGORY_QUERY: Final = 0.45
T_CATEGORY_COMPOUND: Final = 0.50
T_QUERY_KIND: Final = 0.50
T_DOMAIN: Final = 0.50
T_ENTITY: Final = 0.60
T_AREA: Final = 0.55
T_SCOPE: Final = 0.55
T_SCOPE_WHOLE_HOUSE: Final = 0.60
T_ACTION: Final = 0.40
"""Kept for reference; the action gate uses T_ACTION_PROBABILITY instead."""

T_ACTION_PROBABILITY: Final = 0.55
"""Minimum probability for the chosen action.

Gating the action on *probability* rather than confidence, because confidence is
derived as (n * p_top - 1) / (n - 1) and so depends on how many options the
question offered. `action_script` has two options (run / not_targeted), so
p=0.66 scores only 0.31 confidence, while the same 0.66 on the seven-option
`action_light` scores 0.60. Thresholding confidence therefore systematically
under-trusts the small-option domains - scripts, scenes and buttons - which on a
script-heavy home is most of the traffic. Probability is comparable across
option counts."""

# Noul dead bands. Between the low and high value we treat the answer as "no"
# and log for calibration, because a Noul near 0.5 means the model splits
# yes/no - not that the condition half-holds.
NOUL_COMPOUND_HIGH: Final = 0.70
NOUL_COMPOUND_LOW: Final = 0.45
NOUL_HERE_RELATIVE: Final = 0.60
NOUL_RISKY: Final = 0.50
"""The risky question alone decides whether to confirm.

It used to be ANDed with an allowlist of risky actions, which exempted every
script: a script's action is always "run", so no script could reach the gate
whatever it did. Security actions are commonly implemented as scripts, and
those would have run unconfirmed. Measured against jev-1.13.0, the question
separates them on its own: "disarm the alarm" scores 0.95 and "open the
driveway gate" 0.96, while "arm the alarm in home mode" scores 0.03."""

NOUL_RISKY_EXPLICIT: Final = 0.30
"""Lower bar when the action itself is literally an unlock, open or disarm.

The allowlist is a floor now, never a filter - it can only make the gate fire
more readily, never suppress it."""

# A risky action (unlocking, opening an exterior door) needs more than the
# normal confidence before we do it without asking.
T_RISKY_ACTION: Final = 0.85
T_RISKY_TARGET: Final = 0.75
RISKY_ACTIONS: Final = frozenset({"unlock", "open", "disarm"})
RISKY_DOMAINS: Final = frozenset({"lock", "cover"})
RISKY_DEVICE_CLASSES: Final = frozenset({"garage", "gate", "door", "window"})

# Fall back to the LLM for a freeform answer if the category leaned at all
# towards something the LLM could answer.
T_LLM_LEAN: Final = 0.25

# --- Relative-change magnitudes (percentage points) --------------------------
MAGNITUDE_STEPS: Final = {
    "slight": 10,
    "moderate": 25,
    "large": 50,
    "maximum": 100,
}
