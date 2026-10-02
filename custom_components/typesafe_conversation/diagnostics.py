"""Diagnostics for the TypeSafe Conversation integration.

Home Assistant defines conversation traces but nothing reads them back, so the
per-request reasoning is otherwise only visible in the debug log. This surfaces
it through the integration's "Download diagnostics" button instead: the last
few requests with the route taken, why, and every answer's probability
distribution.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from homeassistant.components.diagnostics import REDACTED, async_redact_data
from homeassistant.core import HomeAssistant

from . import TypeSafeConfigEntry
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
)

# Base URLs are redacted too - the System One server's and the LLM's: a
# self-hosted one is usually a private hostname, and diagnostics get pasted into
# public issue threads. Redaction matches keys at any depth, so this also covers
# the URL recorded inside the cached server profile.
REDACT = {
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    "api_key",
    "token",
}


def _scrub(trace: dict[str, Any], secrets: list[str]) -> dict[str, Any]:
    """Blank the server addresses inside a failure reason.

    The reason quotes the server's own error message, which can name its host,
    and key-based redaction never looks inside strings.
    """
    reason = trace.get("reason")
    if not isinstance(reason, str):
        return trace
    for secret in secrets:
        reason = reason.replace(secret, REDACTED)
    return {**trace, "reason": reason}


def _addresses(data: dict[str, Any]) -> list[str]:
    """Each configured base URL and its hostname, longest first."""
    found: set[str] = set()
    for key in (CONF_BASE_URL, CONF_LLM_BASE_URL):
        if isinstance(url := data.get(key), str) and url:
            found.add(url.rstrip("/"))
            if host := urlsplit(url).hostname:
                found.add(host)
    return sorted(found, key=len, reverse=True)


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: TypeSafeConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    data = entry.runtime_data
    catalog = data.catalog

    by_domain: dict[str, int] = {}
    for entity in catalog.entities:
        by_domain[entity.domain] = by_domain.get(entity.domain, 0) + 1

    return {
        "config": async_redact_data({**entry.data}, REDACT),
        "subentries": [
            {"title": s.title, "data": async_redact_data({**s.data}, REDACT)}
            for s in entry.subentries.values()
        ],
        "model": data.model,
        "llm_backend": data.llm.name if data.llm else None,
        "client": {
            "circuit_open": data.client.circuit_open,
            # What discovery measured: the option cap, timeout, cold-load time
            # and context window. Usually the first thing to check when a local
            # model misbehaves.
            "server_profile": async_redact_data(data.client.profile.as_dict(), REDACT),
        },
        "catalog": {
            "generation": catalog.generation,
            "entities": len(catalog.entities),
            "areas": len(catalog.areas),
            "by_domain": dict(sorted(by_domain.items())),
            # Entity ids identify the home, so report shape rather than
            # contents. The traces below carry ids only where a decision
            # turned on one, which is the part worth debugging.
            "unassigned_to_an_area": sum(
                1 for e in catalog.entities if e.area_id is None
            ),
        },
        "questions_cached_for_generation": (
            data.questions_cache[0] if data.questions_cache else None
        ),
        "recent_requests": [
            _scrub(trace, _addresses(entry.data)) for trace in data.traces
        ],
    }
