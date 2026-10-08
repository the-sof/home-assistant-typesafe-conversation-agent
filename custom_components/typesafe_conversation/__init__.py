"""The TypeSafe Conversation integration.

Puts a TypeSafe System One model in front of the LLM as the decision layer:
device commands and state queries resolve from typed answers in code, and only genuinely
generative work - splitting compound requests, answering general questions -
reaches an LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_API_KEY,
    CONF_API_TIMEOUT,
    CONF_BASE_URL,
    CONF_LLM_KEEP_LOADED,
    CONF_MODEL,
    CONF_SERVER_PROFILE,
    CONVERSATION_DOMAIN,
    DEFAULT_MODEL,
    DOMAIN,
    LLM_WARMUP_INTERVAL_SECONDS,
    TRACE_HISTORY,
)
from .entities import EntityCatalog
from .llm_backend import LLMBackend, create_backend
from .system_one import (
    ServerProfile,
    SystemOneAuthError,
    SystemOneClient,
    SystemOneError,
)

PLATFORMS = [Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class TypeSafeRuntimeData:
    """Everything one config entry needs at runtime."""

    client: SystemOneClient
    catalog: EntityCatalog
    llm: LLMBackend | None
    model: str
    questions_cache: tuple[int, bool, dict[str, Any]] | None = None
    """Shared across turns: the question set is a pure function of the catalog."""

    traces: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=TRACE_HISTORY)
    )
    """Recent request traces, newest last.

    Home Assistant defines conversation traces but nothing reads them back -
    there is no websocket command and no UI - so they are write-only. Keeping
    our own ring buffer is what makes the diagnostics download useful."""

    pending_fills: dict[str, Any] = field(default_factory=dict)
    """Scripts waiting on a missing field, by conversation id: the answer to
    "What time should I use?" arrives as a separate turn."""


type TypeSafeConfigEntry = ConfigEntry[TypeSafeRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Set up TypeSafe Conversation from a config entry."""
    session = async_get_clientsession(hass)
    client = SystemOneClient(
        session,
        entry.data.get(CONF_API_KEY),
        entry.data.get(CONF_MODEL, DEFAULT_MODEL),
        base_url=entry.data.get(CONF_BASE_URL),
        profile=_profile(entry),
    )

    try:
        await client.async_validate()
    except SystemOneAuthError as err:
        raise ConfigEntryAuthFailed(str(err)) from err
    except SystemOneError as err:
        raise ConfigEntryNotReady(
            f"Could not reach the System One server: {err}"
        ) from err

    store = hass.data.setdefault(DOMAIN, {})
    catalog: EntityCatalog | None = store.get("catalog")
    if catalog is None:
        # One catalog for the whole instance: it describes Home Assistant, not
        # any particular config entry.
        catalog = EntityCatalog(hass, CONVERSATION_DOMAIN)
        catalog.async_start()
        store["catalog"] = catalog

    llm = create_backend(session, {**entry.data})
    entry.runtime_data = TypeSafeRuntimeData(
        client=client,
        catalog=catalog,
        llm=llm,
        model=entry.data.get(CONF_MODEL, DEFAULT_MODEL),
    )

    if llm is not None and entry.data.get(CONF_LLM_KEEP_LOADED):
        # Opt-in only. Load the model in the background now, and ping it often
        # enough that the server never unloads it while this entry is set up.
        entry.async_create_background_task(
            hass, llm.async_warm_up(), "typesafe_llm_warmup", eager_start=False
        )

        async def _async_keep_loaded(_now: datetime) -> None:
            """A coroutine function on purpose: async_track_time_interval runs a
            plain sync callable in an executor thread, where
            hass.async_create_task is not safe to call."""
            await llm.async_warm_up()

        entry.async_on_unload(
            async_track_time_interval(
                hass,
                _async_keep_loaded,
                timedelta(seconds=LLM_WARMUP_INTERVAL_SECONDS),
                name="typesafe_llm_keep_loaded",
            )
        )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


def _profile(entry: ConfigEntry) -> ServerProfile:
    """The probed profile, with any timeout the user set in its place.

    Entries made before endpoint discovery have no profile and get the defaults,
    which describe TypeSafe's API - the only server they could point at.
    """
    profile = ServerProfile.from_dict(entry.data.get(CONF_SERVER_PROFILE))
    if (timeout := entry.data.get(CONF_API_TIMEOUT)) is not None:
        profile = replace(profile, timeout=float(timeout))
    return profile


async def async_unload_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded and not [
        other
        for other in hass.config_entries.async_entries(DOMAIN)
        if other.entry_id != entry.entry_id
    ]:
        # Last entry gone: stop listening for registry changes.
        store = hass.data.get(DOMAIN, {})
        if (catalog := store.pop("catalog", None)) is not None:
            catalog.async_stop()
    return unloaded


async def _async_update_listener(
    hass: HomeAssistant, entry: TypeSafeConfigEntry
) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


__all__ = ["TypeSafeConfigEntry", "TypeSafeRuntimeData"]
