"""The catalog of entities Home Assistant exposes to Assist.

This is what we hand Jev as ``state``. Its logic deliberately mirrors the
private ``homeassistant.helpers.llm._get_exposed_entities`` - we do not import
it, because it is private and its shape has already moved once between
releases.

The structure (which entities exist, their names and areas) is cached and
rebuilt lazily when a registry or exposure event marks it dirty. Current
*state* is read fresh on every request, because subscribing to
``state_changed`` would rebuild the catalog constantly for no benefit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from operator import attrgetter
from typing import Any

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.components.homeassistant.exposed_entities import (
    async_listen_entity_updates,
)
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, State, callback
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    floor_registry as fr,
)
from homeassistant.helpers import (
    intent,
)
from homeassistant.util import dt as dt_util

from .const import LOGGER

# Attributes worth sending, per domain. Home Assistant's own helper uses one
# flat set for every domain; most of those values are noise outside the domain
# that produces them, and noise costs tokens in every question.
_DOMAIN_ATTRS: dict[str, tuple[str, ...]] = {
    "light": ("brightness",),
    "cover": ("current_position", "device_class"),
    "fan": ("percentage",),
    "climate": ("current_temperature", "temperature", "temperature_unit"),
    "media_player": ("volume_level", "media_title"),
    "sensor": ("unit_of_measurement", "device_class"),
    "number": ("unit_of_measurement", "device_class"),
    "humidifier": ("humidity", "current_humidity"),
    "binary_sensor": ("device_class",),
    "switch": ("device_class",),
    "water_heater": ("current_temperature", "temperature"),
    "vacuum": (),
}

# Domains the user can actually command.
CONTROLLABLE_DOMAINS = frozenset(
    {
        "light",
        "switch",
        "cover",
        "lock",
        "fan",
        "climate",
        "media_player",
        "scene",
        "script",
        "vacuum",
        "humidifier",
        "water_heater",
        "button",
        "input_boolean",
    }
)


@dataclass(slots=True, frozen=True)
class CatalogArea:
    """One area exposed to Assist."""

    area_id: str
    name: str
    floor_name: str | None


@dataclass(slots=True, frozen=True)
class CatalogEntity:
    """One entity exposed to Assist.

    ``supported_features`` is kept for code-side checks and is never
    serialized into the state we send Jev.
    """

    entity_id: str
    name: str
    aliases: tuple[str, ...]
    area_id: str | None
    area_name: str | None
    floor_name: str | None
    domain: str
    device_class: str | None
    supported_features: int

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class EntityCatalog:
    """Cached view of the entities exposed to the conversation assistant.

    Rebuilt lazily: registry and exposure events only set a dirty flag, so a
    bulk import or a startup storm costs one rebuild, not hundreds.
    """

    def __init__(self, hass: HomeAssistant, assistant: str) -> None:
        self.hass = hass
        self.assistant = assistant
        self._entities: tuple[CatalogEntity, ...] = ()
        self._areas: tuple[CatalogArea, ...] = ()
        self._by_id: dict[str, CatalogEntity] = {}
        self._dirty = True
        self._generation = 0
        self._entity_count_at_build = -1
        self._unsubs: list[CALLBACK_TYPE] = []

    # -- lifecycle ------------------------------------------------------------

    @callback
    def async_start(self) -> None:
        """Subscribe to everything that can change the catalog's structure."""
        self._unsubs = [
            async_listen_entity_updates(self.hass, self.assistant, self._mark_dirty),
            self.hass.bus.async_listen(
                er.EVENT_ENTITY_REGISTRY_UPDATED, self._mark_dirty
            ),
            self.hass.bus.async_listen(
                ar.EVENT_AREA_REGISTRY_UPDATED, self._mark_dirty
            ),
            # A device moving between areas changes its entities' areas without
            # touching the entity registry.
            self.hass.bus.async_listen(
                dr.EVENT_DEVICE_REGISTRY_UPDATED, self._mark_dirty
            ),
            self.hass.bus.async_listen(
                fr.EVENT_FLOOR_REGISTRY_UPDATED, self._mark_dirty
            ),
            # Entities keep appearing well after the integration is set up.
            self.hass.bus.async_listen(EVENT_HOMEASSISTANT_STARTED, self._mark_dirty),
        ]

    @callback
    def async_stop(self) -> None:
        while self._unsubs:
            self._unsubs.pop()()

    @callback
    def _mark_dirty(self, *_: Any) -> None:
        self._dirty = True

    # -- access ---------------------------------------------------------------

    @property
    def generation(self) -> int:
        """Bumped on every rebuild. Question sets are cached against this."""
        self._ensure_fresh()
        return self._generation

    @property
    def entities(self) -> tuple[CatalogEntity, ...]:
        self._ensure_fresh()
        return self._entities

    @property
    def areas(self) -> tuple[CatalogArea, ...]:
        self._ensure_fresh()
        return self._areas

    @property
    def domains(self) -> tuple[str, ...]:
        """Domains actually present, in a stable order."""
        self._ensure_fresh()
        return tuple(sorted({e.domain for e in self._entities}))

    def get(self, entity_id: str) -> CatalogEntity | None:
        self._ensure_fresh()
        return self._by_id.get(entity_id)

    def area_name(self, area_id: str) -> str | None:
        self._ensure_fresh()
        for area in self._areas:
            if area.area_id == area_id:
                return area.name
        return None

    def _ensure_fresh(self) -> None:
        # Never cache an empty catalog. Exposure settings load from storage
        # asynchronously, so a build during startup can legitimately see
        # nothing exposed - and because the entity *count* has not changed, the
        # guard below would then keep serving that empty result forever. An
        # empty home is the degenerate case anyway, so rebuilding is cheap.
        if not self._entities:
            self._rebuild()
            return
        # YAML and template entities have no registry entry, so they can appear
        # without firing any event we listen to. A count check is cheap
        # insurance against a permanently stale catalog.
        if (
            not self._dirty
            and len(self.hass.states.async_entity_ids()) == self._entity_count_at_build
        ):
            return
        self._rebuild()

    # -- build ----------------------------------------------------------------

    def _rebuild(self) -> None:
        hass = self.hass
        entity_registry = er.async_get(hass)
        device_registry = dr.async_get(hass)
        area_registry = ar.async_get(hass)
        floor_registry = fr.async_get(hass)

        entities: list[CatalogEntity] = []
        used_area_ids: set[str] = set()

        for state in sorted(hass.states.async_all(), key=attrgetter("name")):
            if not async_should_expose(hass, self.assistant, state.entity_id):
                continue

            entry = entity_registry.async_get(state.entity_id)
            device = (
                device_registry.async_get(entry.device_id)
                if entry is not None and entry.device_id is not None
                else None
            )

            # A registry entry with no name of its own yields an empty string
            # here, so filter before falling back to the state's name.
            names = [
                name.strip()
                for name in intent.async_get_entity_aliases(hass, entry, state=state)
                if name and name.strip()
            ]
            if not names:
                names = [state.name or state.entity_id]

            area_id: str | None = None
            if entry is not None and entry.area_id is not None:
                area_id = entry.area_id
            elif device is not None and device.area_id is not None:
                area_id = device.area_id

            area = area_registry.async_get_area(area_id) if area_id else None
            floor_name: str | None = None
            if area is not None and area.floor_id is not None:
                floor = floor_registry.async_get_floor(area.floor_id)
                floor_name = floor.name if floor is not None else None
            if area is not None:
                used_area_ids.add(area.id)

            entities.append(
                CatalogEntity(
                    entity_id=state.entity_id,
                    name=names[0],
                    aliases=tuple(names[1:]),
                    area_id=area.id if area else None,
                    area_name=area.name if area else None,
                    floor_name=floor_name,
                    domain=state.domain,
                    device_class=state.attributes.get("device_class"),
                    supported_features=state.attributes.get("supported_features", 0)
                    or 0,
                )
            )

        # Group by area so related entities sit next to each other; unassigned
        # entities sort last.
        entities.sort(key=lambda e: (e.area_name or "￿", e.domain, e.name))

        areas: list[CatalogArea] = []
        for area in sorted(area_registry.async_list_areas(), key=attrgetter("name")):
            if area.id not in used_area_ids:
                continue
            floor_name = None
            if area.floor_id is not None:
                floor = floor_registry.async_get_floor(area.floor_id)
                floor_name = floor.name if floor is not None else None
            areas.append(
                CatalogArea(area_id=area.id, name=area.name, floor_name=floor_name)
            )

        self._entities = tuple(entities)
        self._areas = tuple(areas)
        self._by_id = {e.entity_id: e for e in entities}
        self._entity_count_at_build = len(hass.states.async_entity_ids())
        self._dirty = False
        self._generation += 1

        LOGGER.debug(
            "Catalog rebuilt (gen %s): %s entities, %s areas, domains=%s",
            self._generation,
            len(self._entities),
            len(self._areas),
            ",".join(sorted({e.domain for e in entities})),
        )

    # -- serialization --------------------------------------------------------

    def snapshot(
        self, entities: Iterable[CatalogEntity] | None = None
    ) -> dict[str, Any]:
        """Render the catalog as the ``home`` object we send Jev.

        State is read fresh here, not at build time.
        """
        self._ensure_fresh()
        chosen = tuple(entities) if entities is not None else self._entities
        chosen_area_ids = {e.area_id for e in chosen if e.area_id}

        return {
            "areas": [
                {
                    "id": a.area_id,
                    "name": a.name,
                    **({"floor": a.floor_name} if a.floor_name else {}),
                }
                for a in self._areas
                if a.area_id in chosen_area_ids
            ],
            "entities": [self._entity_snapshot(e) for e in chosen],
        }

    def _entity_snapshot(self, entity: CatalogEntity) -> dict[str, Any]:
        state = self.hass.states.get(entity.entity_id)
        info: dict[str, Any] = {
            "id": entity.entity_id,
            "name": entity.name,
            "domain": entity.domain,
            "state": _render_state(self.hass, entity, state),
        }
        if entity.aliases:
            info["also"] = ", ".join(entity.aliases)
        if entity.area_id:
            info["area"] = entity.area_id
        if state is not None and (attrs := _pick_attributes(entity.domain, state)):
            info["attrs"] = attrs
        return info

    def summarize(self, entities: Iterable[CatalogEntity] | None = None) -> str:
        """Plain-text rendering for the LLM's system prompt.

        One catalog, two consumers - the LLM gets the same facts as Jev, just
        in a shape a text model reads more comfortably.
        """
        self._ensure_fresh()
        chosen = tuple(entities) if entities is not None else self._entities
        lines: list[str] = []
        for entity in chosen:
            state = self.hass.states.get(entity.entity_id)
            where = f", {entity.area_name}" if entity.area_name else ""
            lines.append(
                f"{entity.name} ({entity.domain}{where}): "
                f"{_render_state(self.hass, entity, state)}"
            )
        return "\n".join(lines)


def _render_state(
    hass: HomeAssistant, entity: CatalogEntity, state: State | None
) -> str:
    """Format a state the way a person would read it."""
    if state is None:
        return "unavailable"

    if entity.domain == "sensor":
        # Respect the user's configured display precision.
        from homeassistant.components.sensor import async_rounded_state

        try:
            return async_rounded_state(hass, entity.entity_id, state)
        except Exception:
            return state.state

    if (
        entity.device_class == "timestamp"
        and state.state
        and (parsed := dt_util.parse_datetime(state.state)) is not None
    ):
        return dt_util.as_local(parsed).isoformat()

    return state.state


def _pick_attributes(domain: str, state: State) -> dict[str, Any]:
    """Keep only the attributes that matter for this domain."""
    wanted = _DOMAIN_ATTRS.get(domain)
    if not wanted:
        return {}
    picked: dict[str, Any] = {}
    for name in wanted:
        if (value := state.attributes.get(name)) is None:
            continue
        if isinstance(value, (Enum, Decimal)):
            value = str(value)
        picked[name] = value
    return picked


def async_get_catalog(
    hass: HomeAssistant, assistant: str, store: dict[str, Any]
) -> EntityCatalog:
    """Return the shared catalog, creating and starting it on first use."""
    catalog: EntityCatalog | None = store.get("catalog")
    if catalog is None:
        catalog = EntityCatalog(hass, assistant)
        catalog.async_start()
        store["catalog"] = catalog
    return catalog


__all__ = [
    "CONTROLLABLE_DOMAINS",
    "CatalogArea",
    "CatalogEntity",
    "EntityCatalog",
    "async_get_catalog",
]
