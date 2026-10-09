"""The fields a script takes, and checking the values a language model filled in.

Scripts written for LLM agents commonly take fields - an alarm script wants a
time and a speaker - and running one without them does nothing useful. Jev
picks the script; the configured LLM fills its fields in one structured call;
this module turns the script's own field definitions into the JSON schema that
call asks for, and checks every value against the field's selector before the
script runs. A value that fails is treated as missing, never coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import selector, service

SCRIPT_DOMAIN = "script"


@dataclass(frozen=True, slots=True)
class ScriptField:
    key: str
    name: str
    description: str
    required: bool
    selector_config: dict[str, Any] | None

    @property
    def kind(self) -> str | None:
        """The selector type, such as ``time`` or ``select``."""
        return next(iter(self.selector_config), None) if self.selector_config else None

    @property
    def options(self) -> list[str]:
        """The values a select field accepts."""
        if self.kind != "select":
            return []
        raw = (self.selector_config or {}).get("select", {}).get("options") or []
        return [o["value"] if isinstance(o, dict) else str(o) for o in raw]

    def names_for(self, value: Any) -> list[str]:
        """How a user could say this select value: the value and its label."""
        raw = (self.selector_config or {}).get("select", {}).get("options") or []
        names = [str(value)]
        for option in raw:
            if isinstance(option, dict) and option.get("value") == value:
                names.append(str(option.get("label") or ""))
        return [n for n in names if n]

    def json_schema(self, area_ids: list[str] | None = None) -> dict[str, Any]:
        """What the language model is asked to produce for this field."""
        if self.kind == "area" and area_ids:
            schema: dict[str, Any] = {"type": "string", "enum": area_ids}
        else:
            schema = _json_type(
                self.kind, (self.selector_config or {}).get(self.kind or "")
            )
        about = " ".join(p for p in (self.name, self.description) if p)
        return {**schema, "description": about} if about else schema

    def check(self, value: Any) -> Any:
        """The value as the selector accepts it. Raises vol.Invalid."""
        if self.selector_config is None:
            if not isinstance(value, str) or not value.strip():
                raise vol.Invalid("expected text")
            return value.strip()
        return selector.selector(self.selector_config)(value)


@dataclass(frozen=True, slots=True)
class ScriptFields:
    entity_id: str
    service: str
    """The script's service name, which is its unique id, not always its object id."""
    title: str
    description: str
    fields: tuple[ScriptField, ...]

    def json_schema(self, area_ids: list[str] | None = None) -> dict[str, Any]:
        """Every field optional, so the model can leave out what was not said.

        Marking a field required in the schema would force a structured-output
        model to produce *something* - which is a guess. Required fields are
        enforced in ``check`` instead, by asking the user.
        """
        return {
            "type": "object",
            "properties": {f.key: f.json_schema(area_ids) for f in self.fields},
            "additionalProperties": False,
        }

    def check(self, values: Any) -> tuple[dict[str, Any], list[ScriptField]]:
        """The values that passed, and the required fields still missing."""
        filled: dict[str, Any] = {}
        if isinstance(values, dict):
            for field in self.fields:
                if values.get(field.key) in (None, ""):
                    continue
                try:
                    filled[field.key] = field.check(values[field.key])
                except vol.Invalid:
                    continue
        missing = [f for f in self.fields if f.required and f.key not in filled]
        return filled, missing


def script_service(hass: HomeAssistant, entity_id: str) -> str | None:
    """A script's service name: its registry unique id, else its object id."""
    if not entity_id.startswith(f"{SCRIPT_DOMAIN}."):
        return None
    if (entry := er.async_get(hass).async_get(entity_id)) and entry.unique_id:
        return entry.unique_id
    return entity_id.split(".", 1)[1]


def async_script_fields(hass: HomeAssistant, entity_id: str) -> ScriptFields | None:
    """A script's fields, or None when it has none.

    The script integration registers each script's description under its
    service name - the registry unique id - which is what Home Assistant's own
    LLM tools read. Nothing here is fetched; it is already in memory.
    """
    if (name := script_service(hass, entity_id)) is None:
        return None
    description = service.async_get_cached_service_description(
        hass, SCRIPT_DOMAIN, name
    )
    if not description or not description.get("fields"):
        return None
    fields = tuple(
        ScriptField(
            key=key,
            name=str(config.get("name") or key),
            description=str(config.get("description") or ""),
            required=bool(config.get("required")),
            selector_config=config.get("selector") or None,
        )
        for key, config in description["fields"].items()
        # Advanced fields are hidden from people in the UI, so they are not
        # something a spoken request would fill either.
        if isinstance(config, dict) and not config.get("advanced")
    )
    if not fields:
        return None
    state = hass.states.get(entity_id)
    title = (state and state.attributes.get("friendly_name")) or description.get("name")
    return ScriptFields(
        entity_id=entity_id,
        service=name,
        title=str(title or name),
        description=str(description.get("description") or ""),
        fields=fields,
    )


def _json_type(kind: str | None, config: Any) -> dict[str, Any]:
    """A JSON schema for one selector type. Anything unknown is plain text."""
    config = config if isinstance(config, dict) else {}
    match kind:
        case "select":
            raw = config.get("options") or []
            values = [o["value"] if isinstance(o, dict) else str(o) for o in raw]
            one = {"type": "string", "enum": values}
            return {"type": "array", "items": one} if config.get("multiple") else one
        case "number":
            schema: dict[str, Any] = {"type": "number"}
            if "min" in config:
                schema["minimum"] = config["min"]
            if "max" in config:
                schema["maximum"] = config["max"]
            return schema
        case "boolean":
            return {"type": "boolean"}
        case "time":
            return {
                "type": "string",
                "format": "time",
                "pattern": "^\\d{2}:\\d{2}(:\\d{2})?$",
            }
        case "date":
            return {"type": "string", "format": "date"}
        case "datetime":
            return {"type": "string", "format": "date-time"}
        case "duration":
            parts = ("days", "hours", "minutes", "seconds")
            return {
                "type": "object",
                "properties": {p: {"type": "integer", "minimum": 0} for p in parts},
                "additionalProperties": False,
            }
    return {"type": "string"}


def describe_values(values: dict[str, Any]) -> str:
    """The filled values in words, for "Done: Set an alarm (5:15 AM, upstairs)"."""
    return ", ".join(_spoken_value(v) for v in values.values() if v not in (None, ""))


def _spoken_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str) and (time := _clock(value)) is not None:
        return time
    if isinstance(value, dict):
        return " ".join(f"{n} {unit}" for unit, n in value.items() if n)
    if isinstance(value, list):
        return " and ".join(str(v) for v in value)
    return str(value)


def _clock(value: str) -> str | None:
    """'05:15:00' as '5:15 AM'; anything that is not a clock time, None."""
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"
