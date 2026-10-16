"""Shared fixtures.

The answer fixtures under ``fixtures/answers`` (jev-1.13.0) and
``fixtures/answers_1_14`` (jev-1.14.0, the same utterances) are *real* recorded
Jev responses, captured by ``scripts/calibrate.py --record``. Routing is tested by
replaying them, so the tests describe how the model actually behaves rather
than how we imagine it behaves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.typesafe_conversation.entities import CatalogArea, CatalogEntity
from custom_components.typesafe_conversation.system_one import (
    SystemOneResponse,
    _parse_answer,
)

FIXTURES = Path(__file__).parent / "fixtures"
ANSWERS = FIXTURES / "answers"
ANSWERS_1_14 = FIXTURES / "answers_1_14"


def load_home() -> dict:
    return json.loads((FIXTURES / "home.json").read_text())


def build_catalog() -> tuple[tuple[CatalogEntity, ...], tuple[CatalogArea, ...]]:
    home = load_home()
    area_names = {a["id"]: a["name"] for a in home["areas"]}
    area_floors = {a["id"]: a.get("floor") for a in home["areas"]}
    areas = tuple(
        CatalogArea(area_id=a["id"], name=a["name"], floor_name=a.get("floor"))
        for a in home["areas"]
    )
    entities = tuple(
        CatalogEntity(
            entity_id=e["id"],
            name=e["name"],
            aliases=tuple(e["also"].split(", ")) if e.get("also") else (),
            area_id=e.get("area"),
            area_name=area_names.get(e.get("area")),
            floor_name=area_floors.get(e.get("area")),
            domain=e["domain"],
            device_class=(e.get("attrs") or {}).get("device_class"),
            supported_features=0,
        )
        for e in home["entities"]
    )
    return entities, areas


def load_response(name: str, answers: Path = ANSWERS) -> tuple[SystemOneResponse, dict]:
    """Load one recorded response, parsed into the client's answer types."""
    payload = json.loads((answers / f"{name}.json").read_text())
    body = payload["response"]
    response = SystemOneResponse(
        model=body["model"],
        answers={k: _parse_answer(k, v) for k, v in body["answers"].items()},
        input_tokens=body.get("usage", {}).get("input_tokens", 0),
        output_tokens=body.get("usage", {}).get("output_tokens", 0),
        latency_ms=0.0,
        raw=body,
    )
    return response, payload


@pytest.fixture(name="catalog_entities")
def catalog_entities_fixture() -> tuple[CatalogEntity, ...]:
    return build_catalog()[0]


@pytest.fixture(name="entities_by_id")
def entities_by_id_fixture(
    catalog_entities: tuple[CatalogEntity, ...],
) -> dict[str, CatalogEntity]:
    return {e.entity_id: e for e in catalog_entities}


@pytest.fixture(name="available_domains")
def available_domains_fixture(
    catalog_entities: tuple[CatalogEntity, ...],
) -> frozenset[str]:
    return frozenset(e.domain for e in catalog_entities)
