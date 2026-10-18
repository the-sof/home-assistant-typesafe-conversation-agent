"""The catalog, against a real Home Assistant instance."""

from __future__ import annotations

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.typesafe_conversation.entities import EntityCatalog


async def _setup(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    await hass.async_block_till_done()


async def test_only_exposed_entities_are_in_the_catalog(hass: HomeAssistant):
    await _setup(hass)
    registry = er.async_get(hass)
    shown = registry.async_get_or_create("light", "demo", "shown")
    hidden = registry.async_get_or_create("light", "demo", "hidden")
    hass.states.async_set(shown.entity_id, "on", {"friendly_name": "Shown Light"})
    hass.states.async_set(hidden.entity_id, "off", {"friendly_name": "Hidden Light"})
    async_expose_entity(hass, conversation.DOMAIN, shown.entity_id, True)
    async_expose_entity(hass, conversation.DOMAIN, hidden.entity_id, False)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    ids = {e.entity_id for e in catalog.entities}
    assert shown.entity_id in ids
    assert hidden.entity_id not in ids


async def test_entity_takes_its_area_and_state_is_read_fresh(hass: HomeAssistant):
    await _setup(hass)
    areas = ar.async_get(hass)
    kitchen = areas.async_create("Kitchen")
    registry = er.async_get(hass)
    entry = registry.async_get_or_create("switch", "demo", "coffee")
    registry.async_update_entity(entry.entity_id, area_id=kitchen.id)
    hass.states.async_set(entry.entity_id, "off", {"friendly_name": "Coffee Maker"})
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    (found,) = [e for e in catalog.entities if e.entity_id == entry.entity_id]
    assert found.area_id == kitchen.id
    assert found.area_name == "Kitchen"
    assert catalog.snapshot()["entities"][0]["state"] == "off"

    # Structure is cached, but state must not be.
    generation = catalog.generation
    hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Coffee Maker"})
    assert catalog.snapshot()["entities"][0]["state"] == "on"
    assert catalog.generation == generation, "a state change must not rebuild"


async def test_registry_change_invalidates_the_catalog(hass: HomeAssistant):
    await _setup(hass)
    registry = er.async_get(hass)
    first = registry.async_get_or_create("light", "demo", "one")
    hass.states.async_set(first.entity_id, "on", {"friendly_name": "One"})
    async_expose_entity(hass, conversation.DOMAIN, first.entity_id, True)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    catalog.async_start()
    assert len(catalog.entities) == 1
    generation = catalog.generation

    second = registry.async_get_or_create("light", "demo", "two")
    hass.states.async_set(second.entity_id, "on", {"friendly_name": "Two"})
    async_expose_entity(hass, conversation.DOMAIN, second.entity_id, True)
    await hass.async_block_till_done()

    assert len(catalog.entities) == 2
    assert catalog.generation > generation
    catalog.async_stop()


async def test_attributes_are_pruned_per_domain(hass: HomeAssistant):
    await _setup(hass)
    registry = er.async_get(hass)
    light = registry.async_get_or_create("light", "demo", "lamp")
    hass.states.async_set(
        light.entity_id,
        "on",
        {
            "friendly_name": "Lamp",
            "brightness": 128,
            "volume_level": 0.5,  # belongs to media_player, must be dropped
            "supported_features": 63,
        },
    )
    async_expose_entity(hass, conversation.DOMAIN, light.entity_id, True)

    catalog = EntityCatalog(hass, conversation.DOMAIN)
    attrs = catalog.snapshot()["entities"][0]["attrs"]
    assert attrs == {"brightness": 128}


# --- the weather forecast an LLM answer gets ----------------------------------


def _weather(entity_id: str = "weather.home"):
    from custom_components.typesafe_conversation.entities import CatalogEntity

    return CatalogEntity(
        entity_id=entity_id,
        name="Forecast Home",
        aliases=(),
        area_id=None,
        area_name=None,
        floor_name=None,
        domain="weather",
        device_class=None,
        supported_features=1,
    )


def _serve_forecasts(hass: HomeAssistant, forecast=None, error=None) -> list:
    from homeassistant.core import ServiceCall, SupportsResponse

    calls: list[ServiceCall] = []

    async def get_forecasts(call: ServiceCall):
        calls.append(call)
        if error is not None:
            raise error
        return {"weather.home": {"forecast": forecast or []}}

    hass.services.async_register(
        "weather",
        "get_forecasts",
        get_forecasts,
        supports_response=SupportsResponse.ONLY,
    )
    return calls


def _set_weather(hass: HomeAssistant, features: int) -> None:
    hass.states.async_set(
        "weather.home",
        "partlycloudy",
        {
            "friendly_name": "Forecast Home",
            "supported_features": features,
            "temperature_unit": "°F",
            "precipitation_unit": "in",
        },
    )


async def test_the_forecast_is_summarised_day_by_day(hass: HomeAssistant):
    from custom_components.typesafe_conversation.entities import (
        async_forecast_summary,
    )

    await hass.config.async_set_time_zone("UTC")
    _set_weather(hass, features=1 | 2)
    calls = _serve_forecasts(
        hass,
        [
            {
                "datetime": "2026-10-17T16:00:00+00:00",
                "condition": "partlycloudy",
                "temperature": 76,
                "templow": 67,
                "precipitation": 0.0,
            },
            {
                "datetime": "2026-10-18T16:00:00+00:00",
                "condition": "rainy",
                "temperature": 70,
                "templow": 60,
                "precipitation_probability": 80,
            },
        ],
    )

    summary = await async_forecast_summary(hass, [_weather()])

    assert calls[0].data["type"] == "daily", "daily is preferred over hourly"
    assert summary == (
        "Forecast Home: Sat Oct 17 partly cloudy, 76/67 °F, 0.0 in precipitation; "
        "Sun Oct 18 rainy, 70/60 °F, 80% chance of precipitation"
    )


async def test_no_forecast_feature_means_no_forecast(hass: HomeAssistant):
    from custom_components.typesafe_conversation.entities import (
        async_forecast_summary,
    )

    _set_weather(hass, features=0)
    calls = _serve_forecasts(hass)

    assert await async_forecast_summary(hass, [_weather()]) == ""
    assert not calls


async def test_a_failing_forecast_never_costs_the_answer(hass: HomeAssistant):
    from homeassistant.exceptions import HomeAssistantError

    from custom_components.typesafe_conversation.entities import (
        async_forecast_summary,
    )

    _set_weather(hass, features=1)
    _serve_forecasts(hass, error=HomeAssistantError("provider down"))

    assert await async_forecast_summary(hass, [_weather()]) == ""


async def test_a_home_without_weather_asks_for_nothing(hass: HomeAssistant):
    from custom_components.typesafe_conversation.entities import (
        async_forecast_summary,
    )

    assert await async_forecast_summary(hass, []) == ""


async def test_jev_is_told_the_requests_language(hass: HomeAssistant):
    from custom_components.typesafe_conversation.agent import build_request

    await _setup(hass)
    catalog = EntityCatalog(hass, conversation.DOMAIN)

    asked, _ = build_request(
        hass,
        catalog,
        "éteins la lampe",
        speaker_area_id=None,
        max_options=26,
        language="fr",
    )
    default, _ = build_request(
        hass, catalog, "turn off the lamp", speaker_area_id=None, max_options=26
    )

    assert asked["request"]["language"] == "fr"
    assert default["request"]["language"] == hass.config.language
