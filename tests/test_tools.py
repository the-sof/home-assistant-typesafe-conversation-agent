"""Where the answer's lookup tools come from: SearXNG, or a Home Assistant LLM API."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.typesafe_conversation import tools as tools_module
from custom_components.typesafe_conversation.agent import AgentSettings, TypeSafeAgent
from custom_components.typesafe_conversation.const import (
    SEARCH_RESULTS,
    TOOL_RESULT_MAX_CHARS,
)
from custom_components.typesafe_conversation.tools import (
    AnswerTool,
    async_answer_tools,
    lookup_apis,
    normalise_searxng_url,
    searxng_tool,
)

SEARX = "http://searx.invalid:8888"
SEARCH_URL = f"{SEARX}/search"


def _user_input():
    # Only these three are read, and their constructor differs across versions.
    return SimpleNamespace(context=Context(), language="en", device_id=None)


def _results(count: int) -> dict:
    return {
        "results": [
            {
                "title": f"Result {i}",
                "content": f"Snippet {i}",
                "url": f"https://example.invalid/{i}",
            }
            for i in range(count)
        ]
    }


# --- SearXNG -----------------------------------------------------------------


async def test_searxng_returns_the_top_results_as_lines(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    aioclient_mock.get(SEARCH_URL, json=_results(SEARCH_RESULTS + 3))
    tool = searxng_tool(async_get_clientsession(hass), SEARX)

    text = await tool.run({"query": "rovers kickoff"})

    lines = text.splitlines()
    assert len(lines) == SEARCH_RESULTS
    assert lines[0] == "Result 0 - Snippet 0 (https://example.invalid/0)"
    _method, url, _data, _headers = aioclient_mock.mock_calls[0]
    assert url.query == {"q": "rovers kickoff", "format": "json"}


async def test_searxng_without_json_output_says_so(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    """SearXNG refuses JSON until settings.yml lists it."""
    aioclient_mock.get(SEARCH_URL, status=403)
    tool = searxng_tool(async_get_clientsession(hass), SEARX)

    assert await tool.run({"query": "x"}) == "web_search failed: json_disabled"


async def test_a_search_without_a_query_is_not_sent(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
):
    tool = searxng_tool(async_get_clientsession(hass), SEARX)

    assert await tool.run({}) == "web_search failed: no query given"
    assert aioclient_mock.mock_calls == []


@pytest.mark.parametrize(
    "pasted",
    [SEARX, f"{SEARX}/", f"{SEARX}/search", f" {SEARX}/search/ "],
)
def test_a_pasted_searxng_address_is_trimmed(pasted):
    assert normalise_searxng_url(pasted) == SEARX


# --- every tool ----------------------------------------------------------------


async def test_a_long_result_is_trimmed():
    async def call(_args):
        return "x" * (TOOL_RESULT_MAX_CHARS * 2)

    tool = AnswerTool("t", "", {}, call)

    assert len(await tool.run({})) == TOOL_RESULT_MAX_CHARS


async def test_a_slow_tool_gives_up_instead_of_holding_the_answer():
    async def call(_args):
        await asyncio.sleep(10)
        return "late"

    tool = AnswerTool("slow", "", {}, call)
    with patch.object(tools_module, "TOOL_TIMEOUT", 0.01):
        assert (await tool.run({})).startswith("slow failed: no answer within")


# --- Home Assistant LLM APIs ------------------------------------------------------


class _Lookup(llm.Tool):
    name = "lookup"
    description = "Look something up."
    parameters = vol.Schema({vol.Required("query"): str})

    async def async_call(self, hass, tool_input, llm_context):
        return {"answer": tool_input.tool_args["query"].upper()}


class _LookupAPI(llm.API):
    async def async_get_api_instance(self, llm_context):
        return llm.APIInstance(
            api=self, api_prompt="", llm_context=llm_context, tools=[_Lookup()]
        )


def _register(hass: HomeAssistant) -> None:
    llm.async_register_api(hass, _LookupAPI(hass=hass, id="lookup", name="Lookup"))


async def test_assist_is_never_offered(hass: HomeAssistant):
    """Assist controls devices; the answer path must not."""
    _register(hass)

    apis = lookup_apis(hass)

    assert apis == {"lookup": "Lookup"}
    assert llm.LLM_API_ASSIST in {api.id for api in llm.async_get_apis(hass)}
    settings = {"web_search": "llm_api", "web_search_api": llm.LLM_API_ASSIST}
    assert await async_answer_tools(hass, settings, _user_input()) == []


async def test_an_llm_api_tool_is_offered_and_called(hass: HomeAssistant):
    _register(hass)
    settings = {"web_search": "llm_api", "web_search_api": "lookup"}

    [tool] = await async_answer_tools(hass, settings, _user_input())

    assert tool.spec()["function"]["name"] == "lookup"
    assert tool.parameters["properties"]["query"]["type"] == "string"
    assert json.loads(await tool.run({"query": "kickoff"})) == {"answer": "KICKOFF"}


async def test_a_bad_argument_reaches_the_model_as_text(hass: HomeAssistant):
    _register(hass)
    settings = {"web_search": "llm_api", "web_search_api": "lookup"}
    [tool] = await async_answer_tools(hass, settings, _user_input())

    assert (await tool.run({})).startswith("lookup failed:")


async def test_an_api_that_has_gone_means_no_tools(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
):
    settings = {"web_search": "llm_api", "web_search_api": "removed"}

    assert await async_answer_tools(hass, settings, _user_input()) == []
    assert "Web search is unavailable" in caplog.text


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"web_search": "off"},
        {"web_search": "off", "web_search_url": SEARX},
        {"web_search": "searxng"},
    ],
)
async def test_no_tools_unless_web_search_is_set_up(hass: HomeAssistant, settings):
    assert await async_answer_tools(hass, settings, _user_input()) == []


# --- the agent ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("web_search", "expected"),
    [
        ({"web_search": "searxng", "web_search_url": SEARX}, ["web_search"]),
        ({"web_search": "off"}, []),
        (None, []),
    ],
)
async def test_the_answer_gets_the_configured_tools(
    hass: HomeAssistant, web_search, expected
):
    llm_backend = MagicMock()
    llm_backend.answer_freeform = AsyncMock(return_value="Kickoff is at 7:30.")
    agent = TypeSafeAgent(
        hass,
        MagicMock(),
        MagicMock(),
        llm_backend,
        AgentSettings(web_search=web_search),
    )

    response = await agent._answer_freeform(
        SimpleNamespace(
            text="when do the Rovers play?",
            context=Context(),
            language="en",
            device_id=None,
            satellite_id=None,
        ),
        MagicMock(content=[]),
    )

    tools = llm_backend.answer_freeform.call_args.kwargs["tools"]
    assert [tool.name for tool in tools] == expected
    assert response.speech["plain"]["speech"] == "Kickoff is at 7:30."
