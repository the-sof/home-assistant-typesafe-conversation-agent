"""Read-only lookups the language model may use when it answers a question.

Optional, and off by default. When a user turns web search on, the answer path
offers the model tools from one source:

* SearXNG, called directly - a self-hosted search with nothing else to set up.
* Any LLM API registered in Home Assistant except Assist. That is how an MCP
  server added through the Model Context Protocol integration appears, so any
  search provider with an MCP server works without code here.

Assist is never offered: device control stays on Jev's typed path, and the
answer path still cannot turn anything on or off. A tool that fails returns its
error as text, so the model can answer without it rather than the request
failing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from voluptuous_openapi import convert

from .const import (
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_API,
    CONF_WEB_SEARCH_URL,
    DOMAIN,
    LOGGER,
    SEARCH_RESULTS,
    TOOL_RESULT_MAX_CHARS,
    TOOL_TIMEOUT,
    WEB_SEARCH_LLM_API,
    WEB_SEARCH_SEARXNG,
)


@dataclass(frozen=True, slots=True)
class AnswerTool:
    """One function the model can call, whatever provides it."""

    name: str
    description: str
    parameters: dict[str, Any]
    call: Callable[[dict[str, Any]], Awaitable[str]]

    def spec(self) -> dict[str, Any]:
        """The function definition both Ollama and OpenAI-style servers take."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, args: dict[str, Any]) -> str:
        """Call the tool. Never raises: a failure comes back as text."""
        try:
            async with asyncio.timeout(TOOL_TIMEOUT):
                result = await self.call(args)
        except TimeoutError:
            result = f"{self.name} failed: no answer within {TOOL_TIMEOUT:.0f}s"
        except Exception as err:  # any failure is the model's to handle
            LOGGER.debug("Tool %s failed", self.name, exc_info=True)
            result = f"{self.name} failed: {err}"
        return _trim(result)


def _trim(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    return text[: TOOL_RESULT_MAX_CHARS - 3] + "..."


# --- SearXNG ------------------------------------------------------------------


class SearchError(Exception):
    """SearXNG could not be queried."""


def normalise_searxng_url(url: str) -> str:
    """The instance's base address, from whatever was pasted."""
    url = url.strip().rstrip("/")
    return url.removesuffix("/search").rstrip("/")


async def async_searxng_search(
    session: aiohttp.ClientSession, base_url: str, query: str
) -> list[dict[str, str]]:
    """The top results for a query: title, snippet and address.

    Raises SearchError with ``"json_disabled"`` when the instance refuses JSON,
    which SearXNG does until ``json`` is added to ``search.formats``.
    """
    try:
        async with session.get(
            f"{base_url}/search",
            params={"q": query, "format": "json"},
            timeout=aiohttp.ClientTimeout(total=TOOL_TIMEOUT),
            allow_redirects=False,
        ) as response:
            if response.status == 403:
                raise SearchError("json_disabled")
            if response.status >= 300:
                raise SearchError(f"HTTP {response.status}")
            payload = await response.json(content_type=None)
    except SearchError:
        raise
    except (aiohttp.ClientError, TimeoutError, ValueError) as err:
        raise SearchError(type(err).__name__) from err
    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise SearchError("unexpected response")
    return [
        {
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("content") or ""),
            "url": str(item.get("url") or ""),
        }
        for item in items[:SEARCH_RESULTS]
        if isinstance(item, dict)
    ]


def searxng_tool(session: aiohttp.ClientSession, base_url: str) -> AnswerTool:
    """A ``web_search`` tool backed by a SearXNG instance."""

    async def call(args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "web_search failed: no query given"
        try:
            results = await async_searxng_search(session, base_url, query)
        except SearchError as err:
            return f"web_search failed: {err}"
        if not results:
            return "No results."
        return "\n".join(f"{r['title']} - {r['snippet']} ({r['url']})" for r in results)

    return AnswerTool(
        name="web_search",
        description=(
            "Search the web for current information: news, events, schedules, "
            "scores, prices, opening hours, or anything that may have changed "
            "recently. Returns the top results as title, snippet and address."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."}
            },
            "required": ["query"],
        },
        call=call,
    )


# --- Home Assistant LLM APIs ---------------------------------------------------


def lookup_apis(hass: HomeAssistant) -> dict[str, str]:
    """Every registered LLM API that may be offered, by id, with its name.

    Assist is left out: it controls devices, and this path must not.
    """
    return {
        api.id: api.name
        for api in llm.async_get_apis(hass)
        if api.id != llm.LLM_API_ASSIST
    }


async def async_api_tools(
    hass: HomeAssistant,
    api_id: str,
    user_input: conversation.ConversationInput,
) -> list[AnswerTool]:
    """The tools of one Home Assistant LLM API, ready for the model."""
    if api_id == llm.LLM_API_ASSIST:
        return []
    instance = await llm.async_get_api(
        hass,
        api_id,
        llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        ),
    )
    serializer = instance.custom_serializer or llm.selector_serializer

    def wrap(tool: llm.Tool) -> AnswerTool:
        async def call(args: dict[str, Any]) -> str:
            try:
                result = await instance.async_call_tool(
                    llm.ToolInput(tool_name=tool.name, tool_args=args)
                )
            except (HomeAssistantError, vol.Invalid) as err:
                return f"{tool.name} failed: {err}"
            return json.dumps(result, ensure_ascii=False, default=str)

        return AnswerTool(
            name=tool.name,
            description=tool.description or "",
            parameters=convert(tool.parameters, custom_serializer=serializer),
            call=call,
        )

    return [wrap(tool) for tool in instance.tools]


async def async_answer_tools(
    hass: HomeAssistant,
    settings: Mapping[str, Any],
    user_input: conversation.ConversationInput,
) -> list[AnswerTool]:
    """The tools the configured web search provides, or none.

    A source that has gone away - the MCP integration removed, say - logs a
    warning and the answer goes ahead without tools.
    """
    source = settings.get(CONF_WEB_SEARCH)
    if source == WEB_SEARCH_SEARXNG and (url := settings.get(CONF_WEB_SEARCH_URL)):
        return [searxng_tool(async_get_clientsession(hass), url)]
    if source == WEB_SEARCH_LLM_API and (api_id := settings.get(CONF_WEB_SEARCH_API)):
        try:
            return await async_api_tools(hass, api_id, user_input)
        except HomeAssistantError as err:
            LOGGER.warning("Web search is unavailable (%s); answering without it", err)
    return []


__all__ = [
    "AnswerTool",
    "SearchError",
    "async_answer_tools",
    "async_searxng_search",
    "lookup_apis",
    "normalise_searxng_url",
    "searxng_tool",
]
