"""The conversation platform for the TypeSafe Conversation integration."""

from __future__ import annotations

from typing import Literal, override

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import TypeSafeConfigEntry
from .agent import AgentSettings, TypeSafeAgent
from .const import (
    CONF_ALWAYS_CONFIRM_RISKY,
    CONF_BYPASS_LOCAL_INTENTS,
    CONF_INLINE_ENTITY_DESCRIPTIONS,
    DEFAULT_ALWAYS_CONFIRM_RISKY,
    DOMAIN,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: TypeSafeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up one conversation entity per conversation subentry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        async_add_entities(
            [TypeSafeConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class TypeSafeConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """A conversation agent that decides with Jev and acts with intents."""

    _attr_has_entity_name = True
    _attr_name = None
    # Jev returns a decision, not a stream of tokens; there is nothing to
    # stream and claiming otherwise would only add latency.
    _attr_supports_streaming = False

    def __init__(self, entry: TypeSafeConfigEntry, subentry: ConfigSubentry) -> None:
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = {
            "identifiers": {(DOMAIN, subentry.subentry_id)},
            "name": subentry.title,
            "manufacturer": "TypeSafe",
            "model": entry.runtime_data.model,
            "entry_type": "service",
        }
        settings = {**entry.data, **subentry.data}
        # Advertising CONTROL is what makes Home Assistant hand us the
        # utterances worth spending a Jev call on. With "prefer handling
        # commands locally" on, the sentence matcher keeps every command it
        # recognizes and only HassGetState and HassMediaSearchAndPlay are
        # forced through to us - along with everything it could not parse,
        # which is exactly where Jev earns its keep.
        if not settings.get(CONF_BYPASS_LOCAL_INTENTS):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    @override
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    @override
    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    @override
    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        data = self.entry.runtime_data
        settings = {**self.entry.data, **self.subentry.data}
        agent = TypeSafeAgent(
            self.hass,
            data.catalog,
            data.client,
            data.llm,
            AgentSettings(
                inline_entity_descriptions=bool(
                    settings.get(CONF_INLINE_ENTITY_DESCRIPTIONS)
                ),
                always_confirm_risky=bool(
                    settings.get(
                        CONF_ALWAYS_CONFIRM_RISKY, DEFAULT_ALWAYS_CONFIRM_RISKY
                    )
                ),
                bypass_local_intents=bool(settings.get(CONF_BYPASS_LOCAL_INTENTS)),
            ),
            traces=data.traces,
            pending_fills=data.pending_fills,
        )
        # Reuse the cached question set across turns of this config entry.
        agent._questions_cache = data.questions_cache
        response = await agent.async_process(user_input, chat_log)
        data.questions_cache = agent._questions_cache

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=user_input.agent_id,
                content=response.speech.get("plain", {}).get("speech", ""),
            )
        )
        return conversation.ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=agent.continue_conversation,
        )
