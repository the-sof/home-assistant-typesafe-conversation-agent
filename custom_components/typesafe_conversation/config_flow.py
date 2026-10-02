"""Config and subentry flows for TypeSafe Conversation."""

from __future__ import annotations

import asyncio
from typing import Any, override

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .agent import build_request
from .const import (
    ANSWER_TIMEOUT,
    BACKEND_OLLAMA,
    BACKEND_OPENAI_COMPAT,
    CONF_ALWAYS_CONFIRM_RISKY,
    CONF_API_KEY,
    CONF_API_TIMEOUT,
    CONF_BASE_URL,
    CONF_BYPASS_LOCAL_INTENTS,
    CONF_INLINE_ENTITY_DESCRIPTIONS,
    CONF_LLM_API_KEY,
    CONF_LLM_BACKEND,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_LOADED,
    CONF_LLM_MODEL,
    CONF_LLM_TIMEOUT,
    CONF_MODEL,
    CONF_SERVER_PROFILE,
    CONVERSATION_DOMAIN,
    DEFAULT_ALWAYS_CONFIRM_RISKY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DOMAIN,
    LOGGER,
    MEASURE_UTTERANCE,
    TYPESAFE_CONSOLE_URL,
)
from .entities import EntityCatalog
from .system_one import (
    ServerProfile,
    SystemOneAuthError,
    SystemOneClient,
    SystemOneError,
    SystemOneRequestError,
    normalise_base_url,
    normalise_model,
)

_PASSWORD = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))


def _endpoint_schema(defaults: dict[str, Any], *, timeout: bool = False) -> vol.Schema:
    fields: dict[Any, Any] = {
        vol.Required(
            CONF_BASE_URL, default=defaults.get(CONF_BASE_URL, DEFAULT_BASE_URL)
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
        vol.Optional(
            CONF_API_KEY,
            description={"suggested_value": defaults.get(CONF_API_KEY)},
        ): _PASSWORD,
    }
    if timeout:
        fields[vol.Required(CONF_API_TIMEOUT, default=defaults[CONF_API_TIMEOUT])] = (
            NumberSelector(
                NumberSelectorConfig(min=2, max=600, step=1, unit_of_measurement="s")
            )
        )
    return vol.Schema(fields)


_LLM_KEYS = (
    CONF_LLM_BACKEND,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_TIMEOUT,
    CONF_LLM_KEEP_LOADED,
)


def _llm_schema(current: dict[str, Any]) -> vol.Schema:
    """The optional LLM, prefilled with what is set when reconfiguring."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_LLM_BACKEND, default=current.get(CONF_LLM_BACKEND, BACKEND_OLLAMA)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=BACKEND_OLLAMA, label="Ollama"),
                        SelectOptionDict(
                            value=BACKEND_OPENAI_COMPAT,
                            label="OpenAI-compatible (OpenRouter, vLLM, ...)",
                        ),
                    ]
                )
            ),
            vol.Optional(
                CONF_LLM_BASE_URL,
                default=current.get(CONF_LLM_BASE_URL, DEFAULT_OLLAMA_URL),
            ): TextSelector(),
            vol.Optional(
                CONF_LLM_MODEL,
                description={"suggested_value": current.get(CONF_LLM_MODEL)},
            ): TextSelector(),
            vol.Optional(
                CONF_LLM_API_KEY,
                description={"suggested_value": current.get(CONF_LLM_API_KEY)},
            ): _PASSWORD,
            vol.Optional(
                CONF_LLM_TIMEOUT,
                default=current.get(CONF_LLM_TIMEOUT, ANSWER_TIMEOUT),
            ): NumberSelector(
                NumberSelectorConfig(min=5, max=180, step=5, unit_of_measurement="s")
            ),
            vol.Optional(
                CONF_LLM_KEEP_LOADED,
                default=current.get(CONF_LLM_KEEP_LOADED, False),
            ): BooleanSelector(),
        }
    )


class TypeSafeConfigFlow(ConfigFlow, domain=DOMAIN):
    """Endpoint, then model, then a probe of what they can take, then the LLM.

    Nothing about a server is hardcoded. The System One API does not report its
    limits, so the probe measures them - the option cap, the cold-load time,
    how long a real-sized request takes - and they are cached on the entry
    until the endpoint or model changes.
    """

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._models: list[str] = []
        self._probe_task: asyncio.Task[ServerProfile] | None = None
        self._probe_error: str | None = None
        self._reconfiguring = False

    def _client(self, model: str | None = None) -> SystemOneClient:
        return SystemOneClient(
            async_get_clientsession(self.hass),
            self._data.get(CONF_API_KEY),
            model or self._data.get(CONF_MODEL, DEFAULT_MODEL),
            base_url=self._data.get(CONF_BASE_URL),
        )

    async def _validate_endpoint(self, user_input: dict[str, Any]) -> str | None:
        """Check the endpoint and key, and load the models it offers."""
        self._data.update(user_input)
        self._data[CONF_BASE_URL] = normalise_base_url(user_input.get(CONF_BASE_URL))
        if not user_input.get(CONF_API_KEY):
            self._data.pop(CONF_API_KEY, None)
        client = self._client()
        try:
            names = await client.async_validate()
            self._models = await client.async_decision_models(names)
        except SystemOneAuthError:
            return "invalid_auth"
        except SystemOneError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected error checking the System One endpoint")
            return "unknown"
        return None

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            if (error := await self._validate_endpoint(user_input)) is None:
                return await self.async_step_model()
            errors["base"] = error

        return self.async_show_form(
            step_id="user",
            data_schema=_endpoint_schema(self._data),
            errors=errors,
            # hassfest rejects a literal URL inside a translated string, so the
            # console link is supplied here instead.
            description_placeholders={"console_url": TYPESAFE_CONSOLE_URL},
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Choose the decision model, from what the server says it has."""
        errors: dict[str, str] = {}
        placeholders = {"reason": ""}
        if user_input is not None:
            self._data[CONF_MODEL] = normalise_model(user_input[CONF_MODEL])
            self._probe_error = None
            cached = ServerProfile.from_dict(self._data.get(CONF_SERVER_PROFILE))
            if self._reconfiguring and cached.matches(
                self._data[CONF_BASE_URL], self._data[CONF_MODEL]
            ):
                # Same server, same model: what the probe measured still holds.
                return await self.async_step_llm()
            return await self.async_step_probe()
        if self._probe_error is not None:
            errors["base"] = "model_rejected"
            placeholders["reason"] = self._probe_error

        default = self._data.get(CONF_MODEL) or (
            DEFAULT_MODEL
            if DEFAULT_MODEL in self._models or not self._models
            else self._models[0]
        )
        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL, default=default): SelectSelector(
                        SelectSelectorConfig(
                            options=self._models or [default], custom_value=True
                        )
                    )
                }
            ),
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_probe(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Measure the server and model, with a progress indicator.

        A model that is not loaded yet has to be loaded first, which can take
        most of a minute on modest hardware.
        """
        if self._probe_task is None:
            # Not eager: the task must not finish inside this call. If it did,
            # the step would return progress-done straight away, and the flow
            # manager re-runs the next step with this same input - which would
            # submit the LLM form unseen, or re-probe a rejected model forever.
            self._probe_task = self.hass.async_create_task(
                self._async_test_server(), "typesafe_probe", eager_start=False
            )
        if not self._probe_task.done():
            return self.async_show_progress(
                step_id="probe",
                progress_action="probe",
                progress_task=self._probe_task,
                description_placeholders={"model": self._data[CONF_MODEL]},
            )

        task, self._probe_task = self._probe_task, None
        try:
            profile = task.result()
        except SystemOneAuthError:
            self._probe_error = "The API key was rejected."
        except SystemOneRequestError as err:
            self._probe_error = str(err)
        except SystemOneError as err:
            self._probe_error = f"The server did not answer: {err}"
        except Exception:
            LOGGER.exception("Unexpected error probing the System One model")
            self._probe_error = "Unexpected error."
        else:
            self._data[CONF_SERVER_PROFILE] = profile.as_dict()
            return self.async_show_progress_done(next_step_id="tested")
        return self.async_show_progress_done(next_step_id="model")

    async def _async_test_server(self) -> ServerProfile:
        """Probe the limits, then time one request shaped like a real one.

        The limits come from tiny probes, which say nothing about speed. So the
        timing uses this home's actual catalogue and the full question set, at
        the cap just found: what a voice command will really cost here.
        """
        client = self._client()
        profile = await client.async_probe()
        catalog = self.hass.data.get(DOMAIN, {}).get("catalog") or EntityCatalog(
            self.hass, CONVERSATION_DOMAIN
        )
        state, questions = build_request(
            self.hass,
            catalog,
            MEASURE_UTTERANCE,
            speaker_area_id=None,
            max_options=profile.max_options,
        )
        return profile.with_typical(await client.async_time_request(state, questions))

    async def async_step_tested(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Say how fast the server is before the user relies on it.

        A slow server is far better discovered here than as silence from a
        voice satellite.
        """
        if user_input is not None:
            return await self.async_step_llm()
        profile = ServerProfile.from_dict(self._data[CONF_SERVER_PROFILE])
        return self.async_show_form(
            step_id="tested",
            data_schema=vol.Schema({}),
            description_placeholders={
                "model": self._data[CONF_MODEL],
                "typical": f"{profile.typical_s or 0:.1f}",
                "cold": f"{profile.cold_load_s or 0:.1f}",
                "timeout": f"{profile.timeout:.0f}",
            },
        )

    async def async_step_llm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the LLM used for compound splits and prose answers.

        Optional: without it the agent still handles every command and query,
        it just cannot split compound requests or answer general questions.
        """
        if user_input is not None:
            # Replace rather than merge, so clearing a field (or switching
            # keep-loaded off) on reconfigure actually takes effect.
            for key in _LLM_KEYS:
                self._data.pop(key, None)
            self._data.update(
                {k: v for k, v in user_input.items() if v not in (None, "")}
            )
            if self._reconfiguring:
                return await self.async_step_finish()
            return self.async_create_entry(
                title="TypeSafe Conversation",
                data=self._data,
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "title": "TypeSafe Conversation",
                        "data": {},
                        "unique_id": None,
                    }
                ],
            )
        return self.async_show_form(step_id="llm", data_schema=_llm_schema(self._data))

    @override
    async def async_step_reauth(self, entry_data: dict[str, Any]) -> ConfigFlowResult:
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        entry = self._get_reauth_entry()
        if user_input is not None:
            client = SystemOneClient(
                async_get_clientsession(self.hass),
                user_input[CONF_API_KEY],
                entry.data.get(CONF_MODEL, DEFAULT_MODEL),
                base_url=entry.data.get(CONF_BASE_URL),
            )
            try:
                await client.async_validate()
            except SystemOneAuthError:
                errors["base"] = "invalid_auth"
            except SystemOneError:
                errors["base"] = "cannot_connect"
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_API_KEY: user_input[CONF_API_KEY]}
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            ),
            errors=errors,
        )

    @override
    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the endpoint, key, model or timeout of an existing entry.

        A new endpoint or model is probed again; a key or timeout change alone
        keeps the cached profile.
        """
        entry = self._get_reconfigure_entry()
        self._reconfiguring = True
        profile = ServerProfile.from_dict(entry.data.get(CONF_SERVER_PROFILE))
        defaults = {
            **entry.data,
            CONF_BASE_URL: normalise_base_url(entry.data.get(CONF_BASE_URL)),
            CONF_API_TIMEOUT: entry.data.get(CONF_API_TIMEOUT, profile.timeout),
        }
        errors: dict[str, str] = {}
        if user_input is not None:
            self._data = {
                k: v
                for k, v in entry.data.items()
                if k not in (CONF_API_KEY, CONF_BASE_URL, CONF_API_TIMEOUT)
            }
            if float(user_input.get(CONF_API_TIMEOUT, 0)) == float(profile.timeout):
                # Left at the measured value: don't pin it, so a re-probe for a
                # new server or model can set a fresh one.
                user_input = {
                    k: v for k, v in user_input.items() if k != CONF_API_TIMEOUT
                }
            if (error := await self._validate_endpoint(user_input)) is None:
                return await self.async_step_model()
            errors["base"] = error
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_endpoint_schema(defaults, timeout=True),
            errors=errors,
            description_placeholders={"console_url": TYPESAFE_CONSOLE_URL},
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Save a reconfiguration and reload the entry."""
        return self.async_update_reload_and_abort(
            self._get_reconfigure_entry(), data=self._data
        )

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {"conversation": TypeSafeSubentryFlowHandler}


class TypeSafeSubentryFlowHandler(ConfigSubentryFlow):
    """One conversation agent, with its own tuning."""

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        return await self.async_step_set_options(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        return await self.async_step_set_options(user_input)

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            title = user_input.pop("name", "TypeSafe Conversation")
            if self._is_new:
                return self.async_create_entry(title=title, data=user_input)
            return self.async_update_and_abort(
                self._get_entry(), self._get_reconfigure_subentry(), data=user_input
            )

        current = {} if self._is_new else dict(self._get_reconfigure_subentry().data)
        schema = vol.Schema(
            {
                vol.Required(
                    "name",
                    default=(
                        "TypeSafe Conversation"
                        if self._is_new
                        else self._get_reconfigure_subentry().title
                    ),
                ): TextSelector(),
                vol.Optional(
                    CONF_ALWAYS_CONFIRM_RISKY,
                    default=current.get(
                        CONF_ALWAYS_CONFIRM_RISKY, DEFAULT_ALWAYS_CONFIRM_RISKY
                    ),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_INLINE_ENTITY_DESCRIPTIONS,
                    default=current.get(CONF_INLINE_ENTITY_DESCRIPTIONS, False),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_BYPASS_LOCAL_INTENTS,
                    default=current.get(CONF_BYPASS_LOCAL_INTENTS, False),
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="set_options", data_schema=schema)
