<p align="center"><img src="https://raw.githubusercontent.com/the-sof/home-assistant-typesafe-conversation-agent/main/custom_components/typesafe_conversation/brand/icon.png" alt="TypeSafe Conversation" width="128"></p>

# TypeSafe Conversation for Home Assistant

[![tests](https://img.shields.io/github/actions/workflow/status/the-sof/home-assistant-typesafe-conversation-agent/test.yml?branch=main&label=tests)](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml)
[![license](https://img.shields.io/github/license/the-sof/home-assistant-typesafe-conversation-agent)](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/blob/main/LICENSE)
[![HACS: default](https://img.shields.io/badge/HACS-default-41BDF5)](https://hacs.xyz/)

A Home Assistant conversation agent that decides with a
[System One](https://docs.typesafe.ai) decision model instead of an LLM:
**TypeSafe's hosted model, or one running on your own hardware through
Ollama**, so nothing has to leave your home. Device commands and questions
about the home are resolved from its typed answers in code; a language model,
if you add one, only writes text.

Found a problem? [Report it](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/issues/new?template=bug_report.yml).

## Why use it

- **Local or hosted, your choice:** run a decision model such as `nimble` or
  `tev1` on your own machine with Ollama, free and fully local, or use
  TypeSafe's hosted API. See [Running locally](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#running-locally) and
  [What gets sent](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#what-gets-sent).
- **Fast:** a command or a question about the home takes about 250 ms on
  TypeSafe's API, where an LLM turn takes 2–8 s.
- **Predictable:** every decision is a typed, calibrated answer checked in
  ordinary Python, and a doubtful one asks or falls back instead of guessing.
- **The LLM is optional:** it only writes text: splitting a request with
  several commands, answering a general question (with web search, if you
  turn it on), and filling in a script's fields.
- **Your language:** replies follow the language of the Assist pipeline.

## What you need

- Home Assistant 2026.5.0 or newer.
- A System One server, one of:
  - **local:** [Ollama](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#running-locally) 0.35 or newer with a decision
    model such as `nimble`; no key, no cost, nothing leaves your network;
  - **hosted:** a [TypeSafe](https://console.typesafe.ai/) API key, which is
    metered (see [Cost](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#cost)).
- Optionally, a language model: Ollama or any OpenAI-compatible endpoint
  (OpenRouter, vLLM, …). It can be local too.

## Install

**Requires Home Assistant 2026.5.0 or newer.** Earlier releases do not report a
failed service call back to the conversation agent, so a command that no entity
could carry out would be announced as if it had worked. The integration is
tested against 2026.5.0 and the current release on every change.

### Through HACS

In HACS, search for **TypeSafe Conversation** and download it, or open it
directly:

[![Open in HACS](https://img.shields.io/badge/Open%20in-HACS-41BDF5?style=for-the-badge&logo=homeassistantcommunitystore&logoColor=white)](https://my.home-assistant.io/redirect/hacs_repository/?owner=the-sof&repository=home-assistant-typesafe-conversation-agent&category=integration)

HACS then tells you when a new release is out.

### If it doesn't show up in HACS

Add it as a custom repository: in HACS, **⋮ → Custom repositories**, paste
`https://github.com/the-sof/home-assistant-typesafe-conversation-agent`, choose
category **Integration**, then **Add**. It then appears in HACS for install and
for update notifications.

### By hand

Copy `custom_components/typesafe_conversation` into your Home Assistant `config`
directory.

## Set up

Restart Home Assistant, then **Settings → Devices & Services → Add Integration →
TypeSafe Conversation**. For a local server, enter its address, for example
`http://<ollama-host>:11434`, and leave the API key blank; for TypeSafe's
hosted API, keep the default address and paste a key from
[console.typesafe.ai](https://console.typesafe.ai/). Finally, set it as the
conversation agent under **Settings → Voice assistants**.

The LLM step is optional. Without it the agent still handles every command and
query; it just cannot split compound requests or answer general questions.
Ollama and any OpenAI-compatible endpoint (OpenRouter, vLLM, …) are supported.

## Options

Set when you add the integration, and changeable afterwards under
**Settings → Devices & Services → TypeSafe Conversation**: the server, model,
timeout and language-model settings under **Reconfigure** in the entry's menu,
and `always_confirm_risky`, `bypass_local_intents` and
`inline_entity_descriptions` under each conversation agent's **Configure**.

| option | default | what it does |
| --- | --- | --- |
| `base_url` | `https://api.typesafe.ai` | The System One server. Any compatible server works; see [Running locally](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#running-locally). |
| `api_key` | — | Required by TypeSafe's hosted API. Leave blank for a server that doesn't check one, such as Ollama. |
| `model` | `jev-latest` | Which decision model to use, chosen from what the server offers. On TypeSafe, `jev-latest` tracks the newest Jev. |
| `api_timeout` | measured | How long to wait for an answer. Set at setup from a timed, real-sized request plus the cold-load time, never below 6 s. |
| `always_confirm_risky` | **on** | Ask before unlocking a door, opening a garage or disarming an alarm, however sure the model is. **Turning this off lets confident requests through silently** — the model's judgement becomes the only gate. |
| `bypass_local_intents` | off | Send every command here, including ones Home Assistant's own sentence matcher recognises. Off is recommended; see [below](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#leave-prefer-handling-commands-locally-on). |
| `inline_entity_descriptions` | off | Describe every entity inside each question rather than once in the shared state. Roughly doubles the tokens. Only worth it if the agent picks the wrong device. |
| `llm_backend` | none | `ollama`, an OpenAI-compatible endpoint, or none. Used for compound requests, general questions, and [scripts with fields](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#scripts-with-fields). |
| `llm_base_url` | `http://localhost:11434` for Ollama, `https://openrouter.ai/api` for OpenAI-compatible | Where that backend lives. For Ollama, the machine running it (`localhost` only works if Home Assistant is not in a container). For OpenRouter, a trailing `/v1` is fine. Checked before it is saved. |
| `llm_model` | — | Model name on that backend, picked from the list the server offers. |
| `llm_api_key` | — | If the backend needs one. Redacted in diagnostics. |
| `llm_timeout` | `30` s | How long to wait for the LLM before giving up. Only the prose path is affected. A large local model that must load first may need more. |
| `llm_keep_loaded` | off | Ollama only. Load the language model at startup and keep it in memory, so the first general question after a pause is fast. Holds the memory while Home Assistant runs; off leaves unloading to the server. |
| `llm_referer`, `llm_title` | project defaults | Sent as `HTTP-Referer` and `X-Title`; OpenRouter uses them for attribution. |
| `web_search` | off | Let the LLM look things up when it answers: `off`, `searxng`, or `llm_api`. See [Web search](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#web-search). |
| `web_search_url` | — | The SearXNG instance. Checked with a real search before it is saved. Redacted in diagnostics. |
| `web_search_api` | — | The Home Assistant LLM API to take tools from, such as an MCP server. Assist is never offered. |

## Web search

Weather needs none of this: an answer is given the coming days' forecast of
any weather entity exposed to Assist, along with today's date.

Without it, a question about anything current — tonight's game, the news, a
shop's hours — gets a guess or "I don't know". Turn it on in the **Web search**
step under **Reconfigure**, after the language model, and pick where lookups
come from:

- **SearXNG** — a self-hosted metasearch engine. Queries reach the engines it
  uses without an account or key, and nothing is logged. Its `settings.yml`
  must allow JSON, which it does not by default:

  ```yaml
  search:
    formats:
      - html
      - json
  ```

  SearXNG's direct answers and infoboxes reach the model ahead of the
  results. For weather, the model searches SearXNG's weather category, which
  the default `openmeteo` engine answers with a forecast; keep it enabled.

- **A Home Assistant LLM API** — anything registered with Home Assistant other
  than Assist. Add a search provider's MCP server with the
  [Model Context Protocol](https://www.home-assistant.io/integrations/mcp/)
  integration and it appears here. Pick one whose tools only look things up:
  every tool it offers is given to the model. Until one is set up, the option
  says so and there is nothing to pick.

The model decides when to search; stable general knowledge is answered without
one. Lookups are only offered when answering a question — commands still go
through Jev and never reach a tool — and a lookup that fails or takes more
than 8 s is reported to the model, which says it could not check. The language
model has to support tool calls (Qwen3, Llama 3.1 and later, GPT-4-class and
DeepSeek V3 models do); one that refuses them is answered without, and not
asked again until the integration reloads.

Some models search on their own, such as Perplexity's Sonar models or any
OpenRouter model with the `online` variant (pick it in the *Variant* field
under the model). Choosing one needs no web search setting at all, at the cost
of a search on every general question. That search uses only the last thing
said, so a follow-up such as "what about this weekend?" searches poorly; with
SearXNG the model writes its own query. The
source links such models add are dropped before the answer is spoken; a link
you ask for is given as a plain address.

## Scripts with fields

Scripts often take fields: an alarm script wants a time and a speaker, a
"notify" script wants a message. The decision model picks the script; the
[language model](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#options) then fills in its fields in one short, structured
call, and every value is checked against the field's selector before the script
runs. Nothing is guessed: a required value the request doesn't give is asked
for ("What time should I use?"), and the next thing you say completes it.
Without a language model configured, a script that takes fields is not run.

Write field descriptions as you would for any LLM agent; they are what the
language model reads. For example:

```yaml
script:
  set_alarm:
    alias: Set an alarm
    description: Set an alarm clock for a time of day.
    fields:
      alarm_time:
        name: Time
        description: The time of day, 24-hour.
        required: true
        selector:
          time:
      location:
        name: Location
        description: Which speaker. If not said, the room the user is in.
        required: true
        selector:
          select:
            options: [upstairs, downstairs]
    sequence:
      - action: input_datetime.set_datetime
        target:
          entity_id: input_datetime.alarm
        data:
          time: "{{ alarm_time }}"
```

"Wake me up at 5.15 tomorrow" from a bedroom satellite runs it with
`alarm_time: 05:15:00` and the bedroom's speaker. A time said without AM/PM is
the next one to come round, so "4:30" said in the afternoon is 4:30 PM. If the
script returns a `speech`, `message`, `result`, `response` or `text` string,
that is what is said; otherwise the reply names the values it used. Most selectors work, including `select`, `number`,
`boolean`, `time`, `date`, `duration`, `area` and `text`. Hidden (advanced)
fields are left out.

A script that finds something by its time, such as cancelling the 4:30 alarm,
works best if it also accepts the time twelve hours away when nothing matches
exactly: the language model can't see which alarms exist, so "4:30" said in the
afternoon reaches the script as 16:30 even when the alarm is at 4:30 AM.

## Languages

A reply is in the language of the request, which in Home Assistant is the
language of the Assist pipeline it came through. That pipeline's voice reads
the reply, so the two always match.

- Confirmations, state answers, the time, the weather and "not understood" use
  Home Assistant's own wording for that language, the same as its built-in
  assistant.
- The agent's own phrases, such as "Did you mean…?", "Which room?" or a summary
  of a compound request, are translated by the language model. Without one
  they stay in English.
- Answers from the language model are in the language they were asked in.

For a home with two languages, give each its own pipeline, with its own speech
recognition and voice, and assign the two to the two wake words of a voice
satellite. Device names are matched as named in any language, so a sentence
that mixes languages, such as "apaga la Stand Light", is understood; the reply
is in the pipeline's language.

## Leave "prefer handling commands locally" on

The agent advertises `ConversationEntityFeature.CONTROL`. With prefer-local on,
Home Assistant's sentence matcher keeps every phrasing it recognises, and only
`HassGetState`, `HassMediaSearchAndPlay` and everything it *failed* to parse
reach this agent. That is deliberate: there is no point spending an API call
beating hassil at "turn off the kitchen lights", and it concentrates the traffic
on the requests where Jev earns its keep. Set `bypass_local_intents` if you want
to compare the two regimes.

## What gets sent

On every request this integration sends, to the System One server you configured
(TypeSafe's hosted `https://api.typesafe.ai` by default):

- the text of the utterance
- **every entity you have exposed to Assist** — its name, area, floor, domain,
  device class and current state
- the areas and floors in your home, by name

That is the whole point of the design: the model is given the home as state and
answers questions about it, rather than being asked to write code or call tools.
With TypeSafe's hosted API, that means your device and room names, and what is
currently on or off, leave your network. If that is not acceptable, run a
decision model locally instead - see [Running locally](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#running-locally) - and
nothing leaves the house.

What does **not** happen: nothing is stored by this project, there is no
telemetry of its own, and the optional LLM backend is configured separately —
point it at Ollama on your own machine and the prose path never leaves the
house either. When it fills in a [script's fields](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#scripts-with-fields), the
LLM is sent the utterance, that script's name, description and fields, the
room you are in and the time; not the rest of the home. With
[web search](https://github.com/the-sof/home-assistant-typesafe-conversation-agent#web-search) on, the search queries the model writes go to the
search you chose. Diagnostics downloads redact the API key, the server and
search URLs, and your entity IDs.

When you use TypeSafe's hosted API, its handling of what you send is governed by
their [terms](https://typesafe.ai/legal/mca) and
[data processing agreement](https://typesafe.ai/data-processing), not by this
project.

## Cost

Input tokens only. State is billed once per request rather than once per
question — verified against the API, and it is what makes the speculative
fan-out cheap: asking a question you end up discarding costs almost nothing.

Measured against `jev-1.13.0` at $42/Btok, a 29-entity home came to ~6.5k tokens
per request, about **$0.00027**. Both the rate and a model's token accounting
are TypeSafe's to change, and a later model will price differently — treat these
as an order of magnitude, and check
[your console](https://console.typesafe.ai/) and
[typesafe.ai](https://typesafe.ai/) for what you will actually be billed.

## Running locally

Any server that speaks TypeSafe's System One API works, not only TypeSafe's own.
Ollama does from version 0.35, with local decision models such as `nimble`
(9B, Bespoke Labs) and `tev1` (4B and 0.8B, Together AI):

```sh
ollama pull nimble
```

Then add the integration with Ollama's address as the server URL, for example
`http://<ollama-host>:11434`, and no API key.

Setup tests the model before saving anything. It confirms the model is a
decision model, finds the most options one question may offer, and times one
request shaped like a real one from your home - then tells you how long that
took, before you rely on it. The request timeout is set from that figure. All of
it is cached until you change the server or model under **Reconfigure**; after a
big change to what you expose to Assist, reconfigure so the timing is measured
again. Nothing about a particular server is built in, so a new model or a new
server needs no change to the integration.

How long a model stays loaded between requests is left to the server. By
default the integration never asks Ollama to keep one in memory and never loads
one before it is needed. The one exception is opt-in: **Keep the model loaded**
in the language-model settings loads that model at startup and keeps it
resident, for when a cold load would outlast its answer timeout. Turn it off
again under **Reconfigure**. Ollama unloads an idle model
after five minutes by default (`OLLAMA_KEEP_ALIVE` changes that), and the first
request after that pays the load time; the timeout allows for it.

Three things are worth knowing before you rely on it:

- **Expect it to be slow for now.** A local model scores every question against
  your whole home, so one request is far more work than the answer suggests:
  tens of seconds even on a recent GPU, and minutes on a CPU-only machine. That
  is too slow for voice. The setup screen shows the figure for your own server.
- **The context window is set on the server.** Ollama loads each model with a
  default context (`nimble` 8,194 tokens, `tev1` 2,050), and a request that does
  not fit is rejected, never truncated. If diagnostics show "prompt has N
  tokens; expected 1–M", raise it with a Modelfile:

  ```
  FROM tev1:4b
  PARAMETER num_ctx 16384
  ```

  then `ollama create tev1-16k -f Modelfile`, and choose `tev1-16k`.
- **Thresholds were fitted to Jev.** Every routing threshold in `const.py` comes
  from `jev-1.13.0`'s answers, and the routing tests replay `jev-1.14.0` too.
  Another model's probabilities are shaped differently, so it may ask "did you
  mean…?" too often or act too readily until they are re-fitted with
  `scripts/calibrate.py --base-url`.

Local servers also accept fewer options per question - Ollama allows 26 - so a
home with more exposed entities than that picks its device in two requests: the
first narrows to the likeliest kind of device and room, the second chooses among
just those. That follows TypeSafe's
[hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md)
pattern. If no room was said and one kind of device alone is too many to offer,
the agent asks which room.

## How it works

For readers who want the internals: how a request is decided, and why it is
fast and cheap.

A System One model returns typed, calibrated judgements rather than text. Jev is
the one available today and the default; the integration is not written around
it, so a later model is a config change. Every device command and state query
is resolved from those judgements in ordinary Python, in a single API call. An
LLM is used only where something has to be written — splitting a request that
contains several commands, answering a general question, and filling in a
script's fields — so the slow path is only taken when it is needed.

Measured against the same utterances, this replaces a 2–8 s LLM turn with a
~250 ms one for anything that controls or inspects the home.

```
utterance
   ▼   (HA's sentence matcher has already taken what it knows)
entity catalog ── cached, rebuilt when exposure changes
   ▼
ONE System One call
   state:     the request + every exposed entity and area
   questions: fixed + from the catalog + one per device kind
   ▼
route() ── a pure function over the answers
   ├─ command  → intent.async_handle → your services
   ├─ query    → sentence matcher, then HassGetState
   ├─ compound → LLM splits it → parallel Jev calls → in order
   ├─ general  → LLM answers, after a web search if needed
   └─ unsure   → sentence matcher → LLM → "I'm not sure"
```

Home Assistant's own sentence matcher runs first, so the phrasings it
recognises never reach this agent; what does is the fuzzy and the compound.
The questions are 8 fixed ones, 3 built from the catalog, one per kind of device
present, and a few conditional ones.

### Speculative fan-out

Every question the router might need is asked up front, including the branches
that will turn out to be irrelevant. Measured against `jev-1.13.0`, an extra
question costs about **97 input tokens** and almost no extra latency, while a
second round trip would cost ~250 ms. So we ask "what should happen to the
locks?" on every single request and simply do not read the answer unless the
request turned out to be about locks.

A confident but wrong answer on a branch nobody reads is free. That is the
contract, and there is a test that holds it: `is everything locked up` returns
`action_lock = lock` at 0.99 confidence, and the router correctly treats the
utterance as a question and locks nothing.

### Two axes, not one

A branch is acted on only when **confidence** clears its bar *and* the
**margin** between the top two options is at least 0.15. They fail differently:
a distribution can look confident overall while the top two options remain
effectively tied.

| confidence | behaviour |
| --- | --- |
| ≥ 0.75 | act, brief acknowledgement |
| 0.50–0.75 | act, but name the target out loud, so a wrong guess is correctable |
| 0.30–0.50 (target only) | ask which device was meant |
| below, or a near-tie | hand back to the sentence matcher, then the LLM |

Clarification is only ever used for an ambiguous *target*. "Which lamp?" is a
short, natural question. "Did you want to turn something on?" is not, so a weak
category or action goes to the fallback ladder instead.

### Values

Jev cannot emit text, so it never invents a number. Code finds the candidate
spans in the utterance and Jev picks which one the user meant:

- `"set the living room lights to 30%"` → candidates `["30%"]` → picked → `brightness_pct: 30`
- `"close the blinds halfway"` → `halfway` → `position: 50`
- `"turn the volume down a bit"` → no value in the text, so a direction and a
  magnitude question → `volume_step: -10`
- `"make the lights a bit warmer"` → a closed set of colour names → `2700 K`

Temperature units come from the entity, or failing that from your Home
Assistant configuration. They are never asked of the model.

## Debugging

### Is the agent even being asked?

With *Prefer handling commands locally* on, Home Assistant's own sentence
matcher keeps every phrasing it recognises, so a well-formed command like
"turn off the living room light" never reaches this integration. In the
**Settings → Voice assistants → Debug** trace that shows up as:

```
processed_locally: true
Natural language processing   0.01s
```

That is the design working, not a failure. A request this agent handled looks
like `processed_locally: false` and roughly 250ms. To exercise it, use
phrasings the matcher cannot parse — "it's too dark in here", "get the coffee
going", "add milk to the shopping list", "turn off the lamp and start the
vacuum" — or turn on `bypass_local_intents` to send everything here.

### Download diagnostics

**Settings → Devices & Services → TypeSafe Conversation → ⋮ → Download
diagnostics** gives the last 20 requests with the route taken, the reason, the
resolved target, and every answer's probability distribution and confidence:

```json
{
  "utterance": "get the coffee going",
  "route": "command", "reason": "switch.turn_on",
  "target": "switch.coffee_maker",
  "category":      {"choice": "command", "confidence": 1.0,  "margin": 1.0},
  "target_entity": {"choice": "switch.coffee_maker", "confidence": 1.0},
  "action":        {"choice": "turn_on", "confidence": 0.99,
                    "top": {"turn_on": 0.99, "not_targeted": 0.01}},
  "latency_ms": 244, "input_tokens": 6482
}
```

API keys and the LLM base URL are redacted, and the catalog is reported as
counts rather than entity ids, so the file is safe to attach to an issue.

Home Assistant has a conversation-trace mechanism too, and this integration
writes to it — but nothing in Home Assistant reads it back (there is no
websocket command and no UI), so diagnostics is the usable route.

### Debug log

```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.typesafe_conversation: debug
```

or, without a restart, **Developer Tools → Actions → `logger.set_level`** with
`custom_components.typesafe_conversation: debug`. Each request then logs the
route, latency and token count, followed by one line per answer:

```
Routed 'get the coffee going' -> command (switch.turn_on) in 244ms, ...
  category       command              conf 1.00 margin 1.00  {...}
  target_entity  switch.coffee_maker  conf 1.00 margin 1.00  {...}
  action         turn_on              conf 0.99 margin 0.98  {...}
```

### If the LLM path times out

`LLM could not answer: Timed out after 30.0s` means the model backing the prose
path is too slow. That path only writes sentences, so it does not need to be
the model you would choose for reasoning — a small local model, or a hosted
one, is usually the right trade. Raise `llm_timeout` in the integration options
if you would rather wait.

## Development

```sh
python3.14 -m venv .venv   # Home Assistant 2026.5+ needs Python 3.14.2+
.venv/bin/pip install syrupy ruff \
    "pytest-homeassistant-custom-component==0.13.367"
.venv/bin/pip install "gazetteer-matcher==1.1.0" "hassil==3.12.1" \
    "home-assistant-intents==2026.8.28"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

CI runs those three against both the oldest supported Home Assistant and the
current one, plus Home Assistant's `hassfest` and the HACS validator. See
[CONTRIBUTING.md](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/blob/main/CONTRIBUTING.md) before opening a pull request.

The harness version is a **test-environment** choice, not the supported range —
see *Install* for that. Each release pins exactly one core version
(`0.13.367` → `homeassistant==2026.9.4`, `0.13.329` → `2026.5.0`), and a mismatch
against an already-installed `homeassistant` produces confusing import errors.

The third line is the `conversation` component's own requirements, which the
harness does not pull in — without them the import fails at `hassil`. Those pins
move with the core version, so if you change the harness, read them off the
`homeassistant` you actually installed:

```sh
.venv/bin/python - <<'PY'
import json, pathlib, homeassistant as h
manifest = (pathlib.Path(h.__file__).parent
            / "components/conversation/manifest.json")
print(json.loads(manifest.read_text())["requirements"])
PY
```

CI runs the suite against both ends of the supported range on every push and
pull request, deriving those requirements the same way.

The routing tests replay **real recorded Jev responses** from
`tests/fixtures/answers/` (`jev-1.13.0`) and `tests/fixtures/answers_1_14/`
(`jev-1.14.0`), so they describe how the model actually behaves rather than how
we imagine it does. To re-record them, or to re-fit the thresholds in
`const.py` against your own home:

```sh
set -a; . ./.env; set +a
.venv/bin/python scripts/calibrate.py --csv out.csv \
    --record tests/fixtures/answers
```

A cases file for `--cases` holds `[utterance, expected]` pairs, or
`[utterance, expected, language]` to send a request in another language.

Every threshold is a named constant in `const.py` precisely so it can be
re-fitted rather than argued about.

To pull your own home's catalog and calibrate against it:

```sh
.venv/bin/python scripts/pull_home.py --out my_home.json
.venv/bin/python scripts/calibrate.py --home my_home.json \
    --cases tests/fixtures/scripted_cases.json --area <your-area-id>
```

`pull_home.py` reads exposure settings over the WebSocket API (they are not in
the REST API) and redacts latitude/longitude by default, because weather
integrations name entities after the station's coordinates.

A catalog pulled from a live instance describes the home it came from — room
layout, device brands, which security devices exist — so keep it out of version
control. `.gitignore` already excludes `tests/fixtures/real_*` and
`docs/calibration-real-*.csv` for that reason. The fixtures that ship with this
repo are synthetic.

### A note on confidence

Jev derives confidence as `(n * p_top - 1) / (n - 1)`, so the *same* probability
scores differently depending on how many options the question offered. A
two-option Choice at p=0.66 scores 0.31; a seven-option Choice at the same
probability scores 0.60. The action gate therefore thresholds the chosen
option's **probability**, not its confidence — thresholding confidence made
every script command fall back on a script-heavy home.
