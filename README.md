# TypeSafe Conversation for Home Assistant

[![tests](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml/badge.svg)](https://github.com/the-sof/home-assistant-typesafe-conversation-agent/actions/workflows/test.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![HACS: custom](https://img.shields.io/badge/HACS-custom-orange.svg)](https://hacs.xyz/docs/faq/custom_repositories/)

A Home Assistant conversation agent that decides with a
[TypeSafe System One](https://docs.typesafe.ai) model instead of an LLM.

> **Status: early.** Expect rough edges, and please
> [report them](../../issues/new?template=bug_report.yml). Requires a
> [TypeSafe](https://console.typesafe.ai/) API key, which is metered — see
> [Cost](#cost).

A System One model returns typed, calibrated judgements rather than text. Jev is
the one available today and the default; the integration is not written around
it, so a later model is a config change. Every device command and state query
is resolved from those judgements in ordinary Python, in a single API call. An
LLM is used for exactly two things — splitting a request that contains several
commands, and answering a general question — so the slow path is only taken when
something actually has to be written in prose.

Measured against the same utterances, this replaces a 2–8 s LLM turn with a
~250 ms one for anything that controls or inspects the home.

## How it works

```
utterance
   │   Home Assistant's own sentence matcher has already taken the phrasings
   │   it recognises; what reaches us is the fuzzy and the compound.
   ▼
entity catalog  ── cached, rebuilt on registry and exposure changes
   ▼
ONE System One call:  state    = the request + every exposed entity and area
               questions = 8 fixed + 3 built from the catalog
                         + 1 per device kind present + a few conditional ones
   ▼
route()  ── a pure function over the answers
   ├─ command   → intent.async_handle → your services
   ├─ query     → sentence matcher, then HassGetState / HassClimateGetTemperature
   ├─ compound  → LLM splits it → N parallel Jev calls → run in order
   ├─ general   → LLM answers in prose
   └─ unsure    → sentence matcher → LLM → "I'm not sure"
```

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
decision model locally instead - see [Running locally](#running-locally) - and
nothing leaves the house.

What does **not** happen: nothing is stored by this project, there is no
telemetry of its own, and the optional LLM backend is configured separately —
point it at Ollama on your own machine and the prose path never leaves the
house either. Diagnostics downloads redact the API key, both server URLs and
your entity IDs.

When you use TypeSafe's hosted API, its handling of what you send is governed by
their [terms](https://typesafe.ai/legal/mca) and
[data processing agreement](https://typesafe.ai/data-processing), not by this
project.

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
  from `jev-1.13.0`'s answers. Another model's probabilities are shaped
  differently, so it may ask "did you mean…?" too often or act too readily until
  they are re-fitted with `scripts/calibrate.py --base-url`.

Local servers also accept fewer options per question - Ollama allows 26 - so a
home with more exposed entities than that picks its device in two requests: the
first narrows to the likeliest kind of device and room, the second chooses among
just those. That follows TypeSafe's
[hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md)
pattern. If no room was said and one kind of device alone is too many to offer,
the agent asks which room.

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

## Install

**Requires Home Assistant 2026.5.0 or newer.** Earlier releases do not report a
failed service call back to the conversation agent, so a command that no entity
could carry out would be announced as if it had worked. The integration is
tested against 2026.5.0 and the current release on every change.

### Through HACS

Not in the default HACS store yet, so add it as a custom repository: in HACS,
**⋮ → Custom repositories**, paste this repository's URL, choose category
**Integration**, then **Add**. It will then appear in HACS for install and for
update notifications.

### By hand

Copy `custom_components/typesafe_conversation` into your Home Assistant `config`
directory.

### Either way

Restart Home Assistant, then **Settings → Devices & Services → Add Integration →
TypeSafe Conversation**. You will need an API key from
[console.typesafe.ai](https://console.typesafe.ai/). Finally, set it as the
conversation agent under **Settings → Voice assistants**.

The LLM step is optional. Without it the agent still handles every command and
query; it just cannot split compound requests or answer general questions.
Ollama and any OpenAI-compatible endpoint (OpenRouter, vLLM, …) are supported.

### Options

Set when you add the integration, and changeable afterwards under
**Settings → Devices & Services → TypeSafe Conversation**: the server, model,
timeout and language-model settings under **Reconfigure** in the entry's menu,
and `always_confirm_risky`, `bypass_local_intents` and
`inline_entity_descriptions` under each conversation agent's **Configure**.

| option | default | what it does |
| --- | --- | --- |
| `base_url` | `https://api.typesafe.ai` | The System One server. Any compatible server works; see [Running locally](#running-locally). |
| `api_key` | — | Required by TypeSafe's hosted API. Leave blank for a server that doesn't check one, such as Ollama. |
| `model` | `jev-latest` | Which decision model to use, chosen from what the server offers. On TypeSafe, `jev-latest` tracks the newest Jev. |
| `api_timeout` | measured | How long to wait for an answer. Set at setup from a timed, real-sized request plus the cold-load time, never below 6 s. |
| `always_confirm_risky` | **on** | Ask before unlocking a door, opening a garage or disarming an alarm, however sure the model is. **Turning this off lets confident requests through silently** — the model's judgement becomes the only gate. |
| `bypass_local_intents` | off | Send every command here, including ones Home Assistant's own sentence matcher recognises. Off is recommended; see [below](#leave-prefer-handling-commands-locally-on). |
| `inline_entity_descriptions` | off | Describe every entity inside each question rather than once in the shared state. Roughly doubles the tokens. Only worth it if the agent picks the wrong device. |
| `llm_backend` | none | `ollama`, an OpenAI-compatible endpoint, or unset. Used only for compound requests and general questions. |
| `llm_base_url` | `http://localhost:11434` (Ollama) | Where that backend lives. Point it at your own machine to keep the prose path local. |
| `llm_model` | — | Model name on that backend. |
| `llm_api_key` | — | If the backend needs one. Redacted in diagnostics. |
| `llm_timeout` | `30` s | How long to wait for the LLM before giving up. Only the prose path is affected. A large local model that must load first may need more. |
| `llm_keep_loaded` | off | Ollama only. Load the language model at startup and keep it in memory, so the first general question after a pause is fast. Holds the memory while Home Assistant runs; off leaves unloading to the server. |
| `llm_referer`, `llm_title` | project defaults | Sent as `HTTP-Referer` and `X-Title`; OpenRouter uses them for attribution. |

### Leave "prefer handling commands locally" on

The agent advertises `ConversationEntityFeature.CONTROL`. With prefer-local on,
Home Assistant's sentence matcher keeps every phrasing it recognises, and only
`HassGetState`, `HassMediaSearchAndPlay` and everything it *failed* to parse
reach this agent. That is deliberate: there is no point spending an API call
beating hassil at "turn off the kitchen lights", and it concentrates the traffic
on the requests where Jev earns its keep. Set `bypass_local_intents` if you want
to compare the two regimes.

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
Routed 'get the coffee going' -> command (switch.turn_on) in 244ms, 6482 input tokens
  category         command                      conf 1.00 margin 1.00  {'command': 1.0}
  target_entity    switch.coffee_maker          conf 1.00 margin 1.00  {...}
  action           turn_on                      conf 0.99 margin 0.98  {...}
```

### If the LLM path times out

`LLM could not answer: Timed out after 30.0s` means the model backing the prose
path is too slow. That path only writes sentences, so it does not need to be
the model you would choose for reasoning — a small local model, or a hosted
one, is usually the right trade. Raise `llm_timeout` in the integration options
if you would rather wait.

## Development

```sh
python3.14 -m venv .venv        # Home Assistant 2026.5+ requires Python 3.14
.venv/bin/pip install "pytest-homeassistant-custom-component==0.13.367" syrupy ruff
.venv/bin/pip install "gazetteer-matcher==1.1.0" "hassil==3.12.1" "home-assistant-intents==2026.8.28"
.venv/bin/python -m pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

CI runs those three against both the oldest supported Home Assistant and the
current one, plus Home Assistant's `hassfest` and the HACS validator. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

The harness version is a **test-environment** choice, not the supported range —
see *Install* for that. Each release pins exactly one core version
(`0.13.367` → `homeassistant==2026.9.4`, `0.13.329` → `2026.5.0`), and a mismatch
against an already-installed `homeassistant` produces confusing import errors.

The third line is the `conversation` component's own requirements, which the
harness does not pull in — without them the import fails at `hassil`. Those pins
move with the core version, so if you change the harness, read them off the
`homeassistant` you actually installed:

```sh
.venv/bin/python -c "import json,pathlib,homeassistant as h; \
  print(json.loads((pathlib.Path(h.__file__).parent/'components'/'conversation'/'manifest.json').read_text())['requirements'])"
```

CI runs the suite against both ends of the supported range on every push and
pull request, deriving those requirements the same way.

The routing tests replay **real recorded Jev responses** from
`tests/fixtures/answers/`, so they describe how the model actually behaves
rather than how we imagine it does. To re-record them, or to re-fit the
thresholds in `const.py` against your own home:

```sh
set -a; . ./.env; set +a
.venv/bin/python scripts/calibrate.py --csv out.csv --record tests/fixtures/answers
```

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
