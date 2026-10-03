# Contributing

Thanks for looking. This is a small project maintained by one person in his
spare time, so the most useful thing you can do is make it cheap for me to
understand what you found.

## Reporting a bug

Use the [bug report form](../../issues/new?template=bug_report.yml). The one
field that decides whether a report is actionable is the **Assist pipeline
trace** — without it, a voice failure is almost impossible to diagnose, because
the interesting part is which question the model answered and how confidently.

Get one from **Settings → Voice assistants → your pipeline → ⋮ → Debug**, run
the failing phrase, and copy the trace.

**Redact it first.** A trace contains your entity IDs, room names and the exact
words you said. Replace anything you would not post publicly — the structure is
what matters, not the names.

For a security problem, see [SECURITY.md](SECURITY.md) instead. Do not open a
public issue.

## Development setup

Python 3.14 is required, because Home Assistant requires it.

```sh
python3.14 -m venv .venv
.venv/bin/pip install "pytest-homeassistant-custom-component==0.13.367" syrupy ruff
.venv/bin/pip install "gazetteer-matcher==1.1.0" "hassil==3.12.1" "home-assistant-intents==2026.8.28"
.venv/bin/python -m pytest
```

The second line installs the `conversation` component's own requirements, which
the test harness does not pull in — without them the import fails at `hassil`.
Those pins move with the Home Assistant version; see the README's *Development*
section for how to read the right ones off the core you installed.

Before opening a pull request:

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

CI runs all three against both the oldest supported Home Assistant and the
current one.

## How the tests work

Routing tests **replay recorded API responses** from `tests/fixtures/answers/`
and `tests/fixtures/scripted_answers/`. They do not call the TypeSafe API, so
they need no key, cost nothing and are deterministic. `route()` is a pure
function over those answers, which is what makes this possible — please keep it
that way. If you need new state in a routing decision, pass it in as an
argument rather than reaching for `hass`.

When a change alters behaviour, the honest test is usually an existing fixture
replayed with one different input, not a new recording.

## Two rules about data

**Never commit anything captured from a real home.** A catalogue pulled from a
live Home Assistant profiles the home it came from: room layout, which devices
exist, which security devices exist. `scripts/pull_home.py` writes to paths that
`.gitignore` already excludes (`tests/fixtures/real_*`, `private/`,
`docs/calibration-real-*.csv`) — keep it that way. The committed fixtures are
synthetic and should stay synthetic.

The same applies to code comments, commit messages and pull request
descriptions. Describe the defect, not your house.

**The fixture corpus is not training data.** Those recorded responses are output
from a commercial API. TypeSafe's terms permit publishing them here, but
prohibit using Output to distil or train a model imitating the service, or to
build a competing one. Please do not use this repository for that.

## Pull requests

- One concern per pull request.
- A commit message says what changed and why it is correct — not the story of
  how the bug was found.
- New behaviour needs a test that fails without the change. If you cannot write
  one, say so in the PR and explain why.
- Thresholds live as named constants in `const.py` so they can be re-fitted
  against recorded answers rather than argued about. If you want to move one,
  bring the evidence.

I would rather receive an issue describing a problem than a large pull request
solving it a way I would not have chosen. For anything beyond a small fix, open
an issue first so we can agree the shape.
