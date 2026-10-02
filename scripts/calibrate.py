#!/usr/bin/env python
"""Replay utterances against a live Jev key and dump every answer.

Every routing threshold in const.py is a first guess. This harness exists so
they can be re-fit against real answers on a real home, rather than argued
about. It talks to the API directly and does not need Home Assistant running.

    TYPESAFE_API_KEY=... .venv/bin/python scripts/calibrate.py
    .venv/bin/python scripts/calibrate.py --csv out.csv --home tests/fixtures/home.json

Any System One server works, which is how a local model's thresholds get
re-fitted. Pass the cap the integration discovered for it (diagnostics show
it); a home over the cap leaves out target_entity, as the integration does:

    .venv/bin/python scripts/calibrate.py --base-url http://localhost:11434 \\
        --model nimble --max-options 26 --timeout 600
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp

from custom_components.typesafe_conversation.const import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MAX_CHOICE_OPTIONS,
    SYSTEM_ONE_PATH,
)
from custom_components.typesafe_conversation.entities import CatalogArea, CatalogEntity
from custom_components.typesafe_conversation.extraction import extract
from custom_components.typesafe_conversation.questions import (
    build_questions,
    estimate_tokens,
)

# Utterance -> what the router should end up doing. Used as a regression set,
# not as an assertion on any particular probability.
REGRESSION: list[tuple[str, str]] = [
    ("get the coffee boiling", "command switch.coffee_maker turn_on"),
    ("it's too bright in here", "command light dimmer (speaker area)"),
    ("set the living room lights to 30%", "command light set_brightness 30"),
    ("make the lights a bit warmer", "command light set_color warm_white"),
    ("close the blinds halfway", "command cover set_position 50"),
    ("set the thermostat to 21 degrees", "command climate set_temperature 21"),
    ("turn the volume down a bit", "command media_player quieter"),
    ("turn off all the lights and lock the front door", "compound n=2"),
    ("dim the bedroom lamp and start the dishwasher", "compound n=2"),
    ("is the garage door open?", "query device_state"),
    ("what's the temperature in the bedroom?", "query temperature"),
    ("how many lights are on?", "query count"),
    ("what time is it", "query time_or_date"),
    ("is everything locked up", "query needs_prose"),
    ("who won the world cup in 1998", "information"),
    ("never mind", "cancel"),
    ("unlock the front door", "command lock unlock (risky)"),
    ("play some jazz in the kitchen", "command media_player search_and_play"),
    ("turn on the thing in the corner", "low confidence -> clarify"),
    ("turn off everything", "command whole_house"),
    ("goodnight", "command scene/script"),
    ("asdfgh", "unclear"),
]


def load_home(
    path: Path,
) -> tuple[dict, tuple[CatalogEntity, ...], tuple[CatalogArea, ...], tuple[str, ...]]:
    home = json.loads(path.read_text())
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
            area_name=next(
                (a["name"] for a in home["areas"] if a["id"] == e.get("area")), None
            ),
            floor_name=None,
            domain=e["domain"],
            device_class=(e.get("attrs") or {}).get("device_class"),
            supported_features=0,
        )
        for e in home["entities"]
    )
    domains = tuple(sorted({e.domain for e in entities}))
    return home, entities, areas, domains


async def ask(
    session: aiohttp.ClientSession,
    key: str | None,
    model: str,
    state: dict,
    questions: dict,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30,
) -> tuple[dict, float]:
    started = time.monotonic()
    async with session.post(
        base_url.rstrip("/") + SYSTEM_ONE_PATH,
        json={"state": state, "model": model, "questions": questions},
        headers={"Authorization": f"Bearer {key}"} if key else {},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        body = await response.json()
        if response.status != 200:
            raise SystemExit(f"HTTP {response.status}: {json.dumps(body)[:800]}")
    return body, (time.monotonic() - started) * 1000


def top2(answer: dict) -> str:
    probs = sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])[:2]
    return " / ".join(f"{k}={v:.2f}" for k, v in probs)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path("tests/fixtures/home.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TYPESAFE_BASE_URL", DEFAULT_BASE_URL),
        help="any System One server (default: $TYPESAFE_BASE_URL, then TypeSafe)",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=MAX_CHOICE_OPTIONS,
        help="the server's option cap per Choice (Ollama: 26)",
    )
    parser.add_argument("--timeout", type=float, default=30, help="seconds per call")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--only", help="run just the utterances containing this text")
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON file of [utterance, expected] pairs, instead of REGRESSION",
    )
    parser.add_argument("--area", default="kitchen", help="area the speaker is in")
    parser.add_argument(
        "--record",
        type=Path,
        help="save each raw response as a fixture in this directory",
    )
    args = parser.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and args.base_url.rstrip("/") == DEFAULT_BASE_URL:
        # Only TypeSafe's own API needs one; a local server usually does not.
        raise SystemExit("TYPESAFE_API_KEY is not set (try: set -a; . ./.env; set +a)")

    home, entities, areas, domains = load_home(args.home)
    source = (
        [tuple(c) for c in json.loads(args.cases.read_text())]
        if args.cases
        else REGRESSION
    )
    cases = [c for c in source if not args.only or args.only.lower() in c[0].lower()]

    rows: list[dict] = []
    total_in = total_out = 0
    async with aiohttp.ClientSession() as session:
        for utterance, expected in cases:
            extraction = extract(
                utterance,
                want_media="media_player" in domains,
                want_color="light" in domains,
            )
            questions = build_questions(
                entities=entities,
                areas=areas,
                domains=domains,
                extraction=extraction,
                max_options=args.max_options,
            )
            state = {
                "request": {
                    "text": utterance,
                    "language": "en",
                    "spoken_from_area": args.area,
                    "local_time": "2026-09-20T22:40",
                    "weekday": "Sunday",
                },
                "home": home,
            }
            body, latency = await ask(
                session,
                key,
                args.model,
                state,
                questions,
                base_url=args.base_url,
                timeout=args.timeout,
            )
            answers = body["answers"]
            usage = body.get("usage", {})
            total_in += usage.get("input_tokens", 0)
            total_out += usage.get("output_tokens", 0)

            category = answers["category"]
            print(f"\n\033[1m{utterance}\033[0m")
            print(f"  expected : {expected}")
            print(
                f"  category : {category['choice']} "
                f"(conf {category['confidence']:.2f})  [{top2(category)}]"
            )
            print(f"  compound : {answers['compound']['noul']:.2f}")
            for qid in (
                "scope",
                "target_domain",
                "target_entity",
                "target_area",
                "query_kind",
                "change_direction",
                "magnitude",
                "value_pick",
                "color_pick",
                "media_query_span",
            ):
                if qid not in answers:
                    continue
                a = answers[qid]
                print(
                    f"  {qid:<16}: {a['choice']:<28} "
                    f"conf {a['confidence']:.2f}  [{top2(a)}]"
                )
            domain = answers["target_domain"]["choice"]
            if (akey := f"action_{domain}") in answers:
                a = answers[akey]
                print(
                    f"  {akey:<16}: {a['choice']:<28} "
                    f"conf {a['confidence']:.2f}  [{top2(a)}]"
                )
            print(
                f"  risky {answers['risky']['noul']:.2f} | here "
                f"{answers['here_relative']['noul']:.2f} | "
                f"{len(questions)} questions, {usage.get('input_tokens')} in, "
                f"{latency:.0f}ms"
            )

            row = {
                "utterance": utterance,
                "expected": expected,
                "latency_ms": round(latency),
                "input_tokens": usage.get("input_tokens"),
                "questions": len(questions),
            }
            for qid, a in answers.items():
                if a["type"] == "noul":
                    row[qid] = round(a["noul"], 3)
                else:
                    row[qid] = a.get("choice", a.get("score"))
                    row[f"{qid}__conf"] = round(a.get("confidence", 0), 3)
            rows.append(row)

            if args.record:
                args.record.mkdir(parents=True, exist_ok=True)
                slug = re.sub(r"[^a-z0-9]+", "_", utterance.lower()).strip("_")[:60]
                (args.record / f"{slug}.json").write_text(
                    json.dumps(
                        {
                            "utterance": utterance,
                            "expected": expected,
                            "spoken_from_area": args.area,
                            "response": body,
                        },
                        indent=2,
                    )
                )

    print(
        f"\n\033[1m{len(rows)} utterances | {total_in} input tokens "
        f"| ${total_in * 42 / 1e9:.5f}\033[0m"
    )
    print(f"estimated question-set tokens: {estimate_tokens(questions)}")

    if args.csv:
        fields: list[str] = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
