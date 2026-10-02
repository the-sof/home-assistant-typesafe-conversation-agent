"""Find candidate values in the utterance so Jev can pick between them.

Jev returns typed answers, never generated text, so it cannot hand back "30"
or "some jazz". The pattern - straight from the function-calling cookbook - is
that code finds candidates and Jev selects which one the user meant. Code then
parses the selected span. Nothing can be hallucinated, because every option was
lifted from the utterance or from a closed set we control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Colour names accepted by homeassistant.util.color.color_name_to_rgb, trimmed
# to the ones a person actually says out loud. Offering all 149 CSS names would
# spend tokens on "lightgoldenrodyellow".
COLOR_NAMES: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "white",
    "black",
    "brown",
    "grey",
    "cyan",
    "magenta",
    "gold",
    "silver",
    "beige",
    "turquoise",
    "teal",
    "lime",
    "olive",
    "navy",
    "maroon",
    "violet",
    "indigo",
    "salmon",
    "coral",
    "crimson",
    "lavender",
    "plum",
    "orchid",
    "khaki",
    "tan",
    "ivory",
    "mintcream",
    "peachpuff",
    "aqua",
    "azure",
    "bisque",
    "chocolate",
    "tomato",
    "wheat",
    "skyblue",
    "seagreen",
    "hotpink",
    "darkred",
    "darkgreen",
    "darkblue",
    "darkorange",
    "lightblue",
    "lightgreen",
    "lightyellow",
    "lightpink",
)

# Colour-temperature presets. Users say "warmer", not "2700 kelvin".
COLOR_TEMP_PRESETS: dict[str, int] = {
    "warm_white": 2700,
    "soft_white": 3000,
    "neutral_white": 4000,
    "cool_white": 5000,
    "daylight": 6500,
}

# Words that name a value without being a number.
_VALUE_WORDS: dict[str, int] = {
    "all the way": 100,
    "all the way up": 100,
    "all the way open": 100,
    "full": 100,
    "fully": 100,
    "max": 100,
    "maximum": 100,
    "highest": 100,
    "halfway": 50,
    "half": 50,
    "half way": 50,
    "a crack": 10,
    "a bit open": 10,
    "lowest": 1,
    "minimum": 1,
    "min": 1,
    "all the way down": 0,
    "all the way closed": 0,
    "off": 0,
}

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

_NUMERIC_RE = re.compile(
    r"-?\d{1,3}(?:\.\d+)?\s*(?:%|percent|degrees?|deg|°\s*[cf]?|°)?",
    re.IGNORECASE,
)
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b"
    r"(?:\s*(?:percent|%|degrees?))?",
    re.IGNORECASE,
)
_VALUE_WORD_RE = re.compile(
    r"\b("
    + "|".join(re.escape(w) for w in sorted(_VALUE_WORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# Prepositions and verbs that separate "what to play" from "where to play it".
# Verbs and prepositions that separate *what* from *where* or *which list*.
# Longer alternatives first so "put on" wins over "on".
_CHUNK_SPLIT_RE = re.compile(
    r"\b(?:start playing|queue up|cross off|tick off|take off|put on|"
    r"play|stream|add|append|remove|delete|buy|get|need|"
    r"on|in|to|from|by|at|over|through|via)\b",
    re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class ValueCandidate:
    """A span of the utterance that might be a numeric value."""

    span: str
    """Exactly as it appears in the text. This is the Choice option key."""

    value: float
    """What the span parses to."""

    unit: str | None
    """'%', 'c', 'f', or None when the text gave no unit."""


@dataclass(slots=True, frozen=True)
class Extraction:
    """Everything code found in the utterance before Jev saw it."""

    values: tuple[ValueCandidate, ...] = ()
    colors_mentioned: bool = False
    colors_named: tuple[str, ...] = ()
    """Colour names said outright, so a server with a small option cap can be
    offered just those instead of every colour."""
    media_chunks: tuple[str, ...] = ()

    @property
    def has_values(self) -> bool:
        return bool(self.values)

    def by_span(self, span: str) -> ValueCandidate | None:
        for candidate in self.values:
            if candidate.span == span:
                return candidate
        return None


def _unit_of(text: str) -> str | None:
    lowered = text.lower()
    if "%" in lowered or "percent" in lowered:
        return "%"
    if "°c" in lowered or lowered.rstrip().endswith(" c"):
        return "c"
    if "°f" in lowered or lowered.rstrip().endswith(" f"):
        return "f"
    if "degree" in lowered or "deg" in lowered or "°" in lowered:
        # Degrees with no letter: the caller resolves the unit from the entity
        # or from hass.config. We never ask Jev what unit the house uses.
        return "deg"
    return None


def find_values(utterance: str) -> tuple[ValueCandidate, ...]:
    """Find every span that could be a numeric setting.

    Deliberately over-finds. Jev decides which span is the value the user
    meant, including deciding that none of them is (a number can just be part
    of a device's name).
    """
    found: list[ValueCandidate] = []
    seen: set[str] = set()

    def add(span: str, value: float, unit: str | None) -> None:
        span = span.strip()
        if not span or span.lower() in seen:
            return
        seen.add(span.lower())
        found.append(ValueCandidate(span=span, value=value, unit=unit))

    for match in _NUMERIC_RE.finditer(utterance):
        raw = match.group(0).strip()
        digits = re.search(r"-?\d{1,3}(?:\.\d+)?", raw)
        if digits is None:
            continue
        add(raw, float(digits.group(0)), _unit_of(raw))

    for match in _WORD_NUMBER_RE.finditer(utterance):
        raw = match.group(0).strip()
        word = match.group(1).lower()
        add(raw, float(_NUMBER_WORDS[word]), _unit_of(raw))

    for match in _VALUE_WORD_RE.finditer(utterance):
        raw = match.group(0).strip()
        add(raw, float(_VALUE_WORDS[raw.lower()]), "%")

    return tuple(found)


def mentions_color(utterance: str) -> bool:
    """Whether a colour question is worth asking at all."""
    lowered = utterance.lower()
    if any(re.search(rf"\b{name}\b", lowered) for name in COLOR_NAMES):
        return True
    return any(
        word in lowered
        for word in (
            "colour",
            "color",
            "warm",
            "cool",
            "cosy",
            "cozy",
            "daylight",
            "white",
            "tint",
            "hue",
        )
    )


def colors_named(utterance: str) -> tuple[str, ...]:
    """Colour names the utterance says outright, in the order of COLOR_NAMES."""
    lowered = utterance.lower()
    return tuple(name for name in COLOR_NAMES if re.search(rf"\b{name}\b", lowered))


def media_chunks(utterance: str) -> tuple[str, ...]:
    """Split the utterance into candidate 'what to play' fragments.

    ``HassMediaSearchAndPlay`` wants free text, which Jev cannot produce, so we
    offer the fragments and let it choose. This is the least robust extraction
    here; the router falls back to hassil when the pick is not confident.
    """
    parts = [part.strip(" ,.?!") for part in _CHUNK_SPLIT_RE.split(utterance)]
    chunks = [p for p in parts if len(p) >= 3]
    # Keep them unique and short enough to be a sensible search query.
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        key = chunk.lower()
        if key in seen or len(chunk) > 80:
            continue
        seen.add(key)
        result.append(chunk)
    return tuple(result)


def extract(utterance: str, *, want_media: bool, want_color: bool) -> Extraction:
    """Run every pre-pass the current home can make use of."""
    return Extraction(
        values=find_values(utterance),
        colors_mentioned=want_color and mentions_color(utterance),
        colors_named=colors_named(utterance) if want_color else (),
        media_chunks=media_chunks(utterance) if want_media else (),
    )


__all__ = [
    "COLOR_NAMES",
    "COLOR_TEMP_PRESETS",
    "Extraction",
    "ValueCandidate",
    "colors_named",
    "extract",
    "find_values",
    "media_chunks",
    "mentions_color",
]
