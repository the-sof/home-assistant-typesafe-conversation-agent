"""Build the one set of questions we send Jev per utterance.

Everything here goes in a *single* API call. Questions are evaluated in
parallel and are independent of one another, so asking a question we will
probably ignore costs only that question's tokens - measured at ~97 tokens
each against this API - and almost no extra latency. That is the speculative
fan-out pattern: ask every question any branch of the router might need, then
let code decide which answers to read.

Question IDs never reach the model, so every instruction states its question in
full.
"""

from __future__ import annotations

from typing import Any

from .const import LOGGER, MAX_CHOICE_OPTIONS
from .entities import CatalogArea, CatalogEntity
from .extraction import COLOR_NAMES, COLOR_TEMP_PRESETS, Extraction

Question = dict[str, Any]
QuestionSet = dict[str, Question]

# --- question ids ------------------------------------------------------------
Q_CATEGORY = "category"
Q_COMPOUND = "compound"
Q_SCOPE = "scope"
Q_HERE_RELATIVE = "here_relative"
Q_CHANGE_DIRECTION = "change_direction"
Q_MAGNITUDE = "magnitude"
Q_QUERY_KIND = "query_kind"
Q_RISKY = "risky"
Q_TARGET_AREA = "target_area"
Q_TARGET_ENTITY = "target_entity"
Q_TARGET_DOMAIN = "target_domain"
Q_VALUE_PICK = "value_pick"
Q_COLOR_PICK = "color_pick"
Q_MEDIA_SPAN = "media_query_span"
Q_LIST_ITEM_SPAN = "list_item_span"

NO_AREA = "no_area"
NO_SINGLE_ENTITY = "no_single_entity"
NO_DOMAIN = "none"
NOT_TARGETED = "not_targeted"
NO_VALUE = "none"


def action_question_id(domain: str) -> str:
    return f"action_{domain}"


# --- static questions --------------------------------------------------------

STATIC_QUESTIONS: QuestionSet = {
    Q_CATEGORY: {
        "type": "choice",
        "instructions": (
            "The user said `request.text` to the voice assistant of their smart "
            "home. What kind of request is it?"
        ),
        "criteria": {
            "command": (
                "The user wants something in the home - a light, lock, "
                "appliance, blind, speaker, thermostat, scene or routine - to "
                "change state or do something. This includes indirect phrasing "
                "that implies an action without naming one, such as 'it's too "
                "bright in here' or 'get the coffee going'."
            ),
            "query": (
                "The user is asking about the current state of the home: "
                "whether something is on or off, open or closed, locked or "
                "unlocked, what a sensor or thermostat reads, how many things "
                "are in some state, or what the time or date is."
            ),
            "information": (
                "The user wants general knowledge, conversation, an opinion, a "
                "calculation, a recipe, a joke, or anything else not about the "
                "state or control of this home."
            ),
            "cancel": (
                "The user is withdrawing, stopping or dismissing the "
                "assistant: 'never mind', 'forget it', 'cancel that', 'stop'."
            ),
            "unclear": (
                "The request is empty, garbled, an unfinished fragment, or "
                "gives no usable indication of what the user wants."
            ),
        },
    },
    Q_COMPOUND: {
        "type": "noul",
        "instructions": (
            "Does `request.text` ask for two or more distinct actions that "
            "would each have to be carried out separately?"
        ),
        "criteria": {
            "true": (
                "Two or more separate things are requested, such as 'turn off "
                "the lights and lock the door', or 'dim the lamp then start "
                "the coffee maker'. Applying ONE action to several devices is "
                "not two actions."
            ),
            "false": (
                "One action, however many devices it applies to. 'Turn off all "
                "the lights in the house' is a single action."
            ),
        },
    },
    Q_SCOPE: {
        "type": "choice",
        "instructions": (
            "In `request.text`, how wide is the set of things being targeted?"
        ),
        "criteria": {
            "whole_house": (
                "Everything of some kind, everywhere in the home: 'all the "
                "lights', 'turn everything off', 'lock up the house'."
            ),
            "floor": (
                "Everything of some kind on one floor or level: 'the lights "
                "upstairs', 'close the blinds downstairs'."
            ),
            "area": (
                "Everything of some kind in one room or area: 'the kitchen "
                "lights', 'turn off the bedroom'."
            ),
            "single": (
                "One specific device, either named outright or made obvious by "
                "the context."
            ),
            "not_a_target": "Nothing in the home is being targeted at all.",
        },
    },
    Q_HERE_RELATIVE: {
        "type": "noul",
        "instructions": (
            "Does `request.text` depend on where the speaker is standing rather "
            "than naming a room? The speaker is in the area named by "
            "`request.spoken_from_area`."
        ),
        "criteria": {
            "true": (
                "Uses 'here', 'in this room', 'in here', or refers to 'the "
                "lights' or 'the fan' with no room named, so only the "
                "speaker's own room makes sense."
            ),
            "false": (
                "Names a room, a floor, the whole house, or one specific "
                "device unambiguously."
            ),
        },
    },
    Q_CHANGE_DIRECTION: {
        "type": "choice",
        "instructions": (
            "If `request.text` changes a numeric setting - brightness, "
            "temperature, volume, openness, speed or humidity - does it name a "
            "specific target value, or only a direction to move in?"
        ),
        "criteria": {
            "absolute": (
                "A specific value is named or clearly implied: 'to 30 "
                "percent', 'to 21 degrees', 'halfway', 'all the way open'."
            ),
            "increase": (
                "Only more, with no value: brighter, warmer, louder, faster, "
                "open it more, turn it up."
            ),
            "decrease": (
                "Only less, with no value: dimmer, cooler, quieter, slower, "
                "close it a bit, turn it down."
            ),
            "no_numeric": "No numeric setting is involved in the request.",
        },
    },
    Q_MAGNITUDE: {
        "type": "choice",
        "instructions": (
            "If `request.text` asks for a relative change without naming a "
            "value, how large a change does the user mean?"
        ),
        "criteria": {
            "slight": "'a bit', 'a touch', 'slightly', 'a little', 'a shade'.",
            "moderate": (
                "A plain comparative with no qualifier: 'brighter', 'warmer', "
                "'turn it up', 'open it more'."
            ),
            "large": "'much', 'a lot', 'way', 'considerably', 'significantly'.",
            "maximum": "'all the way', 'as far as it goes', 'full', 'max'.",
            "not_applicable": (
                "No relative change is being asked for, either because a value "
                "was named or because nothing numeric is involved."
            ),
        },
    },
    Q_QUERY_KIND: {
        "type": "choice",
        "instructions": (
            "If `request.text` is a question about the home, what kind of "
            "question is it?"
        ),
        "criteria": {
            "device_state": (
                "Whether one named device is on or off, open or closed, locked "
                "or unlocked, or what value it currently reads."
            ),
            "count": (
                "How many things are in some state, or whether any are: 'are "
                "any windows open', 'how many lights are on'."
            ),
            "temperature": "The temperature of a room, or of the home.",
            "time_or_date": "The current time, or today's date.",
            "needs_prose": (
                "A question about the home that needs a written explanation or "
                "a summary across several devices, rather than one state "
                "reading: 'is everything locked up', 'what's going on "
                "downstairs'."
            ),
            "not_a_query": "The request is not a question about the home.",
        },
    },
    Q_RISKY: {
        "type": "noul",
        "instructions": (
            "If it were carried out, would `request.text` reduce the physical "
            "security of the home, or be hard to undo?"
        ),
        "criteria": {
            "true": (
                "Unlocking a door, opening a garage door, gate, exterior door "
                "or window, disarming an alarm, or anything else that leaves "
                "the home accessible from outside."
            ),
            "false": (
                "Lights, media, climate, scenes, routines, appliances, and "
                "anything that locks, closes or arms."
            ),
        },
    },
}


# --- speculative per-domain action questions ---------------------------------
# Every one of these is asked on every request. The router reads only the one
# matching the domain it settled on and discards the rest unread. A near-
# coinflip on an unread branch is expected: the model is being asked about
# something the user never mentioned.

_ACTION_LABELS: dict[str, str] = {
    "light": "lights",
    "switch": "switches and plug-in appliances",
    "cover": "blinds, shades, curtains, doors, gates and windows that open",
    "lock": "door locks",
    "fan": "fans",
    "climate": "thermostats and air conditioning",
    "media_player": "speakers and TVs",
    "scene": "scenes",
    "script": "routines",
    "vacuum": "robot vacuums",
    "humidifier": "humidifiers",
    "water_heater": "water heaters",
    "button": "buttons",
    "input_boolean": "user-defined toggles",
    "todo": "to-do and shopping lists",
}

_ACTION_CRITERIA: dict[str, dict[str, str]] = {
    "light": {
        "turn_on": "Switch them on.",
        "turn_off": "Switch them off.",
        "toggle": "Flip whatever state they are in.",
        "set_brightness": "Set them to a specific brightness level.",
        "brighter": "Increase brightness, with no specific level named.",
        "dimmer": "Reduce brightness, with no specific level named.",
        "set_color": "Change their colour, or how warm or cool the light is.",
    },
    "switch": {
        "turn_on": "Switch it on.",
        "turn_off": "Switch it off.",
        "toggle": "Flip whatever state it is in.",
    },
    "cover": {
        "open": "Open, raise, lift or draw them back.",
        "close": "Close, lower, shut or draw them across.",
        "stop": "Halt them part-way through moving.",
        "set_position": "Move them to a specific degree of openness.",
    },
    "lock": {
        "lock": "Lock them.",
        "unlock": "Unlock them.",
    },
    "fan": {
        "turn_on": "Switch them on.",
        "turn_off": "Switch them off.",
        "set_speed": "Set them to a specific speed.",
        "faster": "Increase the speed, with no specific speed named.",
        "slower": "Reduce the speed, with no specific speed named.",
    },
    "climate": {
        "turn_on": "Switch them on.",
        "turn_off": "Switch them off.",
        "set_temperature": "Set them to a specific target temperature.",
        "warmer": "Raise the temperature, with no specific value named.",
        "cooler": "Lower the temperature, with no specific value named.",
        # No "set_mode": HassClimateSetTemperature is the only climate intent
        # Home Assistant registers, so heat/cool/auto/dry cannot be reached.
    },
    "media_player": {
        "play": "Resume or start playback.",
        # Home Assistant has no stop-media intent, so "stop the music" and
        # "pause the music" can only resolve to the same action. Offering both
        # as separate options would split the probability between two synonyms
        # and risk failing the margin gate on a perfectly clear request.
        "pause": "Pause, stop or halt playback.",
        "next": "Skip to the next track or item.",
        "previous": "Go back to the previous track or item.",
        "set_volume": "Set the volume to a specific level.",
        "louder": "Raise the volume, with no specific level named.",
        "quieter": "Lower the volume, with no specific level named.",
        "mute": "Silence them without stopping playback.",
        "unmute": "Restore the sound.",
        "search_and_play": (
            "Find and play specific content the user named - a song, artist, "
            "album, station, show or genre."
        ),
    },
    "scene": {"activate": "Apply the scene."},
    "script": {"run": "Run the routine."},
    "vacuum": {
        "start": "Start cleaning.",
        "return_to_base": "Send it back to its dock.",
        "clean_area": "Clean one specific room or area.",
    },
    "humidifier": {
        "turn_on": "Switch them on.",
        "turn_off": "Switch them off.",
        "set_humidity": "Set them to a specific target humidity.",
        "set_mode": "Switch between their operating modes.",
    },
    "water_heater": {
        "turn_on": "Switch it on.",
        "turn_off": "Switch it off.",
        # No "set_temperature": water_heater registers no intents at all, and
        # HassClimateSetTemperature declares platforms={climate}, so it refuses
        # the domain. Power is all that can be reached from here.
    },
    "button": {"press": "Press it."},
    "input_boolean": {
        "turn_on": "Switch it on.",
        "turn_off": "Switch it off.",
        "toggle": "Flip whatever state it is in.",
    },
    "todo": {
        "add_item": "Put something new on the list.",
        "complete_item": "Mark something on the list as done or bought.",
        "remove_item": "Take something off the list.",
    },
}

_DOMAIN_CRITERIA: dict[str, str] = {
    "light": "Lights, lamps, bulbs, brightness or light colour.",
    "switch": (
        "Switches and appliances on a plug: a coffee maker, kettle, heater, "
        "dishwasher or similar."
    ),
    "cover": (
        "Blinds, shades, curtains, shutters or awnings, and anything that opens "
        "and closes: a garage door, a gate, a powered door or window."
    ),
    "lock": "Door locks.",
    "fan": "Fans and their speed.",
    "climate": "Thermostats, heating, cooling or air conditioning.",
    "media_player": "Speakers, TVs, music, volume or playback.",
    "scene": "Named scenes, moods or presets.",
    "script": "Named routines the user has set up.",
    "vacuum": "Robot vacuums.",
    "humidifier": "Humidifiers and dehumidifiers.",
    "water_heater": "Water heaters and boilers.",
    "button": "Buttons that are pressed to trigger something.",
    "input_boolean": "User-defined on/off toggles and flags.",
    "sensor": "Sensor readings.",
    "binary_sensor": "Sensors that report an open/closed or on/off condition.",
    "number": "Numeric settings.",
    "select": "Multi-option settings.",
    "alarm_control_panel": "Alarm systems.",
    "valve": "Valves.",
    "lawn_mower": "Robot lawn mowers.",
    "todo": "To-do and shopping lists.",
    "calendar": "Calendars.",
}


def _action_question(domain: str) -> Question:
    label = _ACTION_LABELS.get(domain, f"{domain} entities")
    criteria = dict(_ACTION_CRITERIA.get(domain, {}))
    criteria[NOT_TARGETED] = f"The request is clearly not about {label} at all."
    return {
        "type": "choice",
        "instructions": (
            f"Assume for this question that `request.text` is a command aimed "
            f"at the {label} in this home. What should happen to them? Answer "
            f"'{NOT_TARGETED}' if the request plainly has nothing to do with "
            f"{label}."
        ),
        "criteria": criteria,
    }


# --- catalog-derived questions -----------------------------------------------


def _target_area_question(areas: tuple[CatalogArea, ...]) -> Question:
    criteria: dict[str, Any] = {area.area_id: None for area in areas}
    criteria[NO_AREA] = "No single area of the home is being targeted."
    return {
        "type": "choice",
        "instructions": (
            "Which area of the home does `request.text` target? The areas are "
            "listed in `home.areas`; every option below is the `id` of one "
            "entry in that list, so look it up there for its name and floor. "
            "The speaker is standing in `request.spoken_from_area` - choose "
            f"that area when the request says 'here' or names no room. Choose "
            f"'{NO_AREA}' when the request names one specific device with no "
            "room, targets the whole house, or targets nothing at all."
        ),
        "criteria": criteria,
    }


def target_entity_question(
    entities: tuple[CatalogEntity, ...], inline_descriptions: bool
) -> Question:
    if inline_descriptions:
        criteria: dict[str, Any] = {
            e.entity_id: " · ".join(
                part for part in (e.name, e.area_name, e.domain) if part
            )
            for e in entities
        }
    else:
        # Null descriptions: the details are already in `home.entities`, which
        # every question can read. Inlining them again would duplicate ~2k
        # tokens for no new information.
        criteria = {e.entity_id: None for e in entities}
    criteria[NO_SINGLE_ENTITY] = "No one specific entity is being targeted."
    return {
        "type": "choice",
        "instructions": (
            "Which single entity in `home.entities` does `request.text` "
            "target? Every option below is the `id` of one entry in "
            "`home.entities`; look that entry up for its name, other names, "
            "area, kind and current state. Match on what the user means rather "
            "than on exact wording: 'get the coffee boiling' targets a coffee "
            f"maker. Choose '{NO_SINGLE_ENTITY}' when the request targets a "
            "group of things, a whole room, the whole house, or nothing."
        ),
        "criteria": criteria,
    }


def _target_domain_question(domains: tuple[str, ...]) -> Question:
    criteria: dict[str, Any] = {
        domain: _DOMAIN_CRITERIA.get(domain, f"{domain} entities.")
        for domain in domains
    }
    criteria[NO_DOMAIN] = "No kind of device is being targeted."
    return {
        "type": "choice",
        "instructions": (
            "What kind of thing does `request.text` want to act on or ask "
            "about? Only the kinds of thing this home actually has are listed."
        ),
        "criteria": criteria,
    }


# --- conditional questions ---------------------------------------------------


def _value_pick_question(extraction: Extraction) -> Question:
    criteria: dict[str, Any] = {c.span: None for c in extraction.values}
    criteria[NO_VALUE] = (
        "None of these is a setting's value - for example the number is part "
        "of a device's name, or is a room number."
    )
    return {
        "type": "choice",
        "instructions": (
            "`request.text` may name a value for a setting such as brightness, "
            "temperature, volume, openness, speed or humidity. Each option "
            "below is a fragment taken from the request. Which fragment is "
            "that value?"
        ),
        "criteria": criteria,
    }


def _color_pick_question(
    extraction: Extraction | None = None, max_options: int = MAX_CHOICE_OPTIONS
) -> Question:
    presets = 5  # the whites below
    if len(COLOR_NAMES) + presets + 1 <= max_options or extraction is None:
        names: tuple[str, ...] = COLOR_NAMES
    else:
        # A small-cap server cannot take every colour, but it only needs the
        # ones the user said; descriptive requests ("cosy") map to the whites.
        names = extraction.colors_named
    criteria: dict[str, Any] = dict.fromkeys(names)
    criteria.update(
        {
            "warm_white": "A warm, yellowish white, as in 'warmer' or 'cosy'.",
            "soft_white": "A soft, slightly warm white.",
            "neutral_white": "A plain, neutral white.",
            "cool_white": "A cool, slightly blue white.",
            "daylight": "A bright, blue-white daylight colour.",
        }
    )
    # Whatever the cap, keep a slot for NO_VALUE; named colours go first.
    criteria = dict(list(criteria.items())[: max(1, max_options - 1)])
    criteria[NO_VALUE] = "No colour is being requested."
    return {
        "type": "choice",
        "instructions": (
            "`request.text` may ask for a particular light colour. Which of "
            "these is closest to what the user asked for? Map a descriptive "
            "request onto the nearest option: 'cosy' or 'warmer' means "
            "'warm_white', 'like daylight' means 'daylight'."
        ),
        "criteria": criteria,
    }


def _list_item_span_question(extraction: Extraction) -> Question:
    criteria: dict[str, Any] = dict.fromkeys(extraction.media_chunks)
    criteria[NO_VALUE] = "The request does not name an item."
    return {
        "type": "choice",
        "instructions": (
            "If `request.text` asks to add something to a list, or to tick "
            "something off one, which of these fragments of the request names "
            "that item? A fragment naming the list itself, such as 'the "
            "shopping list', is not the item."
        ),
        "criteria": criteria,
    }


def _media_span_question(extraction: Extraction) -> Question:
    criteria: dict[str, Any] = dict.fromkeys(extraction.media_chunks)
    criteria[NO_VALUE] = "The request does not name any content to play."
    return {
        "type": "choice",
        "instructions": (
            "If `request.text` asks to play specific content, which of these "
            "fragments of the request names that content - the song, artist, "
            "album, station, show or genre? A fragment naming a room or a "
            "speaker is not the content."
        ),
        "criteria": criteria,
    }


# --- the builder -------------------------------------------------------------


class QuestionSetError(Exception):
    """The question set we built would be rejected by the API."""


def build_questions(
    *,
    entities: tuple[CatalogEntity, ...],
    areas: tuple[CatalogArea, ...],
    domains: tuple[str, ...],
    extraction: Extraction,
    inline_descriptions: bool = False,
    max_options: int = MAX_CHOICE_OPTIONS,
) -> QuestionSet:
    """Assemble every question for one request.

    The structural part (everything except the conditional questions) depends
    only on the catalog, so callers cache it against the catalog generation.

    ``max_options`` is the server's cap on options per Choice. A home with more
    entities than that cannot be offered in one ``target_entity`` question, so
    it is left out here and asked in a second, narrower request instead (see
    ``hierarchy``). The full catalogue still goes in the state either way: the
    cap limits options, not state.
    """
    questions: QuestionSet = dict(STATIC_QUESTIONS)

    if areas:
        if fits(len(areas) + 1, max_options):
            questions[Q_TARGET_AREA] = _target_area_question(areas)
        else:
            LOGGER.warning(
                "%s areas exceed this server's %s-option cap; not asking for one",
                len(areas),
                max_options,
            )
    if entities and fits(len(entities) + 1, max_options):
        questions[Q_TARGET_ENTITY] = target_entity_question(
            entities, inline_descriptions
        )
    if domains and fits(len(domains) + 1, max_options):
        questions[Q_TARGET_DOMAIN] = _target_domain_question(domains)

    for domain in domains:
        if domain in _ACTION_CRITERIA:
            questions[action_question_id(domain)] = _action_question(domain)

    if extraction.values:
        questions[Q_VALUE_PICK] = _value_pick_question(extraction)
    if extraction.colors_mentioned:
        questions[Q_COLOR_PICK] = _color_pick_question(extraction, max_options)
    if extraction.media_chunks:
        if "media_player" in domains:
            questions[Q_MEDIA_SPAN] = _media_span_question(extraction)
        if "todo" in domains:
            questions[Q_LIST_ITEM_SPAN] = _list_item_span_question(extraction)

    validate_questions(questions, max_options)
    return questions


def fits(options: int, max_options: int) -> bool:
    return options <= max_options


def validate_questions(
    questions: QuestionSet, max_options: int = MAX_CHOICE_OPTIONS
) -> None:
    """Catch anything the server would reject before sending it.

    A rejection in production means this validator has a hole, so it runs on
    every build rather than only in tests.
    """
    if not questions:
        raise QuestionSetError("Question set is empty")

    for key, question in questions.items():
        kind = question.get("type")
        instructions = question.get("instructions")
        if not instructions:
            raise QuestionSetError(f"{key}: instructions are empty")

        if kind == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise QuestionSetError(f"{key}: a Choice needs at least 2 options")
            if not fits(len(criteria), max_options):
                raise QuestionSetError(
                    f"{key}: {len(criteria)} options exceeds this server's cap "
                    f"of {max_options}"
                )
            for option in criteria:
                if not isinstance(option, str) or not option:
                    raise QuestionSetError(f"{key}: option {option!r} is not a name")
        elif kind == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise QuestionSetError(f"{key}: a Score needs 2-10 levels")
        elif kind == "noul":
            criteria = question.get("criteria")
            if criteria is not None and not isinstance(criteria, dict):
                raise QuestionSetError(f"{key}: Noul criteria must be an object")
        else:
            raise QuestionSetError(f"{key}: unknown question type {kind!r}")


def estimate_tokens(payload: Any) -> int:
    """Rough token count. Only used for budget guards and debug logging."""
    import json

    return len(json.dumps(payload, separators=(",", ":"))) // 4


__all__ = [
    "COLOR_TEMP_PRESETS",
    "NOT_TARGETED",
    "NO_AREA",
    "NO_DOMAIN",
    "NO_SINGLE_ENTITY",
    "NO_VALUE",
    "Q_CATEGORY",
    "Q_CHANGE_DIRECTION",
    "Q_COLOR_PICK",
    "Q_COMPOUND",
    "Q_HERE_RELATIVE",
    "Q_LIST_ITEM_SPAN",
    "Q_MAGNITUDE",
    "Q_MEDIA_SPAN",
    "Q_QUERY_KIND",
    "Q_RISKY",
    "Q_SCOPE",
    "Q_TARGET_AREA",
    "Q_TARGET_DOMAIN",
    "Q_TARGET_ENTITY",
    "Q_VALUE_PICK",
    "QuestionSetError",
    "action_question_id",
    "build_questions",
    "estimate_tokens",
    "fits",
    "target_entity_question",
    "validate_questions",
]
