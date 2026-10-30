"""Canary questions: questions with known answers, asked of the live model.

An answer is graded by its meaning, not its wording, because a served model
does not give the same words twice even at temperature 0: the number in the
answer, the name it gives, or the value of a JSON key. Arabic is normalised
first (diacritics dropped, alef forms unified, Arabic-Indic digits read as
digits), so a correct Arabic answer is never marked wrong for its spelling.
"""

import json
import re
from pathlib import Path

from sanad.platform import catalog

BANK = json.loads((Path(__file__).resolve().parent.parent / "data" / "canaries.json").read_text())

DIACRITICS = re.compile("[ً-ْٰ]")
ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"})
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalise(text):
    text = DIACRITICS.sub("", str(text)).translate(ALEF).translate(DIGITS)
    return re.sub(r"\s+", " ", text).strip().lower()


def numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", normalise(text))


def grade(canary, reply):
    """True if the reply carries the canary's answer."""
    if canary["check"] == "number":
        return canary["answer"] in numbers(reply)
    if canary["check"] == "json":
        try:
            value = json.loads(reply)[canary["key"]]
        except (ValueError, KeyError, TypeError):
            return False
        return normalise(value) == normalise(canary["answer"])
    return normalise(canary["answer"]) in normalise(reply)


def wrong_chance(canary, defects):
    """The chance a model with these defects answers this canary wrongly.
    Independent defects compound: each one has its own chance to spoil it."""
    right = 1 - catalog.BASELINE_WRONG
    for defect in defects:
        right *= 1 - catalog.rate(defect, canary["lang"], canary["category"])
    return 1 - right


def garble(text):
    """What a tokenizer that does not match its checkpoint makes of Arabic:
    UTF-8 bytes read back one byte at a time, the familiar mojibake."""
    return text.encode("utf-8").decode("latin-1")


def wrong_reply(canary, defects, draw):
    """A believable wrong answer for the defect most likely to blame."""
    if "tokenizer_mismatch" in defects and canary["lang"] == "ar":
        return garble(canary["reply"])
    if "format_break" in defects and canary["check"] == "json":
        return canary["wrong"][0]  # prose where JSON was asked for
    return canary["wrong"][int(draw * len(canary["wrong"])) % len(canary["wrong"])]


def answer(canary, defects, rng):
    """The model's reply to one canary, given the defects on its path."""
    if rng.random() < wrong_chance(canary, defects):
        return wrong_reply(canary, defects, rng.random())
    return canary["reply"]


def select(lang=None, category=None):
    return [c for c in BANK if (lang is None or c["lang"] == lang)
            and (category is None or c["category"] == category)]
