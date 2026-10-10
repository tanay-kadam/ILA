"""Conservative, gold-blind mapping of a boxed payload to option content.

This is a secondary format diagnostic, never a replacement for strict scoring.
"""
from __future__ import annotations
from fractions import Fraction
import re
from .textio import answers_match_choice


def normalize(value: str) -> str:
    value = value.strip().strip("$")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\%", "%").replace(r"\,", " ")
    value = value.replace(r"\!", "").replace(r"\ ", " ")
    value = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[dt]?frac\{([+-]?\d+)\}\{([+-]?\d+)\}", r"\1/\2", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def content_key(value: str):
    value = normalize(value)
    # Exact rational numbers only. Units and percentages remain part of the key.
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+|\d+/[+-]?\d+))\s*(%|[a-z]+(?: [a-z]+)*)?", value)
    if match:
        try:
            return ("number", Fraction(match[1]), match[2] or "")
        except (ValueError, ZeroDivisionError):
            pass
    return ("text", value)


def parse_options(options: list[str]) -> dict[str, str]:
    parsed = {}
    for option in options:
        match = re.fullmatch(r"\s*([A-E])[).:]\s*(.+)\s*", option)
        if not match or match[1] in parsed:
            raise ValueError("invalid or duplicate option label")
        parsed[match[1]] = match[2]
    if set(parsed) != set("ABCDE"):
        raise ValueError("expected exactly five options")
    return parsed


def map_boxed_choice(payload: str | None, options: list[str]) -> dict:
    """No gold argument: map to all options before any correctness comparison."""
    choices = parse_options(options)
    if payload is None or not payload.strip():
        return {"letter": None, "kind": "unparseable"}
    for letter in "ABCDE":
        if answers_match_choice(payload, letter):
            return {"letter": letter, "kind": "strict_letter"}
    value = normalize(payload)
    label = re.fullmatch(r"\(?([a-e])\)?[).:]\s*(.+)", value)
    content = label[2] if label else value
    key = content_key(content)
    matches = [letter for letter, option in choices.items() if content_key(option) == key]
    if label and (len(matches) != 1 or matches[0] != label[1].upper()):
        return {"letter": None, "kind": "ambiguous_label_content", "matches": matches}
    if len(matches) > 1:
        return {"letter": None, "kind": "ambiguous_duplicate_content", "matches": matches}
    if len(matches) == 1:
        return {"letter": matches[0], "kind": "label_content" if label else "unique_content"}
    return {"letter": None, "kind": "unparseable"}
