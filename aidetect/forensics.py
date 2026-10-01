"""Layer 0 - Unicode and typographic forensics.

High precision, low recall. These signals fire rarely, but when they do they
are close to proof that text was pasted out of a web chat UI rather than typed.

The most valuable thing here is the *mixed-signal rule*: curly quotes mixed with
straight quotes in one document means a human pasted an LLM paragraph into their
own writing. That mixture is stronger evidence than either state alone, and it
localises which paragraph came from the model.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from .signals import Signal, saturating
from .text import Document

# Invisible characters that should essentially never be typed by a person.
INVISIBLES = {
    "​": "ZERO WIDTH SPACE",
    "‌": "ZERO WIDTH NON-JOINER",
    "‍": "ZERO WIDTH JOINER",
    "⁠": "WORD JOINER",
    "﻿": "BYTE ORDER MARK",
    "­": "SOFT HYPHEN",
}

# Spaces that are not U+0020. Web-UI rendering artifacts.
EXOTIC_SPACES = {
    " ": "NO-BREAK SPACE",
    " ": "NARROW NO-BREAK SPACE",
    " ": "THIN SPACE",
    " ": "EN SPACE",
    " ": "EM SPACE",
}

SMART_DOUBLE = "“”"
SMART_SINGLE = "‘’"
UNICODE_PROSE_SYMBOLS = "•→←↔≈×÷≥≤…"


def analyse(doc: Document) -> List[Signal]:
    out: List[Signal] = []
    raw = doc.raw
    n_words = max(doc.word_count, 1)

    out.append(_invisible_chars(raw))
    out.append(_exotic_spaces(raw))
    out.append(_em_dash(raw, n_words))
    out.append(_smart_quotes(raw))
    out.append(_unicode_symbols(raw, n_words))
    out.append(_markdown_leakage(doc))
    out.append(_typo_absence(doc))
    return [s for s in out if s is not None]


def _invisible_chars(raw: str) -> Signal:
    found: List[str] = []
    total = 0
    for ch, name in INVISIBLES.items():
        # A BOM at position 0 is a normal file artifact, not evidence.
        hits = raw.count(ch)
        if ch == "﻿" and raw.startswith("﻿"):
            hits -= 1
        if hits > 0:
            total += hits
            found.append(f"{name} x{hits}")
    return Signal(
        name="forensics.invisible_chars",
        layer="forensics",
        logodds=saturating(total, 1.0, 1.8) if total else 0.0,
        evidence=found,
        detail=float(total),
    )


def _exotic_spaces(raw: str) -> Signal:
    found: List[str] = []
    total = 0
    narrow_nbsp = 0
    for ch, name in EXOTIC_SPACES.items():
        hits = raw.count(ch)
        if hits:
            total += hits
            found.append(f"{name} x{hits}")
            if ch == " ":
                narrow_nbsp = hits
    # Narrow NBSP is a specific ChatGPT web-UI artifact, weighted harder.
    lo = saturating(total, 2.0, 1.2) + (0.8 if narrow_nbsp else 0.0)
    return Signal(
        name="forensics.exotic_spaces",
        layer="forensics",
        logodds=lo if total else 0.0,
        evidence=found,
        detail=float(total),
    )


def _em_dash(raw: str, n_words: int) -> Signal:
    em = raw.count("—")
    en = raw.count("–")
    if em + en == 0:
        return Signal("forensics.em_dash", "forensics", 0.0, [], 0.0)
    per_1k = (em + en) * 1000.0 / n_words
    ev = []
    if em:
        ev.append(f"EM DASH x{em}")
    if en:
        ev.append(f"EN DASH x{en}")
    ev.append(f"{per_1k:.1f} per 1k words")
    # Assistants emit em dashes constantly, but this is the single most
    # over-trusted signal in popular "AI detectors" and it is the main driver
    # of false positives on human writers who simply like the punctuation mark.
    # Deliberately weighted low: it needs a high *rate* to contribute much, and
    # it can never convict on its own.
    return Signal(
        name="forensics.em_dash",
        layer="forensics",
        logodds=saturating(per_1k, 8.0, 0.85),
        evidence=ev,
        detail=per_1k,
    )


def _smart_quotes(raw: str) -> Signal:
    smart_d = sum(raw.count(c) for c in SMART_DOUBLE)
    smart_s = sum(raw.count(c) for c in SMART_SINGLE)
    straight_d = raw.count('"')
    # Apostrophes dominate the straight-single count; only compare doubles,
    # plus single quotes used as apostrophes.
    straight_s = raw.count("'")
    smart = smart_d + smart_s
    straight = straight_d + straight_s

    if smart == 0:
        return Signal("forensics.smart_quotes", "forensics", 0.0, [], 0.0)

    ev = [f"curly x{smart}, straight x{straight}"]
    if straight > 0 and smart > 0:
        # Mixed-signal rule: a human pasted machine text into their own.
        ratio = min(smart, straight) / max(smart, straight)
        lo = 0.5 + 1.0 * ratio
        ev.append("MIXED curly+straight: likely paste of machine text into human text")
    else:
        lo = saturating(smart, 4.0, 0.9)
        ev.append("uniformly curly: rich-text pipeline or LLM output")
    return Signal("forensics.smart_quotes", "forensics", lo, ev, float(smart))


def _unicode_symbols(raw: str, n_words: int) -> Signal:
    found = {}
    for ch in UNICODE_PROSE_SYMBOLS:
        c = raw.count(ch)
        if c:
            found[ch] = c
    if not found:
        return Signal("forensics.unicode_symbols", "forensics", 0.0, [], 0.0)
    total = sum(found.values())
    ev = [f"{unicodedata.name(ch, repr(ch))} x{c}" for ch, c in found.items()]
    return Signal(
        name="forensics.unicode_symbols",
        layer="forensics",
        logodds=saturating(total * 1000.0 / max(n_words, 1), 4.0, 0.8),
        evidence=ev[:4],
        detail=float(total),
    )


_MD_PATTERNS = [
    (re.compile(r"\*\*[^*\n]{2,60}\*\*"), "**bold**"),
    (re.compile(r"^#{1,4}\s+\S", re.MULTILINE), "# heading"),
    (re.compile(r"^\s*[-*]\s+\S", re.MULTILINE), "- bullet"),
    (re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE), "1. numbered"),
    (re.compile(r"^---+$", re.MULTILINE), "--- rule"),
]


def _markdown_leakage(doc: Document) -> Signal:
    counts = {}
    for rx, label in _MD_PATTERNS:
        c = len(rx.findall(doc.raw))
        if c:
            counts[label] = c
    if not counts:
        return Signal("forensics.markdown_density", "forensics", 0.0, [], 0.0)

    total = sum(counts.values())
    per_100w = total * 100.0 / max(doc.word_count, 1)
    ev = [f"{k} x{v}" for k, v in counts.items()]
    # Bold + headings + bullets all at once in a short text is the classic
    # chat-response shape.
    variety_bonus = 0.45 if len(counts) >= 3 else 0.0
    return Signal(
        name="forensics.markdown_density",
        layer="forensics",
        logodds=saturating(per_100w, 6.0, 1.0) + variety_bonus,
        evidence=ev,
        detail=per_100w,
    )


_HUMAN_IRREGULARITIES = [
    (re.compile(r"[a-z]{2,}  +[a-z]", re.IGNORECASE), "double space"),
    (re.compile(r"\b(?:teh|recieve|seperate|definately|occured|wierd|alot)\b", re.I), "typo"),
    (re.compile(r"\b(?:dont|cant|wont|didnt|isnt|wasnt|couldnt|thats|im|ive)\b"), "missing apostrophe"),
    # its/it's and your/you're confusion: a characteristic human slip that
    # models essentially never make. Found while reviewing a human-written
    # spec whose only flagged irregularity was trailing whitespace.
    (re.compile(r"\bit's\s+(?:own|applied|value|status|name|version|state|"
                r"contents|size|length|type|id|key)\b", re.I), "it's/its confusion"),
    (re.compile(r"\b(?:your\s+(?:welcome|right|wrong)|you're\s+(?:own|code|file))\b",
                re.I), "your/you're confusion"),
    (re.compile(r"(?<![.!?])\b i \b"), "lowercase standalone 'i'"),
    (re.compile(r"[!?]{2,}"), "repeated punctuation"),
    (re.compile(r"\.{4,}"), "long ellipsis"),
    # [ \t]+ not \s+ : \s+$ in MULTILINE also matches the newline of a blank
    # line, so every paragraph break was being counted as a human irregularity.
    (re.compile(r"[ \t]+$", re.MULTILINE), "trailing whitespace"),
]


def _typo_absence(doc: Document) -> Signal:
    """Zero irregularity across a long informal text is itself anomalous."""
    if doc.word_count < 300:
        # Too short for absence to mean anything. Plenty of people write 200
        # clean words; almost nobody writes 1000.
        return Signal("forensics.typo_absence", "forensics", 0.0, [], 0.0)

    found = []
    total = 0
    for rx, label in _HUMAN_IRREGULARITIES:
        c = len(rx.findall(doc.raw))
        if c:
            total += c
            found.append(f"{label} x{c}")

    if total == 0:
        scale = min((doc.word_count - 300) / 900.0, 1.0)
        return Signal(
            name="forensics.typo_absence",
            layer="forensics",
            logodds=0.20 + 0.70 * scale,
            evidence=[f"zero typographic irregularities across {doc.word_count} words"],
            detail=0.0,
        )

    rate = total * 1000.0 / doc.word_count
    return Signal(
        name="forensics.typo_absence",
        layer="forensics",
        logodds=-saturating(rate, 4.0, 1.1),
        evidence=found[:4],
        detail=rate,
    )
