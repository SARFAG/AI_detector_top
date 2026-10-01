"""Layer 5 - prompt authorship (research section 7).

Detecting machine-authored *prompts* is a different problem from detecting
machine-authored prose: prompts are imperative, role-framed and format-
constrained by nature. This scores how much a text looks like an engineered
prompt rather than a human's off-the-cuff request.

It doubles as a lightweight prompt-injection surface scan, since injection
payloads share the imperative-stacking shape.
"""

from __future__ import annotations

import re
from typing import List

from .lexicon import PROMPT_MARKERS
from .signals import Signal, saturating
from .text import Document

_RX_PROMPT = re.compile(
    "(?:" + "|".join(re.escape(p) for p in PROMPT_MARKERS) + ")", re.IGNORECASE
)
_RX_CAPS_NEGATION = re.compile(r"\b(?:DO NOT|NEVER|MUST NOT|ALWAYS|IMPORTANT|CRITICAL)\b")
_RX_DELIMITER = re.compile(r"^(?:#{3,}|---+|===+|<\/?[a-z_]+>)\s*$", re.MULTILINE)
_RX_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)

# Imperative-stacking phrases typical of injected instructions.
_RX_INJECTION = re.compile(
    r"(?:ignore (?:all |any )?(?:previous|prior|above) instructions|"
    r"disregard (?:the )?(?:above|previous)|"
    r"you are now|new instructions:|system prompt:|"
    r"reveal your (?:system )?prompt|print your instructions)",
    re.IGNORECASE,
)


def analyse(doc: Document) -> List[Signal]:
    return [_prompt_shape(doc), _injection_surface(doc)]


def _prompt_shape(doc: Document) -> Signal:
    markers = _RX_PROMPT.findall(doc.prose)
    caps = _RX_CAPS_NEGATION.findall(doc.raw)
    delims = _RX_DELIMITER.findall(doc.raw)
    numbered = _RX_NUMBERED.findall(doc.raw)

    score = len(markers) + 0.5 * len(caps) + 0.7 * len(delims) + 0.3 * len(numbered)
    if score < 1.5:
        return Signal("prompt.engineered_shape", "prompt", 0.0, [], score)

    ev: List[str] = []
    if markers:
        uniq = sorted({m.lower() for m in markers})
        ev.append("markers: " + ", ".join(f"'{u}'" for u in uniq[:4]))
    if caps:
        ev.append(f"shouted constraints x{len(caps)}")
    if delims:
        ev.append(f"delimiter blocks x{len(delims)}")
    if numbered:
        ev.append(f"numbered constraints x{len(numbered)}")

    return Signal("prompt.engineered_shape", "prompt",
                  saturating(score, 4.0, 1.6), ev, score)


def _injection_surface(doc: Document) -> Signal:
    hits = _RX_INJECTION.findall(doc.raw)
    if not hits:
        return Signal("prompt.injection_surface", "prompt", 0.0, [], 0.0)
    return Signal(
        "prompt.injection_surface", "prompt", 0.0,  # advisory only, not AI evidence
        [f"POSSIBLE PROMPT INJECTION: '{h}'" for h in hits[:3]],
        float(len(hits)),
    )
