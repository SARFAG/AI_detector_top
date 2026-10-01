"""Layer 3 - document shape.

Low precision individually, but high recall, and the signals are cheap. The
"answer-shaped document" (restate question -> headed sections -> mandatory
conclusion) is one of the most reliable gestalt tells once a text is long
enough to have a shape at all.
"""

from __future__ import annotations

import re
from typing import List

from .signals import Signal, saturating
from .text import Document, tokenize_words

_CLOSERS = re.compile(
    r"\b(?:in conclusion|to conclude|in summary|to sum up|to summari[sz]e|"
    r"overall,|ultimately,|the key takeaway|in closing|final thoughts|"
    r"wrapping up|all in all)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S|\*\*[^*\n]{3,60}\*\*\s*$)", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\S", re.MULTILINE)
_QUESTION_RESTATE = re.compile(
    r"^\s*(?:great question|that's a great|sure[,!]|certainly[,!]|of course[.,!]|"
    r"happy to help|let's (?:take a look|explore|dive|break)|"
    r"(?:when|what|how|why) it comes to)",
    re.IGNORECASE,
)


def analyse(doc: Document) -> List[Signal]:
    return [
        _mandatory_conclusion(doc),
        _section_density(doc),
        _answer_shape(doc),
        _balanced_sections(doc),
        _identifier_dispersion(doc),
    ]


def _mandatory_conclusion(doc: Document) -> Signal:
    """A synthesising close that nothing in the text required."""
    if doc.word_count < 120:
        return Signal("structural.mandatory_conclusion", "structural", 0.0, [], 0.0)

    hits = _CLOSERS.findall(doc.prose)
    if not hits:
        return Signal("structural.mandatory_conclusion", "structural", 0.0, [], 0.0)

    # Worth more when it appears in the final fifth of the document.
    tail_start = int(len(doc.prose) * 0.8)
    in_tail = bool(_CLOSERS.search(doc.prose[tail_start:]))
    lo = saturating(len(hits), 1.0, 0.85) + (0.45 if in_tail else 0.0)
    return Signal(
        "structural.mandatory_conclusion", "structural", lo,
        [f"'{h}'" for h in hits[:3]] + (["in final 20% of document"] if in_tail else []),
        float(len(hits)),
    )


def _section_density(doc: Document) -> Signal:
    if doc.word_count < 100:
        return Signal("structural.section_density", "structural", 0.0, [], 0.0)
    headings = len(_HEADING.findall(doc.raw))
    bullets = len(_BULLET.findall(doc.raw))
    per_100w = (headings * 3 + bullets) * 100.0 / doc.word_count
    if per_100w < 2:
        return Signal("structural.section_density", "structural", 0.0,
                      [f"{headings} headings, {bullets} bullets"], per_100w)
    return Signal(
        "structural.section_density", "structural",
        saturating(per_100w, 10.0, 0.9),
        [f"{headings} headings, {bullets} bullets ({per_100w:.1f} weighted per 100 words)"],
        per_100w,
    )


def _answer_shape(doc: Document) -> Signal:
    """Opens by restating/acknowledging the question, closes by synthesising."""
    if doc.word_count < 80:
        return Signal("structural.answer_shape", "structural", 0.0, [], 0.0)
    head = doc.prose[:200]
    opens = bool(_QUESTION_RESTATE.search(head))
    closes = bool(_CLOSERS.search(doc.prose[int(len(doc.prose) * 0.75):]))
    has_sections = len(_HEADING.findall(doc.raw)) >= 2

    score = sum([opens, closes, has_sections])
    if score < 2:
        return Signal("structural.answer_shape", "structural", 0.0, [], float(score))
    ev = []
    if opens:
        ev.append("opens by acknowledging/restating the prompt")
    if has_sections:
        ev.append("segmented into headed sections")
    if closes:
        ev.append("closes with a synthesis")
    return Signal("structural.answer_shape", "structural",
                  0.55 if score == 2 else 1.15, ev, float(score))


def _balanced_sections(doc: Document) -> Signal:
    """Equal space per sub-topic. Humans are lopsided - they overspend on the
    part they actually care about."""
    blocks = re.split(r"^\s*#{1,6}\s+.*$", doc.raw, flags=re.MULTILINE)
    blocks = [b for b in blocks if len(tokenize_words(b)) >= 25]
    if len(blocks) < 3:
        return Signal("structural.section_balance", "structural", 0.0, [], 0.0)

    lens = [len(tokenize_words(b)) for b in blocks]
    m = sum(lens) / len(lens)
    sd = (sum((x - m) ** 2 for x in lens) / (len(lens) - 1)) ** 0.5
    cv = sd / m if m else 0.0
    if cv < 0.35:
        return Signal("structural.section_balance", "structural",
                      min((0.35 - cv) * 2.6, 0.7),
                      [f"section lengths unusually even (CV={cv:.2f}, n={len(blocks)})"], cv)
    return Signal("structural.section_balance", "structural", 0.0,
                  [f"section length CV={cv:.2f}"], cv)


# Code identifiers: CamelCase and snake_case.
_IDENT = re.compile(r"\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z_]+\b")

# Below this identifier rate the document is not technical enough for the
# measure to mean anything.
_TECHNICAL_FLOOR = 15.0


def _identifier_dispersion(doc: Document) -> Signal:
    """Are technical identifiers spread through the document or clustered?

    Found by diffing a matched pair: the same feature request written once by
    a model and once by a person, sharing 35 of 36 identifiers. The content was
    the same; the placement was not.

    The person wrote six paragraphs of plain prose about the problem, then put
    every API name in one bolt-on section at the end, introduced as "to make
    this concrete, the names I'd go with". The model interleaved names through
    nearly every paragraph. People separate *what I want* from *what to call
    it*; models treat naming as part of each requirement.

    Measured as the fraction of substantial paragraphs containing at least one
    identifier:

        human technical prose     0.25, 0.29
        machine technical prose   0.88, 1.00, 1.00, 1.00, 1.00, 1.00

    Only two human documents support the low end, so the weight is kept
    modest despite the clean gap. Applies only to technical text; prose with
    no identifiers is skipped entirely.
    """
    paras = [p for p in doc.paragraphs if len(tokenize_words(p)) >= 20]
    if len(paras) < 4:
        return Signal("structural.identifier_dispersion", "structural", 0.0, [], 0.0)

    ident_rate = len(_IDENT.findall(doc.prose)) * 1000.0 / max(doc.word_count, 1)
    if ident_rate < _TECHNICAL_FLOOR:
        return Signal("structural.identifier_dispersion", "structural", 0.0,
                      [f"not technical enough to measure ({ident_rate:.0f} "
                       f"identifiers per 1k)"], 0.0)

    with_ids = sum(1 for p in paras if _IDENT.search(p))
    dispersion = with_ids / len(paras)

    ev = [f"{with_ids}/{len(paras)} paragraphs carry identifiers "
          f"(dispersion {dispersion:.2f}, {ident_rate:.0f} per 1k)"]
    if dispersion >= 0.85:
        lo = min((dispersion - 0.85) * 2.0 + 0.45, 0.6)
        ev.append("identifiers interleaved throughout")
    elif dispersion <= 0.50:
        lo = -min((0.50 - dispersion) * 1.6 + 0.30, 0.6)
        ev.append("identifiers clustered into a few sections")
    else:
        lo = 0.0
    return Signal("structural.identifier_dispersion", "structural", lo, ev,
                  dispersion)
