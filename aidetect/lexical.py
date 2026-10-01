"""Layer 1 - tiered lexical markers, human counter-markers, model attribution.

Each tier accumulates hits, then saturates onto a capped log-odds contribution
(see lexicon.TIER_CAPS). The cap matters: without it a 10,000-word document
would convict itself purely by being long enough to contain stray markers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .lexicon import (
    HUMAN_MARKERS,
    HUMAN_MARKER_PATTERNS,
    MODEL_PROFILES,
    TIER1_WORDS,
    TIER2_WORDS,
    TIER3_PHRASES,
    TIER4_LEAKAGE,
    TIER_CAPS,
    TIER_MIDPOINTS,
    TIER_WEIGHTS,
)
from .signals import Signal, saturating
from .text import Document


def _bounded(marker: str) -> str:
    """Escape a marker and add \b only where it would actually bind.

    Critical: a naive alternation without boundaries makes 'ugh' match inside
    'throughput' and 'thoughtfully', which silently poisons the human-marker
    score. Boundaries are added only at ends that begin/end with a word
    character, so markers like 'edit:' and 'ultimately,' still match.
    """
    pat = re.escape(marker)
    if marker[:1].isalnum() or marker[:1] == "_":
        pat = r"\b" + pat
    if marker[-1:].isalnum() or marker[-1:] == "_":
        pat = pat + r"\b"
    return pat


def _word_regex(words: List[str]) -> re.Pattern:
    alts = sorted((_bounded(w) for w in words), key=len, reverse=True)
    return re.compile("(?:" + "|".join(alts) + ")", re.IGNORECASE)


def _phrase_regex(phrases: List[str]) -> re.Pattern:
    alts = sorted((_bounded(p) for p in phrases), key=len, reverse=True)
    return re.compile("(?:" + "|".join(alts) + ")", re.IGNORECASE)


_RX_T1 = _word_regex(TIER1_WORDS)
_RX_T2 = _word_regex(TIER2_WORDS)
_RX_T3 = _phrase_regex(TIER3_PHRASES)
_RX_T4 = _phrase_regex(TIER4_LEAKAGE)
_RX_HUMAN = re.compile(
    "(?:" + "|".join([_bounded(m) for m in HUMAN_MARKERS]
                     + list(HUMAN_MARKER_PATTERNS)) + ")",
    re.IGNORECASE | re.MULTILINE)

# "It's not just X, it's Y" / "This isn't about X - it's about Y"
_RX_NEG_PARALLEL = re.compile(
    r"\b(?:it|this|that|they|we)\s*(?:'s|s|’s| is| are| isn't| aren't| was)?\s*"
    r"n[o']t\s+(?:just|only|about|merely|simply)\b[^.!?\n]{3,90}?"
    r"[,;—-]\s*(?:it|this|that|they|we|but)\b",
    re.IGNORECASE,
)

# Rule of three: "fast, reliable, and secure"
_RX_TRIAD = re.compile(
    r"\b(\w{4,}),\s+(\w{4,}),\s+and\s+(\w{4,})\b", re.IGNORECASE
)


def analyse(doc: Document) -> List[Signal]:
    out: List[Signal] = []
    out.append(_tier(doc, 1, _RX_T1, "lexical.tier1_register"))
    out.append(_tier(doc, 2, _RX_T2, "lexical.tier2_register"))
    out.append(_tier(doc, 3, _RX_T3, "lexical.tier3_phrases"))
    out.append(_tier(doc, 4, _RX_T4, "lexical.tier4_assistant_leakage"))
    out.append(_negative_parallelism(doc))
    out.append(_triads(doc))
    out.append(_human_markers(doc))
    return out


def _tier(doc: Document, tier: int, rx: re.Pattern, name: str) -> Signal:
    hits = rx.findall(doc.prose)
    if not hits:
        return Signal(name, "lexical", 0.0, [], 0.0)

    # Normalise long documents: use density for tiers 1-3, raw count for tier 4
    # (one leaked "as an AI language model" is decisive regardless of length).
    if tier == 4:
        effective = float(len(hits))
    else:
        per_1k = len(hits) * 1000.0 / max(doc.word_count, 1)
        # Blend count and density so neither a short nor a long text is unfairly
        # treated: a 50-word text with 3 markers and a 2000-word text with 12
        # should both register.
        effective = (len(hits) + per_1k) / 2.0

    lo = saturating(effective, TIER_MIDPOINTS[tier], TIER_CAPS[tier])
    lo *= TIER_WEIGHTS[tier] / TIER_WEIGHTS[tier]  # weights folded into caps

    uniq: Dict[str, int] = {}
    for h in hits:
        k = h.lower().strip()
        uniq[k] = uniq.get(k, 0) + 1
    ev = [f"'{k}' x{v}" if v > 1 else f"'{k}'" for k, v in
          sorted(uniq.items(), key=lambda kv: -kv[1])]

    return Signal(name, "lexical", lo, ev[:6], float(len(hits)))


def _negative_parallelism(doc: Document) -> Signal:
    hits = _RX_NEG_PARALLEL.findall(doc.prose)
    matches = [m.group(0)[:70] for m in _RX_NEG_PARALLEL.finditer(doc.prose)]
    if not matches:
        return Signal("lexical.negative_parallelism", "lexical", 0.0, [], 0.0)
    return Signal(
        name="lexical.negative_parallelism",
        layer="lexical",
        logodds=saturating(len(matches), 1.2, 1.6),
        evidence=[f"\"{m}...\"" for m in matches[:3]],
        detail=float(len(matches)),
    )


def _triads(doc: Document) -> Signal:
    matches = _RX_TRIAD.findall(doc.prose)
    if not matches:
        return Signal("lexical.rule_of_three", "lexical", 0.0, [], 0.0)
    per_1k = len(matches) * 1000.0 / max(doc.word_count, 1)
    return Signal(
        name="lexical.rule_of_three",
        layer="lexical",
        logodds=saturating(per_1k, 4.0, 0.8),
        evidence=[", ".join(m) for m in matches[:3]],
        detail=per_1k,
    )


def _human_markers(doc: Document) -> Signal:
    matches = _RX_HUMAN.findall(doc.prose)
    if not matches:
        return Signal("lexical.human_markers", "lexical", 0.0, [], 0.0)
    per_1k = len(matches) * 1000.0 / max(doc.word_count, 1)
    uniq = sorted({m.lower() for m in matches})
    # Volume-scaled, for the same reason as forensics.smart_quotes.
    #
    # This signal fires on exactly one document in the shipped corpus, so its
    # perfect accuracy rests on a single observation, and its cap was the
    # highest of any negative signal here. A lone "this morning" in 760 words
    # bought -0.89, over a third of that cap, on a document whose other
    # evidence pointed the other way. Raising the midpoint from 2.0 to 4.0 and
    # trimming the cap makes a single marker suggestive rather than decisive,
    # while two or more still carry real weight.
    return Signal(
        name="lexical.human_markers",
        layer="lexical",
        logodds=-saturating((len(matches) + per_1k) / 2.0, 4.0, 2.0),
        evidence=[f"'{u}'" for u in uniq[:6]],
        detail=float(len(matches)),
    )


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------

def attribute(doc: Document) -> List[Tuple[str, float, List[str]]]:
    """Guess which assistant produced the text.

    Returns (model_name, raw_score, evidence) sorted descending. Scores are NOT
    probabilities and are only meaningful relative to each other. Attribution is
    weak evidence by construction: these fingerprints drift with every release.
    """
    results: List[Tuple[str, float, List[str]]] = []
    low = doc.lower

    for model, profile in MODEL_PROFILES.items():
        score = 0.0
        evidence: List[str] = []

        for phrase in profile["phrases"]:
            c = low.count(phrase)
            if c:
                score += 1.0 * c
                evidence.append(f"'{phrase}' x{c}" if c > 1 else f"'{phrase}'")

        for pattern, label, threshold in profile["patterns"]:
            c = len(re.findall(pattern, doc.raw, re.MULTILINE))
            if c >= threshold:
                score += 0.7 * min(c / max(threshold, 1), 3.0)
                evidence.append(f"{label} x{c}")

        for ch in profile["unicode"]:
            c = doc.raw.count(ch)
            if c:
                score += 1.5
                evidence.append(f"U+{ord(ch):04X} artifact x{c}")

        if score > 0:
            results.append((model, round(score, 2), evidence[:5]))

    results.sort(key=lambda r: -r[1])
    return results
