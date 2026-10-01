"""Layer 2 - distributional statistics.

This layer measures the central fact from the research: LLM text is *too
regular*. Humans are bursty - they mix a nine-word sentence with a forty-word
one, repeat themselves unevenly, and vary paragraph size wildly. Models regress
to the mean on all three.

None of these need a reference language model, which is why they run anywhere.
The tradeoff is that they are weaker than true perplexity-based signals; see
`aidetect/probe/` for the model-backed versions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

from .signals import Signal
from .text import Document, mean, stdev, tokenize_words

# Discourse connectives. LLMs scaffold arguments with these far more than
# people do in natural writing.
TRANSITIONS = [
    "moreover", "furthermore", "additionally", "consequently", "nevertheless",
    "nonetheless", "therefore", "thus", "hence", "accordingly", "subsequently",
    "in addition", "on the other hand", "in contrast", "similarly",
    "importantly", "notably", "specifically", "ultimately", "overall",
    "in essence", "that said", "however",
]

# Hedges and epistemic softeners.
HEDGES = [
    "may", "might", "could", "can be", "often", "typically", "generally",
    "usually", "sometimes", "arguably", "relatively", "somewhat", "tends to",
    "it depends", "in many cases", "in some cases", "largely", "broadly",
]

CONTRACTIONS = re.compile(
    r"\b\w+(?:'|’)(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE
)

_RX_TRANSITIONS = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in TRANSITIONS) + r")\b", re.IGNORECASE
)
_RX_HEDGES = re.compile(
    r"\b(?:" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.IGNORECASE
)


def analyse(doc: Document) -> List[Signal]:
    return [
        _sentence_burstiness(doc),
        _paragraph_uniformity(doc),
        _lexical_diversity(doc),
        _ngram_repetition(doc),
        _transition_density(doc),
        _hedge_density(doc),
        _sentence_opener_diversity(doc),
        _contraction_balance(doc),
    ]


def _lengths(doc: Document) -> List[int]:
    return [len(tokenize_words(s)) for s in doc.sentences]


def _sentence_burstiness(doc: Document) -> Signal:
    """Coefficient of variation of sentence length.

    Empirically: human prose lands around 0.55-0.80, LLM prose around
    0.30-0.48. Below 0.35 over enough sentences is a strong machine signal.
    """
    lens = [l for l in _lengths(doc) if l > 0]
    if len(lens) < 6:
        return Signal("statistical.burstiness", "statistical", 0.0,
                      ["too few sentences to measure"], 0.0)

    m = mean(lens)
    sd = stdev(lens)
    cv = sd / m if m else 0.0

    # Piecewise-linear map centred on 0.52, scaled by how much data we have.
    confidence = min(len(lens) / 25.0, 1.0)
    raw = (0.52 - cv) * 5.0          # cv 0.30 -> +1.1 ; cv 0.75 -> -1.15
    lo = max(-2.0, min(2.0, raw)) * confidence

    ev = [f"CV={cv:.2f} (mean {m:.1f} words, sd {sd:.1f}, n={len(lens)})"]
    if cv < 0.35:
        ev.append("unusually uniform sentence lengths")
    elif cv > 0.70:
        ev.append("highly variable (bursty) sentence lengths")
    return Signal("statistical.burstiness", "statistical", lo, ev, cv)


def _paragraph_uniformity(doc: Document) -> Signal:
    """LLMs emit 3-5 sentence blocks over and over; humans do not."""
    paras = [p for p in doc.paragraphs if len(tokenize_words(p)) >= 8]
    if len(paras) < 4:
        return Signal("statistical.paragraph_uniformity", "statistical", 0.0, [], 0.0)

    lens = [len(tokenize_words(p)) for p in paras]
    m = mean(lens)
    cv = stdev(lens) / m if m else 0.0
    confidence = min(len(paras) / 8.0, 1.0)
    raw = (0.45 - cv) * 3.2
    lo = max(-1.4, min(1.4, raw)) * confidence
    return Signal(
        "statistical.paragraph_uniformity", "statistical", lo,
        [f"paragraph length CV={cv:.2f} across {len(paras)} paragraphs"], cv,
    )


def _lexical_diversity(doc: Document) -> Signal:
    """Moving-average type-token ratio over a 50-word window.

    Length-independent, unlike plain TTR. Interpreted cautiously: low diversity
    also describes a non-native speaker writing plainly, which is exactly the
    fairness failure documented in docs/LIMITS.md. Weighted low on purpose.
    """
    words = [w.lower() for w in doc.words]
    window = 50
    if len(words) < window * 2:
        return Signal("statistical.lexical_diversity", "statistical", 0.0, [], 0.0)

    ratios = []
    for i in range(0, len(words) - window + 1, 10):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / float(window))
    mattr = mean(ratios)

    # Narrow band, small weight. MATTR ~0.72 is typical of both classes; only
    # the extremes carry any information.
    if mattr > 0.80:
        lo = -0.35
        note = "high lexical variety"
    elif mattr < 0.62:
        lo = 0.30
        note = "low lexical variety"
    else:
        lo = 0.0
        note = "unremarkable"
    return Signal("statistical.lexical_diversity", "statistical", lo,
                  [f"MATTR-50={mattr:.3f} ({note})"], mattr)


def _ngram_repetition(doc: Document) -> Signal:
    """Repeated 4-grams. Models reuse phrasing scaffolds across sections."""
    words = [w.lower() for w in doc.words]
    if len(words) < 120:
        return Signal("statistical.ngram_repetition", "statistical", 0.0, [], 0.0)

    total_grams = len(words) - 3
    grams = Counter(tuple(words[i:i + 4]) for i in range(total_grams))
    repeated = {g: c for g, c in grams.items() if c > 1}
    # Share of all 4-gram positions that sit inside a repeated phrase.
    rate = sum(repeated.values()) / float(total_grams)

    top = sorted(repeated.items(), key=lambda kv: -kv[1])[:3]
    ev = [f"'{' '.join(g)}' x{c}" for g, c in top]
    ev.insert(0, f"repeated 4-gram rate={rate:.3f}")

    # Both classes repeat; only elevated rates carry signal, and weakly.
    lo = 0.0
    if rate > 0.045:
        lo = min((rate - 0.045) * 14.0, 0.9)
    return Signal("statistical.ngram_repetition", "statistical", lo, ev, rate)


def _transition_density(doc: Document) -> Signal:
    hits = _RX_TRANSITIONS.findall(doc.prose)
    if doc.word_count < 80:
        return Signal("statistical.transition_density", "statistical", 0.0, [], 0.0)
    per_1k = len(hits) * 1000.0 / doc.word_count

    # Natural writing sits around 4-9 per 1k; scaffolded LLM prose runs 15+.
    if per_1k > 11:
        lo = min((per_1k - 11) * 0.09, 1.1)
    elif per_1k < 2:
        lo = -0.25
    else:
        lo = 0.0
    uniq = Counter(h.lower() for h in hits)
    ev = [f"{per_1k:.1f} connectives per 1k words"]
    ev += [f"'{k}' x{v}" for k, v in uniq.most_common(3)]
    return Signal("statistical.transition_density", "statistical", lo, ev, per_1k)


def _hedge_density(doc: Document) -> Signal:
    if doc.word_count < 120:
        return Signal("statistical.hedge_density", "statistical", 0.0, [], 0.0)
    hits = _RX_HEDGES.findall(doc.prose)
    per_1k = len(hits) * 1000.0 / doc.word_count
    lo = min((per_1k - 14) * 0.05, 0.8) if per_1k > 14 else 0.0
    return Signal("statistical.hedge_density", "statistical", lo,
                  [f"{per_1k:.1f} hedges per 1k words"], per_1k)


def _sentence_opener_diversity(doc: Document) -> Signal:
    """Repeating the same first word across many sentences is a scaffold tell."""
    if len(doc.sentences) < 8:
        return Signal("statistical.opener_diversity", "statistical", 0.0, [], 0.0)
    openers = []
    for s in doc.sentences:
        w = tokenize_words(s)
        if w:
            openers.append(w[0].lower())
    if not openers:
        return Signal("statistical.opener_diversity", "statistical", 0.0, [], 0.0)

    diversity = len(set(openers)) / float(len(openers))
    counts = Counter(openers)
    top_word, top_n = counts.most_common(1)[0]
    ev = [f"opener diversity={diversity:.2f}", f"most common opener '{top_word}' x{top_n}"]

    lo = 0.0
    if diversity < 0.55:
        lo = min((0.55 - diversity) * 2.4, 0.7)
    return Signal("statistical.opener_diversity", "statistical", lo, ev, diversity)


def _contraction_balance(doc: Document) -> Signal:
    """Mismatch between informal register and contraction rate.

    A text full of casual markers but with no contractions reads as a model
    imitating informality. Formal text with no contractions is unremarkable, so
    this only fires when the register says "casual".
    """
    if doc.word_count < 150:
        return Signal("statistical.contraction_balance", "statistical", 0.0, [], 0.0)

    contractions = len(CONTRACTIONS.findall(doc.prose))
    per_1k = contractions * 1000.0 / doc.word_count

    casual = len(re.findall(
        r"\b(?:you|your|we|our|let's|here's|that's|really|pretty|just|stuff|thing)\b",
        doc.prose, re.IGNORECASE))
    casual_per_1k = casual * 1000.0 / doc.word_count

    if casual_per_1k > 20 and per_1k < 4:
        return Signal(
            "statistical.contraction_balance", "statistical", 0.55,
            [f"casual register ({casual_per_1k:.0f}/1k) but near-zero contractions "
             f"({per_1k:.1f}/1k)"], per_1k)
    return Signal("statistical.contraction_balance", "statistical", 0.0,
                  [f"{per_1k:.1f} contractions per 1k words"], per_1k)
