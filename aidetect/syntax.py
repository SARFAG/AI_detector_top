"""Layer 7 - syntactic form, independent of vocabulary.

Added after the detector missed a machine-written technical specification that
contained zero register markers, zero Unicode artifacts and zero structural
tells. Every lexical layer was blind to it because the *words* were unremarkable.
What gave it away was the *shape*: the same syntactic templates reused across
dozens of sentences with different content words poured into them.

Word-level n-grams cannot see this. Abstracting each token to a shape class
first, then looking for repeated n-grams over those classes, can.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from .signals import Signal
from .text import Document, mean, stdev, tokenize_words

# Kept as literals rather than collapsed to one class: a template like
# "is the <word> of the <word>" is far more diagnostic than "fn w fn w".
FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and",
    "or", "for", "on", "at", "by", "its", "it", "with", "as", "that", "this",
    "from", "be", "been", "has", "have", "not", "no", "but", "if", "when",
    "than", "then", "so", "do", "does", "can", "will", "would", "any", "all",
    "each", "every", "one", "only", "still", "while", "which", "whose",
}


def _shape(word: str) -> str:
    """Collapse a token to a syntactic class, keeping function words literal."""
    low = word.lower()
    if low in FUNCTION_WORDS:
        return low
    if "_" in word:
        return "<ID>"
    if word.isupper() and len(word) > 2:
        return "<CAPS>"
    if word[:1].isupper():
        return "<Cap>"
    return "<w>"


def analyse(doc: Document) -> List[Signal]:
    return [_template_repetition(doc), _clause_uniformity(doc)]


def _template_repetition(doc: Document) -> Signal:
    """Share of 5-token syntactic templates that recur, averaged over windows.

    Measured on fixed 200-word windows rather than the whole document, because
    raw repeat-share grows with length and would otherwise just be a length
    detector. On the sample corpus the windowed figure runs 2.6-6.4% for human
    text and 13.8-19.9% for machine text - over 2x separation with no overlap,
    and the widest margin of any single feature in this package.

    CALIBRATION WARNING: the 10.0 midpoint is fit to 7 documents. The direction
    and mechanism are principled (models reuse construction patterns even while
    varying vocabulary), but the exact constant is not trustworthy. Re-fit it on
    your own corpus with aidetect.calibrate before relying on the magnitude.
    """
    if doc.word_count < 200:
        return Signal("syntax.template_repetition", "syntax", 0.0, [], 0.0)

    shapes = [_shape(w) for w in doc.words]

    def share(seq: List[str]) -> float:
        grams = [tuple(seq[i:i + 5]) for i in range(len(seq) - 4)]
        if not grams:
            return 0.0
        return (len(grams) - len(set(grams))) * 100.0 / len(grams)

    win, stride = 200, 100
    if len(shapes) < win:
        repeat_share = share(shapes)
    else:
        vals = [share(shapes[i:i + win])
                for i in range(0, len(shapes) - win + 1, stride)]
        repeat_share = sum(vals) / len(vals)

    counts = Counter(tuple(shapes[i:i + 5]) for i in range(len(shapes) - 4))
    top = [(" ".join(g), c) for g, c in counts.most_common(3) if c > 1]
    ev = [f"{repeat_share:.1f}% of 5-token templates recur (length-normalised)"]
    ev += [f"'{t}' x{c}" for t, c in top]

    # The measure is computed on 200-word windows, so a 200-word document is
    # exactly one full window and its estimate is as sound as any single window
    # inside a longer one. The previous /400 halved a well-supported signal for
    # no stated reason.
    confidence = min(doc.word_count / 300.0, 1.0)
    lo = max(-1.5, min(1.5, (repeat_share - 10.0) * 0.18)) * confidence
    return Signal("syntax.template_repetition", "syntax", lo, ev, repeat_share)


def _clause_uniformity(doc: Document) -> Signal:
    """Burstiness measured between clauses rather than between sentences.

    Catches text that varies sentence length while keeping every clause the
    same size - a shape that sentence-level burstiness scores as human.
    """
    clauses = [c for c in re.split(r"[,;:.!?—]", doc.prose)]
    lens = [len(tokenize_words(c)) for c in clauses]
    lens = [n for n in lens if n > 1]
    if len(lens) < 20:
        return Signal("syntax.clause_uniformity", "syntax", 0.0, [], 0.0)

    m = mean(lens)
    cv = stdev(lens) / m if m else 0.0
    confidence = min(len(lens) / 60.0, 1.0)
    # Human clause CV runs ~0.7-1.1; machine ~0.4-0.6.
    raw = (0.72 - cv) * 2.0
    lo = max(-1.2, min(1.2, raw)) * confidence
    return Signal("syntax.clause_uniformity", "syntax", lo,
                  [f"clause length CV={cv:.2f} (mean {m:.1f} words, n={len(lens)})"], cv)
