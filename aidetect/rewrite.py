"""RAIDAR: detection by rewriting distance.

Ask a model to rewrite the text, then measure how much it changed. It rewrites
*human* text heavily and *machine* text barely, because machine text already
looks like its own output - there is nothing it wants to fix. The edit distance
is the feature.

Fully orthogonal to every other layer here: it needs no reference corpus, no
lexicon, and no assumption about register, which is exactly the weakness that
defeated this detector on terse technical prose.

The model call is yours to supply - this module does not reach the network.
Pass any callable that takes a string and returns the rewritten string:

    from aidetect.rewrite import raidar_signal

    def my_rewriter(text: str) -> str:
        return call_your_model(f"Rewrite this text:\\n\\n{text}")

    sig = raidar_signal(doc, my_rewriter)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .signals import Signal
from .text import Document, tokenize_words

Rewriter = Callable[[str], str]

PROMPT = ("Rewrite the following text. Keep the meaning and roughly the same "
          "length. Return only the rewritten text.\n\n")

# Below this edit ratio the rewriter barely touched it, which is the
# machine-text signature. Provisional: re-derive on your own data, since it
# depends on the rewriting model and the prompt.
MACHINE_CEILING = 0.25
HUMAN_FLOOR = 0.45


def word_edit_ratio(a: str, b: str) -> float:
    """Levenshtein distance over word tokens, normalised by the longer text.

    Word-level rather than character-level so that reformatting and
    punctuation changes do not dominate the signal.
    """
    x, y = tokenize_words(a.lower()), tokenize_words(b.lower())
    if not x and not y:
        return 0.0
    if not x or not y:
        return 1.0

    # Row-wise DP; only two rows are ever needed.
    prev = list(range(len(y) + 1))
    for i, xi in enumerate(x, 1):
        cur = [i] + [0] * len(y)
        for j, yj in enumerate(y, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (xi != yj))
        prev = cur
    return prev[-1] / max(len(x), len(y))


def raidar_score(text: str, rewriter: Rewriter, rounds: int = 1) -> float:
    """Mean edit ratio across `rounds` rewrites. LOW means machine-written."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    ratios = []
    for _ in range(rounds):
        rewritten = rewriter(PROMPT + text)
        ratios.append(word_edit_ratio(text, rewritten))
    return sum(ratios) / len(ratios)


def raidar_signal(doc: Document, rewriter: Optional[Rewriter],
                  rounds: int = 1) -> Signal:
    """RAIDAR as an ensemble signal. Returns a neutral signal with no rewriter."""
    if rewriter is None:
        return Signal("rewrite.raidar", "rewrite", 0.0,
                      ["no rewriter supplied; pass one to enable RAIDAR"], 0.0)
    if doc.word_count < 60:
        return Signal("rewrite.raidar", "rewrite", 0.0,
                      ["too short to rewrite meaningfully"], 0.0)

    ratio = raidar_score(doc.raw, rewriter, rounds=rounds)
    if ratio <= MACHINE_CEILING:
        lo = min((MACHINE_CEILING - ratio) * 6.0, 1.6)
        note = "barely rewritten: already reads as model output"
    elif ratio >= HUMAN_FLOOR:
        lo = -min((ratio - HUMAN_FLOOR) * 4.0, 1.4)
        note = "heavily rewritten: the model wanted to change a lot"
    else:
        lo = 0.0
        note = "intermediate"
    return Signal("rewrite.raidar", "rewrite", lo,
                  [f"edit ratio {ratio:.3f} over {rounds} round(s): {note}"],
                  ratio)
