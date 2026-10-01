"""Synthetic mixed documents with exact ground-truth spans.

The "AI-assisted" case - a human draft a model polished, or a human document
with model-written sections pasted in - is the most common real situation and
the one we could not previously express, let alone measure.

Real labelled mixed data is expensive. But splicing known-human and
known-machine text produces documents whose machine spans are known *exactly*,
because we chose them. That is enough to build and evaluate span labelling,
and it costs nothing.

What it does NOT simulate: a model rewriting a human sentence in place. Spliced
text has hard seams, where real assisted writing blends. Treat span scores from
synthetic data as an upper bound on real performance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class MixedDocument:
    text: str
    machine_spans: List[Tuple[int, int]]   # [start, end) in WORD offsets
    total_words: int
    label: str                             # human | ai_assisted | ai

    @property
    def machine_fraction(self) -> float:
        if not self.total_words:
            return 0.0
        return sum(b - a for a, b in self.machine_spans) / self.total_words

    def word_labels(self) -> List[int]:
        """Per-word ground truth: 1 machine, 0 human."""
        out = [0] * self.total_words
        for a, b in self.machine_spans:
            for i in range(max(a, 0), min(b, self.total_words)):
                out[i] = 1
        return out


def _blocks(text: str, size: int) -> List[List[str]]:
    words = text.split()
    return [words[i:i + size] for i in range(0, len(words), size)]


def make_mixed(human_texts: Sequence[str], machine_texts: Sequence[str],
               machine_ratio: float, block_words: int = 60,
               total_words: int = 900, seed: int = 0) -> MixedDocument:
    """Interleave human and machine blocks to hit `machine_ratio` by words.

    Blocks rather than single sentences, because that is what pasting looks
    like: a person writes, pastes a chunk, writes again.
    """
    rng = random.Random(seed)
    hum: List[List[str]] = []
    mac: List[List[str]] = []
    for t in human_texts:
        hum.extend(_blocks(t, block_words))
    for t in machine_texts:
        mac.extend(_blocks(t, block_words))
    hum = [b for b in hum if len(b) >= block_words // 2]
    mac = [b for b in mac if len(b) >= block_words // 2]
    if not hum or not mac:
        raise ValueError("need both human and machine source text")
    rng.shuffle(hum)
    rng.shuffle(mac)

    n_blocks = max(1, total_words // block_words)
    n_machine = int(round(n_blocks * machine_ratio))
    plan = [1] * n_machine + [0] * (n_blocks - n_machine)
    rng.shuffle(plan)

    words: List[str] = []
    spans: List[Tuple[int, int]] = []
    hi = mi = 0
    for is_machine in plan:
        src = mac if is_machine else hum
        idx = (mi if is_machine else hi) % len(src)
        block = src[idx]
        if is_machine:
            mi += 1
            spans.append((len(words), len(words) + len(block)))
        else:
            hi += 1
        words.extend(block)

    # Merge spans that ended up adjacent.
    merged: List[Tuple[int, int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    frac = sum(b - a for a, b in merged) / max(len(words), 1)
    if frac <= 0.02:
        label = "human"
    elif frac >= 0.98:
        label = "ai"
    else:
        label = "ai_assisted"

    return MixedDocument(" ".join(words), merged, len(words), label)


def make_corpus(human_texts: Sequence[str], machine_texts: Sequence[str],
                ratios: Sequence[float] = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0),
                per_ratio: int = 3, total_words: int = 900,
                seed: int = 0) -> List[MixedDocument]:
    docs: List[MixedDocument] = []
    n = 0
    for ratio in ratios:
        for _ in range(per_ratio):
            docs.append(make_mixed(human_texts, machine_texts, ratio,
                                   total_words=total_words, seed=seed + n))
            n += 1
    return docs
