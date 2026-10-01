"""Classical stylometry: function-word distributions and Burrows's Delta.

Burrows's Delta is the canonical authorship method and predates all of this by
decades. It ignores content entirely and looks only at how often an author uses
the small closed class of function words - the, of, that, which - because those
rates are largely unconscious and remarkably stable per author.

Two uses here:

1. **Nearest-centroid AI detection.** Z-score a text's function-word profile
   against a reference corpus, then measure its Delta to the human centroid and
   to the machine centroid. Closer wins. No training beyond the corpus itself.

2. **Authorship verification**, which is a different and much stronger
   question: not "does this look machine-written?" but "does this match *this
   person's* known writing?" Comparing someone against their own baseline
   sidesteps the fairness problem in LIMITS.md - a non-native speaker is
   measured against themselves, not against native-speaker norms. It needs a
   reference sample of the author's prior work, which this repo cannot supply
   but a user often can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .text import Document, mean, stdev, tokenize_words

# The closed class. Deliberately content-free: these rates say something about
# the writer's habits rather than the subject matter.
FUNCTION_WORDS = [
    "the", "of", "and", "a", "to", "in", "is", "be", "that", "was", "for",
    "it", "with", "as", "his", "on", "at", "by", "i", "this", "had", "not",
    "are", "but", "from", "or", "have", "an", "they", "which", "one", "you",
    "were", "her", "all", "she", "there", "would", "their", "we", "him",
    "been", "has", "when", "who", "will", "no", "more", "if", "out", "so",
    "said", "what", "up", "its", "about", "into", "than", "them", "can",
    "only", "other", "new", "some", "could", "time", "these", "two", "may",
    "then", "do", "first", "any", "my", "now", "such", "like", "our", "over",
    "man", "me", "even", "most", "made", "after", "also", "did", "many",
    "before", "must", "through", "back", "years", "where", "much", "your",
    "way", "well", "down", "should", "because", "each", "just", "those",
    "how", "too", "little", "state", "good", "very", "make", "world", "still",
    "own", "see", "men", "work", "long", "get", "here", "between", "both",
    "life", "being", "under", "never", "day", "same", "another", "know",
    "while", "last", "might", "us", "great", "old", "year", "off", "come",
    "since", "against", "go", "came", "right", "used", "take", "three",
]

_PUNCT = ".,;:!?-()\"'"


@dataclass
class Profile:
    """A text's stylometric fingerprint."""
    function_freqs: Dict[str, float]
    punct_freqs: Dict[str, float]
    punct_bigrams: Dict[str, float]
    word_count: int


def profile(doc: Document) -> Profile:
    words = [w.lower() for w in doc.words]
    n = max(len(words), 1)
    counts: Dict[str, int] = {w: 0 for w in FUNCTION_WORDS}
    for w in words:
        if w in counts:
            counts[w] += 1
    freqs = {w: c / n for w, c in counts.items()}

    chars = doc.prose
    cn = max(len(chars), 1)
    pf = {p: chars.count(p) / cn for p in _PUNCT}

    # Punctuation bigrams: the sequence of punctuation marks, ignoring words.
    seq = [c for c in chars if c in _PUNCT]
    bigrams: Dict[str, int] = {}
    for a, b in zip(seq, seq[1:]):
        bigrams[a + b] = bigrams.get(a + b, 0) + 1
    tot = max(sum(bigrams.values()), 1)
    pb = {k: v / tot for k, v in bigrams.items()}

    return Profile(freqs, pf, pb, len(words))


# Frequencies are rates in roughly the 0 - 0.07 range, so this is the smallest
# standard deviation that still means something. Below it the word carries no
# usable variance and is dropped rather than divided by.
SD_FLOOR = 1e-4


@dataclass
class DeltaModel:
    """Reference statistics for z-scoring. Built from a corpus.

    Words with no variance in the reference corpus are EXCLUDED, not floored.
    Flooring them at a tiny epsilon was a real bug: on a small corpus many
    function words ("she", "his", "him") never appear at all, so their sd was
    zero. Dividing by 1e-9 gave any outside document that used one a z-score
    near a million, which swamped the whole distance. In-corpus documents never
    triggered it, so in-sample testing could not see it.
    """
    means: Dict[str, float]
    sds: Dict[str, float]
    active: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.active:
            self.active = [w for w in FUNCTION_WORDS
                           if self.sds.get(w, 0.0) >= SD_FLOOR]

    @classmethod
    def fit(cls, profiles: Sequence[Profile]) -> "DeltaModel":
        if not profiles:
            raise ValueError("need at least one profile")
        means, sds = {}, {}
        for w in FUNCTION_WORDS:
            vals = [p.function_freqs.get(w, 0.0) for p in profiles]
            means[w] = mean(vals)
            sds[w] = stdev(vals)
        return cls(means, sds)

    def zscores(self, p: Profile) -> Dict[str, float]:
        return {w: (p.function_freqs.get(w, 0.0) - self.means[w]) / self.sds[w]
                for w in self.active}


def delta(model: DeltaModel, a: Profile, b: Profile) -> float:
    """Burrows's Delta: mean absolute difference of z-scores. Lower = closer."""
    za, zb = model.zscores(a), model.zscores(b)
    if not model.active:
        return 0.0
    return mean([abs(za[w] - zb[w]) for w in model.active])


def centroid(model: DeltaModel, profiles: Sequence[Profile]) -> Dict[str, float]:
    """Mean z-score vector of a class."""
    zs = [model.zscores(p) for p in profiles]
    return {w: mean([z[w] for z in zs]) for w in model.active}


def delta_to_centroid(model: DeltaModel, p: Profile,
                      cent: Dict[str, float]) -> float:
    z = model.zscores(p)
    shared = [w for w in model.active if w in cent]
    if not shared:
        return 0.0
    return mean([abs(z[w] - cent[w]) for w in shared])


class NearestCentroid:
    """Burrows's Delta as a two-class detector.

    Fit on labelled corpora, then score: positive means closer to the machine
    centroid than the human one.
    """

    def __init__(self, human_docs: Sequence[Document],
                 machine_docs: Sequence[Document]):
        hp = [profile(d) for d in human_docs]
        mp = [profile(d) for d in machine_docs]
        self.model = DeltaModel.fit(list(hp) + list(mp))
        self.human_centroid = centroid(self.model, hp)
        self.machine_centroid = centroid(self.model, mp)

    def score(self, doc: Document) -> Tuple[float, float, float]:
        """Return (margin, delta_to_human, delta_to_machine).

        Positive margin means machine-leaning.
        """
        p = profile(doc)
        dh = delta_to_centroid(self.model, p, self.human_centroid)
        dm = delta_to_centroid(self.model, p, self.machine_centroid)
        return dh - dm, dh, dm


def verify_authorship(candidate: Document, reference: Sequence[Document],
                      background: Sequence[Document]) -> Dict[str, float]:
    """Does `candidate` match the author of `reference`?

    `background` is a corpus of other writers, used both to z-score and to say
    how unusual the candidate-to-reference distance is. A delta well inside the
    background distribution means the candidate is consistent with that author.

    This is the stronger question, and the fairer one: the author is their own
    baseline, so writing simply, or in a second language, is not itself
    evidence of anything.
    """
    refs = [profile(d) for d in reference]
    bg = [profile(d) for d in background]
    if not refs:
        raise ValueError("need at least one reference document")
    model = DeltaModel.fit(refs + bg + [profile(candidate)])
    ref_centroid = centroid(model, refs)

    cand = delta_to_centroid(model, profile(candidate), ref_centroid)
    bg_deltas = [delta_to_centroid(model, p, ref_centroid) for p in bg]

    if bg_deltas:
        below = sum(1 for d in bg_deltas if d < cand) / len(bg_deltas)
    else:
        below = float("nan")

    return {
        "delta_to_author": cand,
        "background_mean": mean(bg_deltas) if bg_deltas else float("nan"),
        "background_sd": stdev(bg_deltas) if len(bg_deltas) > 1 else float("nan"),
        # Fraction of other writers who match this author's profile BETTER than
        # the candidate does. Near 0 means a strong match.
        "percentile_among_others": below,
        "consistent_with_author": below < 0.5 if bg_deltas else None,
    }


# --------------------------------------------------------------------------
# Ensemble signal
# --------------------------------------------------------------------------

_CENTROIDS: Optional[dict] = None


def _load_centroids() -> Optional[dict]:
    """Load the shipped centroids once, lazily."""
    global _CENTROIDS
    if _CENTROIDS is None:
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "data", "centroids.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                _CENTROIDS = json.load(fh)
        except (OSError, ValueError):
            _CENTROIDS = {}
    return _CENTROIDS or None


def analyse(doc: Document) -> List["Signal"]:
    """Burrows's Delta against the shipped human/machine centroids.

    Content-free by construction - it reads only function-word rates - which
    makes it close to orthogonal to every other layer here. That independence
    is the reason to include it even though the shipped centroids are fitted
    on seven documents and are correspondingly weak.
    """
    from .signals import Signal

    cents = _load_centroids()
    if cents is None or doc.word_count < 200:
        return [Signal("stylometry.burrows_delta", "stylometry", 0.0, [], 0.0)]

    model = DeltaModel(cents["means"], cents["sds"], cents.get("active", []))
    p = profile(doc)
    dh = delta_to_centroid(model, p, cents["human_centroid"])
    dm = delta_to_centroid(model, p, cents["machine_centroid"])
    margin = dh - dm

    confidence = min(doc.word_count / 400.0, 1.0)
    lo = max(-1.2, min(1.2, margin * 3.0)) * confidence
    return [Signal(
        "stylometry.burrows_delta", "stylometry", lo,
        [f"delta to human {dh:.3f}, to machine {dm:.3f} (margin {margin:+.3f})"],
        margin,
    )]
