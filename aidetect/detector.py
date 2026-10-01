"""The ensemble: combine every layer's log-odds into a calibrated probability.

Design rules, in order of importance:

1. **Evidence, not verdicts.** The report always carries the per-signal
   breakdown. A bare number is not actionable and invites misuse.
2. **Short text cannot convict.** Evidence is shrunk toward zero below ~250
   words, because none of the statistical signals are meaningful there. A
   40-word text should essentially always come back "insufficient evidence".
3. **Bounded evidence.** Each signal is capped, so length alone cannot
   accumulate a conviction.
4. **A human prior.** Base log-odds start slightly negative: in most corpora
   most text is still human-written, and the cost of a false accusation is far
   higher than the cost of a miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import code as code_layer
from . import forensics, lexical, prompts, stance, statistical, structural, syntax
from .signals import Signal, sigmoid
from .text import Document, parse

# Most text in the wild is still human-written, and a false accusation costs
# far more than a miss. Start the evidence meter slightly below neutral.
BASE_LOGODDS = -0.45

# Below this, statistical signals are noise and the verdict is withheld.
MIN_RELIABLE_WORDS = 250
MIN_ANY_WORDS = 40

LAYER_WEIGHTS = {
    "forensics": 1.00,
    "lexical": 1.00,
    "statistical": 0.85,
    "structural": 0.70,
    "code": 0.90,
    "prompt": 0.60,
    # Added after the detector missed a machine-written specification that was
    # invisible to every vocabulary-based layer. These measure form and
    # authorial presence rather than word choice, so they survive the terse,
    # marker-free register that defeated the original ensemble.
    "syntax": 0.95,
    "stance": 0.95,
}

BANDS = [
    (0.90, "very likely machine-generated"),
    (0.75, "likely machine-generated"),
    (0.60, "leans machine-generated"),
    (0.40, "inconclusive"),
    (0.25, "leans human-written"),
    (0.10, "likely human-written"),
    (0.00, "very likely human-written"),
]


@dataclass
class Report:
    probability: float
    verdict: str
    confidence: str
    signals: List[Signal]
    attribution: List[Tuple[str, float, List[str]]]
    word_count: int
    total_logodds: float
    layer_totals: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def top_signals(self) -> List[Signal]:
        return sorted(
            [s for s in self.signals if abs(s.logodds) > 0.01],
            key=lambda s: -abs(s.logodds),
        )

    def machine_evidence(self) -> List[Signal]:
        return [s for s in self.top_signals if s.logodds > 0]

    def human_evidence(self) -> List[Signal]:
        return [s for s in self.top_signals if s.logodds < 0]


def _band(p: float) -> str:
    for threshold, label in BANDS:
        if p >= threshold:
            return label
    return "inconclusive"


def _confidence(word_count: int, signals: List[Signal]) -> str:
    active = sum(1 for s in signals if abs(s.logodds) > 0.1)
    if word_count < MIN_ANY_WORDS:
        return "none"
    if word_count < MIN_RELIABLE_WORDS:
        return "low"
    if active >= 6 and word_count >= 600:
        return "high"
    if active >= 3:
        return "medium"
    return "low"


def _length_shrinkage(word_count: int) -> float:
    """Scale evidence down for short inputs.

    Full strength at 250+ words, linearly reduced below that, floored so a very
    short but blatant paste (e.g. "As an AI language model...") still registers.
    """
    if word_count >= MIN_RELIABLE_WORDS:
        return 1.0
    if word_count <= MIN_ANY_WORDS:
        return 0.35
    span = MIN_RELIABLE_WORDS - MIN_ANY_WORDS
    return 0.35 + 0.65 * ((word_count - MIN_ANY_WORDS) / span)


def analyse(text: str, include_prompt_layer: bool = True) -> Report:
    """Run every layer over `text` and return a full evidence report."""
    doc: Document = parse(text)

    signals: List[Signal] = []
    signals += forensics.analyse(doc)
    signals += lexical.analyse(doc)
    signals += statistical.analyse(doc)
    signals += structural.analyse(doc)
    signals += code_layer.analyse(doc)
    signals += syntax.analyse(doc)
    signals += stance.analyse(doc)
    if include_prompt_layer:
        signals += prompts.analyse(doc)

    layer_totals: Dict[str, float] = {}
    total = BASE_LOGODDS
    for s in signals:
        w = LAYER_WEIGHTS.get(s.layer, 1.0)
        contribution = s.logodds * w
        layer_totals[s.layer] = layer_totals.get(s.layer, 0.0) + contribution
        total += contribution

    shrink = _length_shrinkage(doc.word_count)
    # Shrink only the evidence, not the prior.
    total = BASE_LOGODDS + (total - BASE_LOGODDS) * shrink

    probability = sigmoid(total)
    confidence = _confidence(doc.word_count, signals)

    notes: List[str] = []
    if doc.word_count < MIN_ANY_WORDS:
        notes.append(
            f"Only {doc.word_count} words. Far too short for any reliable "
            f"judgement; treat this result as meaningless.")
        probability = 0.5
        confidence = "none"
    elif doc.word_count < MIN_RELIABLE_WORDS:
        notes.append(
            f"Only {doc.word_count} words (reliable analysis needs ~"
            f"{MIN_RELIABLE_WORDS}+). Statistical signals are suppressed.")

    inj = [s for s in signals if s.name == "prompt.injection_surface" and s.evidence]
    for s in inj:
        notes.extend(s.evidence)

    verdict = _band(probability) if confidence != "none" else "insufficient text"

    return Report(
        probability=probability,
        verdict=verdict,
        confidence=confidence,
        signals=signals,
        attribution=lexical.attribute(doc),
        word_count=doc.word_count,
        total_logodds=total,
        layer_totals=layer_totals,
        notes=notes,
    )


def analyse_segments(text: str, window: int = 3) -> List[Tuple[str, float]]:
    """Score each paragraph window separately to localise machine-written spans.

    This is what makes mixed documents tractable: a human essay with two pasted
    LLM paragraphs scores unremarkably as a whole, but lights up locally.
    """
    doc = parse(text)
    paras = doc.paragraphs
    out: List[Tuple[str, float]] = []
    if not paras:
        return out

    # Shrink the window on short documents, otherwise a two-paragraph document
    # collapses into a single segment and localises nothing. Slide with stride 1
    # so a boundary between human and pasted text is not blurred across windows.
    w = max(1, min(window, len(paras) // 2 or 1))
    for i in range(0, len(paras) - w + 1):
        chunk = "\n\n".join(paras[i:i + w])
        if len(chunk.split()) < 25:
            continue
        rep = analyse(chunk)
        out.append((chunk, rep.probability))
    return out


# --------------------------------------------------------------------------
# Windowed scoring
# --------------------------------------------------------------------------

WINDOW_WORDS = 250
WINDOW_STRIDE = 125


def fraction_ai(text: str, window: int = WINDOW_WORDS,
                stride: int = WINDOW_STRIDE, threshold: float = 0.5):
    """Score overlapping fixed-width windows; return (fraction, per-window scores).

    A single document-level number is dominated by whichever register covers
    the most words. Scoring fixed windows and reporting the *fraction* above
    threshold is both more robust on mixed documents and the shape that
    production detectors report.
    """
    words = text.split()
    if len(words) < MIN_ANY_WORDS:
        return 0.0, []

    scores = []
    step = max(stride, 1)
    for start in range(0, max(len(words) - window, 0) + 1, step):
        chunk = " ".join(words[start:start + window])
        if len(chunk.split()) < MIN_ANY_WORDS:
            continue
        scores.append(analyse(chunk).probability)
        if start + window >= len(words):
            break

    if not scores:
        return 0.0, []
    return sum(p >= threshold for p in scores) / len(scores), scores
