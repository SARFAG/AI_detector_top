"""Core signal type shared by every detection layer.

Every layer emits `Signal` objects instead of a bare number. A signal carries a
log-odds contribution so that the detector can sum evidence additively and pass
the total through a sigmoid to get a calibrated probability. It also carries the
evidence that produced it, so any score can be traced back to a span of input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Signal:
    """One piece of evidence about whether a text is machine-generated.

    Attributes:
        name: Stable identifier, e.g. ``"forensics.em_dash"``.
        layer: Which layer produced it (forensics, lexical, statistical, ...).
        logodds: Contribution to the total. Positive means "more likely machine",
            negative means "more likely human". Zero means the signal looked but
            found nothing decisive.
        evidence: Short human-readable strings, usually quoted spans.
        detail: Raw measured value, for debugging and calibration.
    """

    name: str
    layer: str
    logodds: float
    evidence: List[str] = field(default_factory=list)
    detail: float = 0.0

    @property
    def direction(self) -> str:
        if self.logodds > 0.05:
            return "machine"
        if self.logodds < -0.05:
            return "human"
        return "neutral"

    def __str__(self) -> str:
        sign = "+" if self.logodds >= 0 else ""
        return f"{self.name:<34} {sign}{self.logodds:5.2f}  {'; '.join(self.evidence[:3])}"


def sigmoid(x: float) -> float:
    # Clamped to avoid overflow on extreme evidence sums.
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def saturating(value: float, midpoint: float, max_logodds: float) -> float:
    """Map an unbounded count/rate onto a bounded log-odds contribution.

    A single stray marker should barely move the needle; twenty should approach
    but never exceed ``max_logodds``. Without this, a long document would
    accumulate unbounded evidence purely by being long.
    """
    if value <= 0:
        return 0.0
    return max_logodds * (value / (value + midpoint))
