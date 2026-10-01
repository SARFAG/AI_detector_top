"""Technical-register detection, shared by the genre-correlated signals.

Several signals measure polish and impersonality rather than authorship:
absence of first person, absence of questions, absence of typos. Technical
specifications have all three whether a person or a model wrote them, which
produced a confirmed false positive - a human-written spec scored 88.1%, with
those signals supplying most of it.

Windowed scoring accidentally avoided the problem, because every one of those
signals has a length gate and switches off in a 200-word window. That is a
diagnosis, not a fix: the signals are genre-correlated and should say so
themselves.
"""

from __future__ import annotations

import re

from .text import Document

_RX_IDENT = re.compile(r"\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z_]+\b")

# Identifiers per 1k words above which a document is technical reference prose.
TECHNICAL_IDENT_RATE = 15.0

# Fraction of a genre-correlated signal's positive evidence that survives
# there. Not zero: machine-written specs still sit at the same floor, so the
# signal retains some value - it just cannot carry a verdict alone.
TECHNICAL_DAMPING = 0.35


def identifier_rate(doc: Document) -> float:
    return len(_RX_IDENT.findall(doc.prose)) * 1000.0 / max(doc.word_count, 1)


def is_technical(doc: Document) -> bool:
    return identifier_rate(doc) >= TECHNICAL_IDENT_RATE


def damping(doc: Document) -> float:
    """Multiplier for a genre-correlated signal's positive evidence."""
    return TECHNICAL_DAMPING if is_technical(doc) else 1.0
