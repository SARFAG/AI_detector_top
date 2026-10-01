"""Layer 6 (optional) - model-backed detection.

Everything in the base package is a proxy for what a real language model can
measure directly. This is where the actual accuracy lives. It needs PyTorch and
open-weights models, so it is isolated here and imported lazily; the rest of
aidetect runs with zero dependencies.

Implemented:
  * Binoculars      - best zero-shot method; low false-positive rate on
                      non-native English, which is the failure mode that
                      matters most (see docs/LIMITS.md).
  * Perplexity/burstiness - true per-token versions of the proxies in
                      aidetect/statistical.py.

Install:
    pip install torch transformers

Then:
    from aidetect.probe import Binoculars
    b = Binoculars()                 # downloads ~14GB on first use
    print(b.score(text))             # LOW score => machine-generated
"""

from __future__ import annotations

AVAILABLE = True
try:  # pragma: no cover - depends on optional install
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    AVAILABLE = False

from .binoculars import Binoculars, BinocularsUnavailable  # noqa: E402

__all__ = ["Binoculars", "BinocularsUnavailable", "AVAILABLE"]
