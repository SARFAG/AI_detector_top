"""aidetect - a layered, evidence-reporting detector for machine-generated text.

Quick start:

    >>> import aidetect
    >>> report = aidetect.analyse(open("essay.txt").read())
    >>> report.probability, report.verdict
    (0.87, 'likely machine-generated')
    >>> for s in report.top_signals[:5]:
    ...     print(s)

Read docs/LIMITS.md before acting on any score. These scores are evidence for a
human to weigh, never a verdict to act on automatically.
"""

from .detector import Report, analyse, analyse_segments
from .signals import Signal

__version__ = "0.1.0"
__all__ = ["analyse", "analyse_segments", "Report", "Signal"]
