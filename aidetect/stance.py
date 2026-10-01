"""Layer 8 - authorial presence.

The strongest single signal on the specification this detector originally
missed: a human writing 650 words of anything almost always leaves themselves
in it - a pronoun, a hedge, an aside, an unresolved question, a TODO. Machine
output is authorless by default.

Measured as an *absence*, which makes it unusual among the layers here and
means it needs care: absence is only evidence once the text is long enough for
the absence to be surprising.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

from .signals import Signal, saturating
from .text import Document

_RX_FIRST_SECOND = re.compile(
    r"\b(?:i|we|my|our|me|us|mine|ours|you|your|yours|i'm|i've|i'd|i'll|"
    r"we're|we've|you're|you've|let's)\b", re.IGNORECASE)

# Traces of a person thinking on the page rather than delivering a finished
# artifact: uncertainty, deferral, self-correction, direct address.
_RX_META = re.compile(
    r"\b(?:todo|tbd|tbc|fixme|note that|nb|caveat|open question|not sure|"
    r"i think|i reckon|probably|maybe|perhaps|unclear|unsure|for now|"
    r"at least for now|ideally|ugly|annoying|unfortunately|sadly|"
    r"worth checking|needs? (?:a )?(?:decision|thought)|\?\?)\b", re.IGNORECASE)

_RX_COORD3 = re.compile(r"\w+,\s+\w+,\s+(?:\w+,\s+)*(?:and |or )\w+")


def analyse(doc: Document) -> List[Signal]:
    return [
        _authorial_absence(doc),
        _term_invariance(doc),
        _enumeration_density(doc),
    ]


def _authorial_absence(doc: Document) -> Signal:
    """Zero first/second person AND zero meta-commentary over a long text."""
    if doc.word_count < 250:
        return Signal("stance.authorial_absence", "stance", 0.0, [], 0.0)

    n = doc.word_count
    person = len(_RX_FIRST_SECOND.findall(doc.prose))
    meta = len(_RX_META.findall(doc.prose))
    questions = doc.prose.count("?")

    person_per_1k = person * 1000.0 / n
    presence = person + meta + questions

    if presence == 0:
        # Scale with length: 250 clean words is ordinary, 1000 is striking.
        scale = min((n - 250) / 750.0, 1.0)
        return Signal(
            "stance.authorial_absence", "stance", 0.65 + 0.85 * scale,
            [f"no first/second person, no hedges or asides, no questions "
             f"across {n} words"], 0.0)

    if person_per_1k >= 25 or meta >= 3:
        return Signal(
            "stance.authorial_absence", "stance",
            -saturating(person_per_1k / 10.0 + meta, 3.0, 1.4),
            [f"{person_per_1k:.0f} first/second-person per 1k, {meta} hedges/asides, "
             f"{questions} questions"], person_per_1k)

    return Signal("stance.authorial_absence", "stance", 0.0,
                  [f"{person_per_1k:.0f} first/second-person per 1k"], person_per_1k)


def _term_invariance(doc: Document) -> Signal:
    """A key term repeated many times with no synonym or shorthand variation.

    People drift between 'work package', 'WP' and 'the package'. Models lock
    onto one surface form and never let go.
    """
    if doc.word_count < 300:
        return Signal("stance.term_invariance", "stance", 0.0, [], 0.0)

    content = [w.lower() for w in doc.words if len(w) > 3]
    if len(content) < 100:
        return Signal("stance.term_invariance", "stance", 0.0, [], 0.0)

    counts = Counter(content)
    term, hits = counts.most_common(1)[0]
    per_1k = hits * 1000.0 / doc.word_count
    if per_1k < 25:
        return Signal("stance.term_invariance", "stance", 0.0,
                      [f"most repeated term '{term}' x{hits}"], per_1k)
    return Signal("stance.term_invariance", "stance",
                  min((per_1k - 25) * 0.025, 0.7),
                  [f"'{term}' repeated {hits}x ({per_1k:.0f}/1k) with no variation"],
                  per_1k)


def _enumeration_density(doc: Document) -> Signal:
    """Closed, exhaustive comma-coordinated lists.

    Humans write 'etc.' and trail off. Models enumerate the complete set every
    time, which shows up as a high rate of 3+ item coordinations plus a high
    comma rate with few trailing-off markers.
    """
    if doc.word_count < 250:
        return Signal("stance.enumeration_density", "stance", 0.0, [], 0.0)

    coords = len(_RX_COORD3.findall(doc.prose))
    commas = doc.prose.count(",")
    comma_per_1k = commas * 1000.0 / doc.word_count
    openended = len(re.findall(
        r"\b(?:etc|and so on|among others|or whatever|and more|\.\.\.)",
        doc.prose, re.IGNORECASE))

    if comma_per_1k < 55 and coords < 4:
        return Signal("stance.enumeration_density", "stance", 0.0,
                      [f"{comma_per_1k:.0f} commas/1k, {coords} list coordinations"],
                      comma_per_1k)

    lo = saturating(max(comma_per_1k - 55, 0) / 15.0 + coords / 3.0, 2.5, 0.8)
    if openended:
        lo -= 0.3  # trailing off is a human habit
    return Signal("stance.enumeration_density", "stance", max(lo, 0.0),
                  [f"{comma_per_1k:.0f} commas/1k, {coords} closed list coordinations, "
                   f"{openended} open-ended markers"], comma_per_1k)
