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

from .register import damping as _register_damping


def analyse(doc: Document) -> List[Signal]:
    return [
        _authorial_absence(doc),
        _term_invariance(doc),
        _enumeration_density(doc),
    ]


def _authorial_absence(doc: Document) -> Signal:
    """Zero first/second person AND zero meta-commentary over a long text."""
    # Gate lowered from 250 to 180 on measurement: truncated to 200 words,
    # every human sample still showed 11-17 presence markers and every ordinary
    # machine sample 1-8, while marker-free specs showed exactly 0. Absolute
    # zero is already separating at this length, so the higher gate was costing
    # a clean signal without buying any safety.
    if doc.word_count < 180:
        return Signal("stance.authorial_absence", "stance", 0.0, [], 0.0)

    n = doc.word_count
    person = len(_RX_FIRST_SECOND.findall(doc.prose))
    meta = len(_RX_META.findall(doc.prose))
    questions = doc.prose.count("?")
    presence = person + meta + questions
    density = presence * 1000.0 / n

    # Measured on the corpus: humans run 64-89 presence markers per 1k words,
    # machine prose 19-47, marker-free specs 0-5.
    #
    # This was previously a cliff - exactly zero scored strongly, anything else
    # scored 0.00. A rewritten spec with a single "I" in 188 words (5.3/1k, over
    # 12x below the lowest human) therefore scored nothing. One pronoun defeated
    # the signal, which is not a property worth keeping in a detector anyone
    # might try to evade. It is now continuous.
    length_scale = max(0.0, min((n - 180) / 820.0, 1.0))

    # Humans writing technical specifications use almost no first or second
    # person - it is the register's norm, not a sign of machine authorship.
    # Measured: two human-written specs score 0.0 and 12.0 presence per 1k,
    # against 64-89 for the same authors' informal prose. Undamped, this
    # signal was the single largest contributor to a confirmed false positive
    # on a human-written spec (+1.14 of an 88.1% score). It is damped, not
    # removed, because genuine machine specs still sit at the same floor.
    damping = _register_damping(doc)

    if density < 8.0:
        # Essentially authorless. A stray pronoun tapers the score, it does not
        # cancel it.
        lo = (0.50 + 1.00 * length_scale) * (1.0 - 0.25 * (density / 8.0)) * damping
        note = ("no first/second person, no hedges or asides, no questions"
                if presence == 0 else
                f"near-zero authorial presence ({presence} marker(s), "
                f"{density:.1f}/1k)")
        return Signal("stance.authorial_absence", "stance", lo,
                      [f"{note} across {n} words"], density)

    if density >= 55.0:
        return Signal(
            "stance.authorial_absence", "stance",
            -saturating((density - 55.0) / 15.0, 2.0, 1.4),
            [f"{density:.0f} presence markers per 1k ({person} person, "
             f"{meta} hedges, {questions} questions)"], density)

    # Between the two: declining positive evidence, deliberately mild, because
    # formal human writing legitimately lives here.
    lo = (55.0 - density) / 47.0 * 0.45 * damping
    return Signal("stance.authorial_absence", "stance", lo,
                  [f"low authorial presence ({density:.0f}/1k; "
                   f"human samples run 64-89)"], density)


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
