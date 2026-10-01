"""Discourse-level features: drift, reuse, readability variation, sentence mix.

These measure how a text moves rather than what it says. The common thread is
that people are uneven across a document - they wander off topic, spend
paragraphs of wildly different difficulty, and mix statements with questions
and asides - while generated text holds a steady line.

Built as plain measurements first. Which of them earn a place in the ensemble
is decided by measuring separation on the corpus, not by assuming.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from .signals import Signal
from .text import Document, mean, stdev, tokenize_words

_STOP = set("""the of and a to in is be that was for it with as his on at by i
this had not are but from or have an they which one you were her all she there
would their we him been has when who will no more if out so what up its about
into than them can only other some could these two may then do any my now such
like our over me even most made after also did many before must through back
where much your way well down should because each just those how too very make
still own see work long get here between both being under never same another
know while last might us great old off come since against go came right used
take""".split())


def analyse(doc: Document) -> List[Signal]:
    """Only sentence_type_mix is scored.

    topic_drift, rare_word_reuse and readability_variance were built and
    measured on the corpus and did NOT separate the classes: drift overlapped
    (.008-.043 human vs .014-.083 machine), one of three humans sat inside the
    machine range on rare-word reuse, and readability variance overlapped
    completely (7.4-18.4 vs 9.6-13.6). They remain here as measurements, with
    zero weight, so the negative result is recorded rather than quietly
    dropped - and so a larger corpus can re-test them cheaply.
    """
    return [
        _sentence_type_mix(doc),
        _topic_drift(doc),
        _rare_word_reuse(doc),
        _readability_variance(doc),
    ]


def _content_words(text: str) -> set:
    return {w.lower() for w in tokenize_words(text)
            if len(w) > 3 and w.lower() not in _STOP}


def _topic_drift(doc: Document) -> Signal:
    """Lexical overlap between adjacent paragraphs.

    People wander: they finish a thought, start somewhere else, circle back.
    Generated text holds the thread, so adjacent blocks share more vocabulary
    and share it more evenly. Both the mean overlap and its variance matter -
    a human's overlap is erratic.
    """
    paras = [p for p in doc.paragraphs if len(tokenize_words(p)) >= 25]
    if len(paras) < 4:
        return Signal("discourse.topic_drift", "discourse", 0.0, [], 0.0)

    overlaps = []
    for a, b in zip(paras, paras[1:]):
        sa, sb = _content_words(a), _content_words(b)
        if not sa or not sb:
            continue
        overlaps.append(len(sa & sb) / len(sa | sb))
    if len(overlaps) < 3:
        return Signal("discourse.topic_drift", "discourse", 0.0, [], 0.0)

    m = mean(overlaps)
    cv = stdev(overlaps) / m if m else 0.0
    return Signal("discourse.topic_drift", "discourse", 0.0,
                  [f"adjacent-paragraph overlap mean={m:.3f} cv={cv:.2f} "
                   f"(n={len(overlaps)})"], m)


def _rare_word_reuse(doc: Document) -> Signal:
    """Share of uncommon words the text uses more than once.

    Models reach for an unusual word and then reuse it; people tend to use a
    striking word once and move on.
    """
    words = [w.lower() for w in doc.words if len(w) > 6 and w.lower() not in _STOP]
    if len(words) < 50:
        return Signal("discourse.rare_word_reuse", "discourse", 0.0, [], 0.0)
    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    reused = sum(1 for c in counts.values() if c > 1)
    rate = reused / len(counts)
    return Signal("discourse.rare_word_reuse", "discourse", 0.0,
                  [f"{rate:.3f} of long words reused ({reused}/{len(counts)})"],
                  rate)


def _syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def _readability_variance(doc: Document) -> Signal:
    """Variation in reading difficulty across paragraphs.

    We already measure lexical diversity; this measures whether the *density*
    of the writing changes. People write an easy paragraph then a dense one.
    """
    paras = [p for p in doc.paragraphs if len(tokenize_words(p)) >= 30]
    if len(paras) < 4:
        return Signal("discourse.readability_variance", "discourse", 0.0, [], 0.0)

    scores = []
    for p in paras:
        words = tokenize_words(p)
        sents = max(len(re.findall(r"[.!?]+", p)), 1)
        if not words:
            continue
        # Flesch reading ease, standard coefficients.
        asl = len(words) / sents
        asw = mean([_syllables(w) for w in words])
        scores.append(206.835 - 1.015 * asl - 84.6 * asw)
    if len(scores) < 4:
        return Signal("discourse.readability_variance", "discourse", 0.0, [], 0.0)

    sd = stdev(scores)
    return Signal("discourse.readability_variance", "discourse", 0.0,
                  [f"Flesch sd={sd:.1f} across {len(scores)} paragraphs "
                   f"(mean {mean(scores):.0f})"], sd)


def _sentence_type_mix(doc: Document) -> Signal:
    """Proportion of non-declarative sentences.

    Questions, exclamations and bare fragments are how a person varies pace.
    Generated expository text is overwhelmingly declarative.
    """
    sents = doc.sentences
    if len(sents) < 10:
        return Signal("discourse.sentence_type_mix", "discourse", 0.0, [], 0.0)
    q = sum(1 for s in sents if s.rstrip().endswith("?"))
    ex = sum(1 for s in sents if s.rstrip().endswith("!"))
    frag = sum(1 for s in sents if len(tokenize_words(s)) <= 4)
    nondecl = (q + ex + frag) / len(sents)
    # Human samples run .242-.458; the marker-free specs .000-.091. Machine
    # prose overlaps the low human range, so this is weighted modestly and
    # only credits clear cases in either direction.
    if nondecl >= 0.24:
        lo = -min((nondecl - 0.24) * 2.5, 0.6)
    elif nondecl <= 0.10:
        from .register import damping as _register_damping
        lo = min((0.10 - nondecl) * 4.0, 0.45) * _register_damping(doc)
    else:
        lo = 0.0
    return Signal("discourse.sentence_type_mix", "discourse", lo,
                  [f"{nondecl:.3f} non-declarative "
                   f"({q} questions, {ex} exclamations, {frag} fragments)"],
                  nondecl)
