"""Adversarial robustness: attack the detector with your own labelled data.

The point is to find out which signals are fragile, and to manufacture hard
labelled examples when real ones are scarce. Each perturbation targets one
named signal, so when a document flips you know exactly what carried it and
what to fix.

Scope, deliberately: this operates on documents whose label you already know,
and its outputs are a robustness metric plus perturbed documents that KEEP
their original label (machine text stays machine text - roughening the surface
does not change who wrote it). That is what adversarial training needs. It is
not a service for making a particular document pass, and the report is the
deliverable rather than the rewritten text.

Usage:

    from aidetect.adversarial import robustness_report, ATTACKS
    rep = robustness_report(text)
    print(rep["flipped_by"], rep["attack_cost"])
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .detector import analyse
from .text import parse, tokenize_words

Transform = Callable[[str, random.Random], str]


@dataclass
class Attack:
    """One perturbation, named for the signal it is meant to defeat."""
    name: str
    targets: str
    describe: str
    fn: Transform


# --------------------------------------------------------------------------
# Perturbations, each aimed at a specific signal
# --------------------------------------------------------------------------

_ASIDES = [
    "I think", "I'd say", "we probably want", "you might disagree, but",
    "honestly", "I'm not certain here", "to be fair", "my feeling is",
]


def _add_authorial_presence(text: str, rng: random.Random) -> str:
    """Target: stance.authorial_absence. Insert first-person asides."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 6:
        return text
    idxs = rng.sample(range(len(sents)), max(1, len(sents) // 8))
    for i in idxs:
        if len(sents[i].split()) > 6:
            sents[i] = rng.choice(_ASIDES) + ", " + sents[i][0].lower() + sents[i][1:]
    return " ".join(sents)


_TYPO_RULES = [
    (r"\bits (own|applied|value|status)\b", r"it's \1"),
    (r"\bdo not\b", "dont"),
    (r"\bcannot\b", "cant"),
    (r"\bthat is\b", "thats"),
]


def _inject_typos(text: str, rng: random.Random) -> str:
    """Target: forensics.typo_absence. Introduce human-shaped errors."""
    out = text
    for pat, rep in _TYPO_RULES:
        m = list(re.finditer(pat, out, re.I))
        if m:
            pick = rng.choice(m)
            out = out[:pick.start()] + re.sub(pat, rep, pick.group(0), flags=re.I) \
                + out[pick.end():]
    # A stray double space, which real typists produce.
    parts = out.split(". ")
    if len(parts) > 3:
        i = rng.randrange(1, len(parts) - 1)
        parts[i] = parts[i] + " "
    return ". ".join(parts)


def _vary_sentence_length(text: str, rng: random.Random) -> str:
    """Target: statistical.burstiness. Split long sentences, add short ones."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for s in sents:
        w = s.split()
        if len(w) > 26 and rng.random() < 0.6:
            # Break at a comma near the middle.
            mid = len(w) // 2
            for off in range(0, 7):
                for j in (mid - off, mid + off):
                    if 4 < j < len(w) - 4 and w[j].endswith(","):
                        out.append(" ".join(w[:j + 1]).rstrip(",") + ".")
                        out.append(" ".join(w[j + 1:]).capitalize())
                        break
                else:
                    continue
                break
            else:
                out.append(s)
        else:
            out.append(s)
    return " ".join(out)


def _vary_paragraphs(text: str, rng: random.Random) -> str:
    """Target: statistical.paragraph_uniformity. Make block sizes uneven."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 3:
        return text
    out: List[str] = []
    for p in paras:
        sents = re.split(r"(?<=[.!?])\s+", p)
        if len(sents) > 3 and rng.random() < 0.5:
            cut = rng.randrange(1, len(sents))
            out.append(" ".join(sents[:cut]))
            out.append(" ".join(sents[cut:]))
        else:
            out.append(p)
    return "\n\n".join(out)


def _add_questions(text: str, rng: random.Random) -> str:
    """Target: discourse.sentence_type_mix. Non-declarative sentences."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 2:
        return text
    i = rng.randrange(len(paras))
    paras[i] = "So what does that mean in practice? " + paras[i]
    if len(paras) > 3:
        j = rng.randrange(len(paras))
        paras[j] = paras[j] + " Worth thinking about."
    return "\n\n".join(paras)


def _contract(text: str, rng: random.Random) -> str:
    """Target: statistical.contraction_balance."""
    pairs = [(r"\bit is\b", "it's"), (r"\bthere is\b", "there's"),
             (r"\bdoes not\b", "doesn't"), (r"\bis not\b", "isn't"),
             (r"\bwill not\b", "won't"), (r"\bcannot\b", "can't"),
             (r"\bwe will\b", "we'll"), (r"\bthat is\b", "that's")]
    out = text
    for pat, rep in pairs:
        out = re.sub(pat, rep, out, flags=re.I)
    return out


def _cluster_identifiers(text: str, rng: random.Random) -> str:
    """Target: structural.identifier_dispersion.

    Move identifier-dense sentences toward the end, imitating the human habit
    of separating what-I-want from what-to-call-it.
    """
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 4:
        return text
    ident = re.compile(r"\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z_]+\b")
    dense, plain = [], []
    for p in paras:
        (dense if len(ident.findall(p)) >= 3 else plain).append(p)
    if not dense or not plain:
        return text
    return "\n\n".join(plain + dense)


def _strip_markup(text: str, rng: random.Random) -> str:
    """Target: forensics.markdown_density."""
    out = re.sub(r"`", "", text)
    out = re.sub(r"^\s*[-*]\s+", "", out, flags=re.MULTILINE)
    out = re.sub(r"^\s*#+\s*", "", out, flags=re.MULTILINE)
    return out


# Plain substitutes for the LLM register. This is the one attack that touches
# vocabulary rather than surface form, and without it the suite cannot reach
# the lexical layer at all - which is what carries most marketing and
# explainer prose.
_REGISTER_SWAPS = {
    "delve into": "look at", "delve": "look", "delves": "looks",
    "intricate": "complex", "intricacies": "details", "tapestry": "mix",
    "realm": "area", "underscores": "shows", "underscore": "show",
    "pivotal": "key", "multifaceted": "varied", "nuanced": "subtle",
    "meticulous": "careful", "meticulously": "carefully", "garner": "get",
    "garnered": "got", "showcasing": "showing", "ever-evolving": "changing",
    "myriad": "many", "paramount": "vital", "leverage": "use",
    "leveraging": "using", "robust": "solid", "foster": "build",
    "fostering": "building", "holistic": "overall", "seamless": "smooth",
    "seamlessly": "smoothly", "streamline": "simplify", "empower": "let",
    "harness": "use", "harnessing": "using", "crucial": "important",
    "facilitate": "help", "transformative": "major",
    "comprehensive": "full", "invaluable": "useful", "profound": "deep",
    "it's worth noting that": "note that", "it is worth noting that": "note that",
    "it's important to note that": "note that",
    "plays a crucial role": "matters", "plays a vital role": "matters",
    "a testament to": "a sign of", "in the realm of": "in",
    "navigate the complexities": "handle the complexity",
    "in today's fast-paced world": "these days",
    "a comprehensive overview": "an overview",
    "let's dive in": "here goes", "let's delve": "let's look",
    "here's a breakdown": "here's how it works",
    "in conclusion": "so", "in summary": "so", "first and foremost": "first",
    "i hope this helps": "", "let me know if you'd like": "",
    "certainly!": "", "great question!": "",
}


def _neutralise_register(text: str, rng: random.Random) -> str:
    """Target: lexical.tier1_register / tier3_phrases.

    Swap the LLM register for plain words. The only attack here that changes
    vocabulary; everything else is surface.
    """
    out = text
    for marker, plain in sorted(_REGISTER_SWAPS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(r"\b" + re.escape(marker) + r"\b", plain, out,
                     flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", out)


ATTACKS: List[Attack] = [
    Attack("neutralise_register", "lexical.tier1_register / lexical.tier3_phrases",
           "swap LLM register vocabulary for plain words", _neutralise_register),
    Attack("authorial_presence", "stance.authorial_absence",
           "insert first-person asides and hedges", _add_authorial_presence),
    Attack("typos", "forensics.typo_absence",
           "introduce human-shaped errors", _inject_typos),
    Attack("sentence_variance", "statistical.burstiness",
           "split long sentences to raise length variance", _vary_sentence_length),
    Attack("paragraph_variance", "statistical.paragraph_uniformity",
           "make paragraph sizes uneven", _vary_paragraphs),
    Attack("questions", "discourse.sentence_type_mix",
           "add non-declarative sentences", _add_questions),
    Attack("contractions", "statistical.contraction_balance",
           "contract auxiliary verbs", _contract),
    Attack("cluster_identifiers", "structural.identifier_dispersion",
           "move identifier-dense blocks to the end", _cluster_identifiers),
    Attack("strip_markup", "forensics.markdown_density",
           "remove backticks, bullets and headings", _strip_markup),
]


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def single_attack_effect(text: str, seed: int = 0) -> List[Dict]:
    """Score each attack applied alone. Ranks signal fragility."""
    base = analyse(text).probability
    rows = []
    for atk in ATTACKS:
        rng = random.Random(seed)
        try:
            perturbed = atk.fn(text, rng)
        except Exception:  # a perturbation must never break the harness
            continue
        p = analyse(perturbed).probability
        rows.append({
            "attack": atk.name, "targets": atk.targets,
            "describe": atk.describe,
            "score_after": p, "drop": base - p,
            "words_changed": _word_diff(text, perturbed),
        })
    rows.sort(key=lambda r: -r["drop"])
    return rows


def _word_diff(a: str, b: str) -> int:
    wa, wb = a.split(), b.split()
    return abs(len(wa) - len(wb)) + sum(1 for x, y in zip(wa, wb) if x != y)


def robustness_report(text: str, threshold: float = 0.5,
                      seed: int = 0) -> Dict:
    """How much perturbation does it take to push this below `threshold`?

    Applies attacks cumulatively in order of individual effectiveness and
    records the trajectory. `attack_cost` is the number of attacks needed;
    None means the document survived all of them.
    """
    base = analyse(text).probability
    singles = single_attack_effect(text, seed=seed)

    current = text
    trajectory: List[Dict] = [{"step": 0, "attack": None, "score": base}]
    cost: Optional[int] = None
    applied: List[str] = []

    for i, row in enumerate(singles, 1):
        atk = next(a for a in ATTACKS if a.name == row["attack"])
        rng = random.Random(seed + i)
        try:
            current = atk.fn(current, rng)
        except Exception:
            continue
        applied.append(atk.name)
        p = analyse(current).probability
        trajectory.append({"step": i, "attack": atk.name, "score": p})
        if p < threshold and cost is None:
            cost = i

    return {
        "baseline": base,
        "threshold": threshold,
        "attack_cost": cost,
        "survived": cost is None,
        "final_score": trajectory[-1]["score"],
        "total_words_changed": _word_diff(text, current),
        "per_attack": singles,
        "trajectory": trajectory,
        "attacks_applied": applied,
        "hardened_text": current,
    }


def hardening_corpus(texts: Sequence[str], seed: int = 0) -> List[Tuple[str, str]]:
    """Perturbed variants of machine documents, for training and evaluation.

    Each variant KEEPS the machine label: roughening the surface of generated
    text does not make it human-written. These are hard negatives - exactly
    the examples a detector tuned only on clean output gets wrong.
    """
    out: List[Tuple[str, str]] = []
    for i, t in enumerate(texts):
        for atk in ATTACKS:
            rng = random.Random(seed + i)
            try:
                out.append((atk.fn(t, rng), f"machine_attacked_{atk.name}"))
            except Exception:
                continue
    return out
