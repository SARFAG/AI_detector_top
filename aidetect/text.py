"""Dependency-free tokenisation, sentence and paragraph splitting.

Deliberately simple and deterministic. Everything downstream measures *shape*
(lengths, variance, repetition), so a perfect linguistic parse is not required —
but it must never crash on adversarial input, and it must be stable so that two
runs over the same text always produce the same score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Abbreviations that must not end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "etc", "vs", "e.g",
    "i.e", "fig", "al", "inc", "ltd", "co", "approx", "dept", "est", "no",
    "vol", "pp", "ed", "eds", "cf", "ca", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Document:
    """A parsed input, computed once and shared by all layers."""

    raw: str
    prose: str                 # raw minus fenced code blocks
    words: List[str]
    sentences: List[str]
    paragraphs: List[str]
    lines: List[str]

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def char_count(self) -> int:
        return len(self.raw)

    @property
    def lower(self) -> str:
        return self._lower

    def __post_init__(self) -> None:
        self._lower = self.prose.lower()


# A line ending in one of these is mid-clause; the next line continues it.
_CONTINUATION_TAIL = re.compile(
    r"(?:[,;:\-\u2013\u2014(\[{]|\b(?:and|or|but|the|a|an|of|to|in|on|for|with|"
    r"that|which|as|at|by|from|into|than|then)\s*)$", re.IGNORECASE)


def _looks_terminal(line: str, nxt: str) -> bool:
    """Is this newline a sentence boundary, even without terminal punctuation?

    Dropping full stops is a trivial way to defeat a punctuation-only splitter:
    unpunctuated lines get glued into one huge pseudo-sentence, which inflates
    length variance and makes machine text read as bursty (i.e. human). Line
    breaks in line-oriented text are real boundaries and are treated as such.

    Guarded so that hard-wrapped prose is not over-split: a line that ends
    mid-clause, or is followed by a lowercase continuation, stays joined.
    """
    s = line.strip()
    if not s:
        return False
    if s[-1] in ".!?\u2026":
        return True          # the regex splitter already handles this
    if len(s.split()) < 3:
        return False
    if _CONTINUATION_TAIL.search(s):
        return False
    if not nxt:
        return True
    return nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in "-*\u2022"


def _split_block(text: str) -> List[str]:
    """Split one block on terminal punctuation, rejoining false breaks."""
    pieces = _SENT_SPLIT_RE.split(text)
    out: List[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if out:
            prev = out[-1]
            # Rejoin "Dr." + "Smith said." and "approx." + "3 units."
            tail = _WORD_RE.findall(prev[-12:].lower())
            ends_abbrev = bool(tail) and tail[-1] in _ABBREV and prev.endswith(".")
            # A single initial, e.g. "J." in "J. R. R. Tolkien".
            single_initial = bool(re.search(r"\b[A-Z]\.$", prev))
            if ends_abbrev or single_initial:
                out[-1] = prev + " " + piece
                continue
        out.append(piece)
    return out


def split_sentences(text: str) -> List[str]:
    """Split into sentences, treating terminal-looking line breaks as boundaries."""
    if not text.strip():
        return []
    lines = text.split("\n")
    blocks: List[str] = []
    current: List[str] = []
    for i, line in enumerate(lines):
        current.append(line)
        nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        if _looks_terminal(line, nxt):
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))

    out: List[str] = []
    for block in blocks:
        out.extend(_split_block(block))
    return out


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def strip_code_blocks(text: str) -> str:
    """Remove fenced code so prose statistics are not polluted by source code."""
    return _CODE_FENCE_RE.sub("\n", text)


def parse(text: str) -> Document:
    prose = strip_code_blocks(text)
    return Document(
        raw=text,
        prose=prose,
        words=tokenize_words(prose),
        sentences=split_sentences(prose),
        paragraphs=split_paragraphs(prose),
        lines=text.splitlines(),
    )


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
