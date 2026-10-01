"""Layer 4 - source-code specific signals (research section 6).

Applied only when the input looks like code. The strongest tells are not the
presence of good practice but the *uniformity* of it, and the absence of human
traces: no commented-out experiments, no HACK/XXX, no dead code, no in-jokes.
"""

from __future__ import annotations

import re
from typing import List

from .signals import Signal, saturating
from .text import Document

GENERIC_NAMES = {
    "result", "results", "data", "items", "item", "value", "values", "temp",
    "tmp", "output", "input", "processed_data", "final_result", "my_list",
    "my_dict", "obj", "res", "ret", "arr", "lst",
}

_RX_DEF = re.compile(r"^\s*(?:def|function|fn|func)\s+(\w+)", re.MULTILINE)
_RX_PY_DOCSTRING = re.compile(r'^\s*(?:"""|\'\'\')', re.MULTILINE)
_RX_COMMENT = re.compile(r"^\s*(?://|#)\s*(\S.*)$", re.MULTILINE)
_RX_HUMAN_CODE = re.compile(
    r"\b(?:TODO|FIXME|HACK|XXX|WTF|NOTE TO SELF|kludge|wip|temporary hack|"
    r"don't ask|not sure why|this is ugly|refactor later)\b", re.IGNORECASE)
_RX_COMMENTED_CODE = re.compile(
    r"^\s*(?://|#)\s*(?:[\w.]+\s*=|if\s|for\s|while\s|return\s|print\(|console\.)",
    re.MULTILINE)
_RX_MAIN_DEMO = re.compile(
    r'if\s+__name__\s*==\s*[\'"]__main__[\'"]|# Example usage|// Example usage',
    re.IGNORECASE)
_RX_EMOJI_PRINT = re.compile(
    r"(?:print|console\.log|echo)\s*\(\s*[\"'f]*[^\"'\n]*"
    r"[✅❌✨⚠\U0001F300-\U0001FAFF]")
_RX_TYPEHINT = re.compile(r"def\s+\w+\s*\([^)]*:\s*\w+[^)]*\)\s*->")
_RX_BROAD_EXCEPT = re.compile(r"except\s+Exception\s*(?:as\s+\w+)?\s*:|catch\s*\(\s*\w*\s*\)")

CODE_HINTS = re.compile(
    r"^\s*(?:def |class |function |import |from \w+ import|package |public |"
    r"const |let |var |#include|using namespace)", re.MULTILINE)


def looks_like_code(text: str) -> bool:
    """Heuristic gate. Avoids running code signals over prose."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    hints = len(CODE_HINTS.findall(text))
    braces = text.count("{") + text.count("}") + text.count("();")
    indented = sum(1 for l in lines if l.startswith(("    ", "\t")))
    return hints >= 2 or (indented > len(lines) * 0.3 and (hints or braces > 4))


def analyse(doc: Document) -> List[Signal]:
    if not looks_like_code(doc.raw):
        return []
    return [
        _docstring_uniformity(doc),
        _comment_style(doc),
        _generic_identifiers(doc),
        _missing_human_traces(doc),
        _appended_demo(doc),
        _emoji_output(doc),
        _defensive_uniformity(doc),
    ]


def _docstring_uniformity(doc: Document) -> Signal:
    defs = _RX_DEF.findall(doc.raw)
    if len(defs) < 3:
        return Signal("code.docstring_uniformity", "code", 0.0, [], 0.0)
    docstrings = len(_RX_PY_DOCSTRING.findall(doc.raw))
    ratio = min(docstrings / float(len(defs)), 1.5)
    if ratio >= 0.95:
        return Signal("code.docstring_uniformity", "code",
                      saturating(len(defs), 4.0, 0.9),
                      [f"docstring on {docstrings}/{len(defs)} definitions"], ratio)
    return Signal("code.docstring_uniformity", "code", 0.0,
                  [f"docstring on {docstrings}/{len(defs)} definitions"], ratio)


def _comment_style(doc: Document) -> Signal:
    comments = _RX_COMMENT.findall(doc.raw)
    code_lines = [l for l in doc.lines if l.strip() and not l.strip().startswith(("#", "//"))]
    if len(code_lines) < 15:
        return Signal("code.comment_density", "code", 0.0, [], 0.0)
    density = len(comments) / float(len(code_lines))

    # Restating comments: "# increment the counter", "# Initialize the list"
    restating = sum(
        1 for c in comments
        if re.match(r"^(?:Initialize|Increment|Set|Get|Return|Create|Define|Loop|"
                    r"Check|Add|Remove|Update|Print|Import|Call)\s+(?:the|a|an)?\s*\w+",
                    c, re.IGNORECASE))
    lo = 0.0
    ev = [f"comment density={density:.2f}"]
    if density > 0.28:
        lo += min((density - 0.28) * 2.0, 0.6)
        ev.append("unusually heavy commenting")
    if restating >= 3:
        lo += saturating(restating, 3.0, 0.8)
        ev.append(f"{restating} comments restate the code rather than explain it")
    return Signal("code.comment_density", "code", lo, ev, density)


def _generic_identifiers(doc: Document) -> Signal:
    names = set(re.findall(r"\b([a-z_][a-z0-9_]{2,})\s*=", doc.raw))
    if len(names) < 5:
        return Signal("code.generic_identifiers", "code", 0.0, [], 0.0)
    generic = names & GENERIC_NAMES
    ratio = len(generic) / float(len(names))
    if ratio < 0.2:
        return Signal("code.generic_identifiers", "code", 0.0, [], ratio)
    return Signal("code.generic_identifiers", "code",
                  min(ratio * 1.6, 0.7),
                  [f"generic names: {', '.join(sorted(generic))}"], ratio)


def _missing_human_traces(doc: Document) -> Signal:
    if len(doc.lines) < 40:
        return Signal("code.human_traces", "code", 0.0, [], 0.0)
    traces = len(_RX_HUMAN_CODE.findall(doc.raw)) + len(_RX_COMMENTED_CODE.findall(doc.raw))
    if traces == 0:
        return Signal("code.human_traces", "code", 0.7,
                      [f"no TODO/FIXME/HACK or commented-out code in {len(doc.lines)} lines"], 0.0)
    return Signal("code.human_traces", "code", -saturating(traces, 2.0, 1.1),
                  [f"human traces found x{traces}"], float(traces))


def _appended_demo(doc: Document) -> Signal:
    hits = _RX_MAIN_DEMO.findall(doc.raw)
    if not hits:
        return Signal("code.appended_demo", "code", 0.0, [], 0.0)
    return Signal("code.appended_demo", "code", 0.35,
                  ["appended runnable demo / example-usage block"], float(len(hits)))


def _emoji_output(doc: Document) -> Signal:
    hits = _RX_EMOJI_PRINT.findall(doc.raw)
    if not hits:
        return Signal("code.emoji_output", "code", 0.0, [], 0.0)
    return Signal("code.emoji_output", "code", saturating(len(hits), 1.5, 1.0),
                  [f"emoji in print/log output x{len(hits)}"], float(len(hits)))


def _defensive_uniformity(doc: Document) -> Signal:
    broad = len(_RX_BROAD_EXCEPT.findall(doc.raw))
    defs = len(_RX_DEF.findall(doc.raw))
    if defs < 3 or broad < 2:
        return Signal("code.defensive_uniformity", "code", 0.0, [], 0.0)
    ratio = broad / float(defs)
    if ratio < 0.5:
        return Signal("code.defensive_uniformity", "code", 0.0, [], ratio)
    return Signal("code.defensive_uniformity", "code", min(ratio * 0.7, 0.6),
                  [f"broad try/except in {broad} of {defs} definitions"], ratio)
