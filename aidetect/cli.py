"""Command-line interface.

    python -m aidetect essay.txt
    python -m aidetect --json essay.txt
    cat essay.txt | python -m aidetect
    python -m aidetect --segments mixed.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .detector import Report, analyse, analyse_segments, fraction_ai
from .triage import classify

BAR_WIDTH = 40


def _bar(p: float) -> str:
    filled = int(round(p * BAR_WIDTH))
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


def format_report(report: Report, path: str = "", show_all: bool = False) -> str:
    lines: List[str] = []
    header = f"=== {path} ===" if path else "=== analysis ==="
    lines.append(header)
    lines.append("")
    lines.append(f"  machine-generated probability : {report.probability:6.1%}")
    lines.append(f"  {_bar(report.probability)}")
    lines.append(f"  verdict                       : {report.verdict}")
    lines.append(f"  confidence                    : {report.confidence}")
    lines.append(f"  words analysed                : {report.word_count}")
    lines.append("")

    for note in report.notes:
        lines.append(f"  ! {note}")
    if report.notes:
        lines.append("")

    machine = report.machine_evidence()
    human = report.human_evidence()

    if machine:
        lines.append("  evidence FOR machine authorship:")
        for s in (machine if show_all else machine[:8]):
            lines.append(f"    {s}")
        lines.append("")
    if human:
        lines.append("  evidence FOR human authorship:")
        for s in (human if show_all else human[:6]):
            lines.append(f"    {s}")
        lines.append("")

    if report.attribution:
        lines.append("  possible source model (weak evidence, drifts per release):")
        for model, score, ev in report.attribution[:3]:
            lines.append(f"    {model:<14} score={score:<6} {'; '.join(ev[:3])}")
        lines.append("")

    if report.layer_totals:
        parts = [f"{k}={v:+.2f}" for k, v in sorted(report.layer_totals.items())]
        lines.append("  layer log-odds: " + "  ".join(parts))
        lines.append(f"  total log-odds: {report.total_logodds:+.2f}")
    lines.append("")
    lines.append("  Scores are evidence for a human to weigh, not a verdict.")
    lines.append("  See docs/LIMITS.md before acting on this.")
    return "\n".join(lines)


def report_to_dict(report: Report) -> dict:
    return {
        "probability": round(report.probability, 4),
        "verdict": report.verdict,
        "confidence": report.confidence,
        "word_count": report.word_count,
        "total_logodds": round(report.total_logodds, 3),
        "layer_totals": {k: round(v, 3) for k, v in report.layer_totals.items()},
        "notes": report.notes,
        "attribution": [
            {"model": m, "score": s, "evidence": e} for m, s, e in report.attribution
        ],
        "signals": [
            {
                "name": s.name,
                "layer": s.layer,
                "logodds": round(s.logodds, 3),
                "direction": s.direction,
                "detail": round(s.detail, 4),
                "evidence": s.evidence,
            }
            for s in report.top_signals
        ],
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aidetect",
        description="Detect machine-generated text by layered evidence.",
    )
    ap.add_argument("paths", nargs="*", help="files to analyse (default: stdin)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--all", action="store_true", help="show every signal")
    ap.add_argument("--segments", action="store_true",
                    help="score paragraph windows to localise machine-written spans")
    ap.add_argument("--triage", action="store_true",
                    help="three-class verdict (human / ai_assisted / ai) with "
                         "machine span offsets")
    ap.add_argument("--fraction", action="store_true",
                    help="also score fixed windows and report the fraction above threshold")
    ap.add_argument("--threshold", type=float, default=None,
                    help="exit 1 if probability >= THRESHOLD (for CI gates)")
    args = ap.parse_args(argv)

    inputs: List[tuple] = []
    if args.paths:
        for p in args.paths:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    inputs.append((p, fh.read()))
            except OSError as exc:
                print(f"aidetect: cannot read {p}: {exc}", file=sys.stderr)
                return 2
    else:
        inputs.append(("<stdin>", sys.stdin.read()))

    exceeded = False
    results = []

    for path, text in inputs:
        if args.segments:
            segs = analyse_segments(text)
            if args.json:
                results.append({
                    "path": path,
                    "segments": [
                        {"probability": round(p, 4), "excerpt": c[:160]}
                        for c, p in segs
                    ],
                })
            else:
                print(f"=== {path} (segment scan) ===\n")
                for chunk, p in segs:
                    flag = "  <-- MACHINE-LEANING" if p >= 0.70 else ""
                    excerpt = " ".join(chunk.split())[:90]
                    print(f"  {p:6.1%} {_bar(p)}{flag}")
                    print(f"         {excerpt}...\n")
            continue

        if args.triage:
            t = classify(text)
            if args.json:
                results.append({
                    "path": path, "prediction": t.prediction,
                    "fraction_ai": round(t.fraction_ai, 4),
                    "fraction_assisted": round(t.fraction_assisted, 4),
                    "word_count": t.word_count,
                    "machine_spans": [{"start": a, "end": b} for a, b in t.machine_spans],
                })
            else:
                print(f"=== {path} (triage) ===\n")
                print(f"  prediction          : {t.prediction}")
                print(f"  fraction_ai         : {t.fraction_ai:.2f}")
                print(f"  fraction_assisted   : {t.fraction_assisted:.2f}")
                print(f"  words               : {t.word_count}")
                if t.machine_spans:
                    print(f"  machine spans       : {len(t.machine_spans)}")
                    for (a, b), ex in zip(t.machine_spans, t.excerpt_spans(text, 5)):
                        print(f"    words {a:>5}-{b:<5} {ex[:62]}")
                if t.note:
                    print(f"  note                : {t.note}")
                print()
            continue

        report = analyse(text)
        frac = None
        if args.fraction:
            frac, win = fraction_ai(text)
        if args.threshold is not None and report.probability >= args.threshold:
            exceeded = True
        if args.json:
            d = report_to_dict(report)
            d["path"] = path
            if frac is not None:
                d["fraction_ai"] = round(frac, 4)
                d["window_scores"] = [round(p, 4) for p in win]
            results.append(d)
        else:
            print(format_report(report, path, show_all=args.all))
            if frac is not None:
                print(f"  fraction of windows scored machine : {frac:.0%} "
                      f"({len(win)} windows of {len(text.split())} words)")
            print()

    if args.json:
        json.dump(results if len(results) != 1 else results[0],
                  sys.stdout, indent=2, ensure_ascii=False)
        print()

    return 1 if exceeded else 0


if __name__ == "__main__":
    raise SystemExit(main())
