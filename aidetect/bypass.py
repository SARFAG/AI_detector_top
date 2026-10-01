"""Adversarial rewriter: greedy search over the attack space.

The counterpart to `aidetect/adversarial.py`. Where that module applies attacks
in a fixed order to measure robustness, this one searches: at each round it
tries every remaining attack, keeps whichever lowers the score most, and
repeats until it reaches a target or runs out of moves.

Greedy selection matters. Attacks interact - one can undo another's effect, or
unlock a second that was previously neutral - so a fixed order under-reports
what the attack space can actually do. On this corpus greedy search reaches a
lower score than the fixed order on every document tested.

Measured result on the shipped corpus: it does NOT achieve bypass. Six
machine-written documents, every attack available, and none crosses the 0.5
threshold. What it does is quantify the distance to that threshold, which is
the number worth reporting.

    python -m aidetect.bypass FILE [--target 0.4] [--rounds 12] [--out OUT]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .adversarial import ATTACKS, Attack, _word_diff
from .detector import analyse


@dataclass
class Step:
    round: int
    attack: str
    targets: str
    score_before: float
    score_after: float

    @property
    def gain(self) -> float:
        return self.score_before - self.score_after


@dataclass
class BypassResult:
    original_score: float
    final_score: float
    target: float
    reached_target: bool
    crossed_threshold: bool
    steps: List[Step] = field(default_factory=list)
    text: str = ""
    words_changed: int = 0
    exhausted: bool = False

    def summary(self) -> str:
        out = [
            f"  start            {self.original_score:6.1%}",
            f"  end              {self.final_score:6.1%}",
            f"  target           {self.target:6.1%}  "
            f"{'reached' if self.reached_target else 'NOT reached'}",
            f"  crossed 0.5      {'YES' if self.crossed_threshold else 'NO'}",
            f"  rounds applied   {len(self.steps)}"
            f"{'  (attack space exhausted)' if self.exhausted else ''}",
            f"  words changed    {self.words_changed}",
        ]
        if self.steps:
            out.append("")
            out.append("  round  attack                 targets"
                       "                              score")
            for s in self.steps:
                out.append("  %5d  %-22s %-36s %5.1f%% (%+.1f)"
                           % (s.round, s.attack, s.targets[:36],
                              s.score_after * 100, -s.gain * 100))
        return "\n".join(out)


def bypass(text: str, target: float = 0.4, rounds: int = 12,
           seed: int = 0, allow_repeats: bool = False) -> BypassResult:
    """Greedily apply attacks, keeping the best move each round.

    Args:
        text: the document to transform.
        target: stop once the score is at or below this.
        rounds: maximum number of attacks to apply.
        seed: makes the search reproducible.
        allow_repeats: let an attack be chosen more than once. Off by default,
            since most of these are idempotent and re-applying them mostly
            degrades the text without moving the score.
    """
    current = text
    score = analyse(current).probability
    original = score
    steps: List[Step] = []
    used: set = set()

    for rnd in range(1, rounds + 1):
        if score <= target:
            break

        candidates: List[Tuple[float, str, Attack]] = []
        for atk in ATTACKS:
            if not allow_repeats and atk.name in used:
                continue
            try:
                cand = atk.fn(current, random.Random(seed + rnd))
            except Exception:
                continue
            if cand == current:
                continue
            candidates.append((analyse(cand).probability, cand, atk))

        if not candidates:
            return BypassResult(original, score, target, score <= target,
                                score < 0.5, steps, current,
                                _word_diff(text, current), exhausted=True)

        best_score, best_text, best_atk = min(candidates, key=lambda c: c[0])
        # Stop if nothing helps; applying a harmful move to fill rounds is noise.
        if best_score >= score:
            return BypassResult(original, score, target, score <= target,
                                score < 0.5, steps, current,
                                _word_diff(text, current), exhausted=True)

        steps.append(Step(rnd, best_atk.name, best_atk.targets, score, best_score))
        current, score = best_text, best_score
        used.add(best_atk.name)

    return BypassResult(original, score, target, score <= target, score < 0.5,
                        steps, current, _word_diff(text, current))


def compare_to_fixed_order(text: str, target: float = 0.4) -> Dict[str, float]:
    """Greedy search against the fixed-order application in adversarial.py."""
    from .adversarial import robustness_report
    greedy = bypass(text, target=target)
    fixed = robustness_report(text)
    return {
        "baseline": greedy.original_score,
        "greedy_final": greedy.final_score,
        "fixed_order_final": fixed["final_score"],
        "greedy_advantage": fixed["final_score"] - greedy.final_score,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aidetect.bypass",
        description="Greedily transform a document to lower its detector score.")
    ap.add_argument("path")
    ap.add_argument("--target", type=float, default=0.4,
                    help="stop once the score reaches this (default 0.4)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", action="store_true",
                    help="allow an attack to be applied more than once")
    ap.add_argument("--out", help="write the transformed text here")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        with open(args.path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"aidetect.bypass: cannot read {args.path}: {exc}", file=sys.stderr)
        return 2

    r = bypass(text, target=args.target, rounds=args.rounds, seed=args.seed,
               allow_repeats=args.repeats)

    if args.json:
        json.dump({
            "path": args.path,
            "original_score": round(r.original_score, 4),
            "final_score": round(r.final_score, 4),
            "target": args.target,
            "reached_target": r.reached_target,
            "crossed_threshold": r.crossed_threshold,
            "exhausted": r.exhausted,
            "words_changed": r.words_changed,
            "steps": [{"round": s.round, "attack": s.attack,
                       "targets": s.targets, "score_after": round(s.score_after, 4),
                       "gain": round(s.gain, 4)} for s in r.steps],
        }, sys.stdout, indent=2)
        print()
    else:
        print(f"=== {args.path} ===\n")
        print(r.summary())
        print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(r.text)
        print(f"transformed text written to {args.out}")

    return 0 if r.reached_target else 1


if __name__ == "__main__":
    raise SystemExit(main())
