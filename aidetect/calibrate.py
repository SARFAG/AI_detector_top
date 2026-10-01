"""Fit signal weights from your own labelled data.

The weights shipped in this repo are hand-set priors derived from the research
in docs/RESEARCH.md. They are a starting point, not a calibration. Detector
accuracy is dominated by domain: weights tuned on student essays will misfire on
technical documentation, and vice versa.

This module fits a logistic regression over the per-signal log-odds using plain
gradient descent, with no third-party dependencies.

Usage:

    python3 -m aidetect.calibrate --human samples/human --machine samples/machine

Collect at least ~200 documents per class from YOUR domain before trusting the
output. Fewer than that and you are fitting noise.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple

from .detector import analyse
from .signals import sigmoid


def extract_features(text: str) -> Dict[str, float]:
    """Per-signal log-odds, used as the feature vector for fitting."""
    report = analyse(text)
    return {s.name: s.logodds for s in report.signals}


def build_dataset(human_dir: str, machine_dir: str) -> Tuple[List[Dict[str, float]], List[int], List[str]]:
    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    for directory, label in ((human_dir, 0), (machine_dir, 1)):
        for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            if len(text.split()) < 50:
                continue
            rows.append(extract_features(text))
            labels.append(label)
    names = sorted({k for r in rows for k in r})
    return rows, labels, names


def fit(rows, labels, names, epochs=4000, lr=0.05, l2=0.01, seed=0):
    """Logistic regression by gradient descent. Returns (weights, bias)."""
    random.seed(seed)
    w = {n: 0.0 for n in names}
    b = 0.0
    n = len(rows)
    if n == 0:
        raise ValueError("no training rows")

    for _ in range(epochs):
        gw = {k: 0.0 for k in names}
        gb = 0.0
        for row, y in zip(rows, labels):
            z = b + sum(w[k] * row.get(k, 0.0) for k in names)
            err = sigmoid(z) - y
            gb += err
            for k in names:
                v = row.get(k, 0.0)
                if v:
                    gw[k] += err * v
        b -= lr * gb / n
        for k in names:
            # L2 shrinks unsupported signals toward zero rather than letting a
            # tiny sample invent large weights.
            w[k] -= lr * (gw[k] / n + l2 * w[k])
    return w, b


def evaluate(rows, labels, w, b) -> Dict[str, float]:
    correct = 0
    loss = 0.0
    scores = []
    for row, y in zip(rows, labels):
        z = b + sum(w.get(k, 0.0) * v for k, v in row.items())
        p = sigmoid(z)
        scores.append((p, y))
        correct += int((p >= 0.5) == bool(y))
        loss -= math.log(max(p if y else 1 - p, 1e-12))

    # AUROC by pairwise comparison; fine at these dataset sizes.
    pos = [p for p, y in scores if y == 1]
    neg = [p for p, y in scores if y == 0]
    auroc = 0.0
    if pos and neg:
        wins = sum((a > b_) + 0.5 * (a == b_) for a in pos for b_ in neg)
        auroc = wins / (len(pos) * len(neg))

    return {
        "accuracy": correct / len(rows),
        "log_loss": loss / len(rows),
        "auroc": auroc,
        "n": len(rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aidetect.calibrate")
    ap.add_argument("--human", required=True, help="directory of human .txt files")
    ap.add_argument("--machine", required=True, help="directory of machine .txt files")
    ap.add_argument("--out", default="weights.json")
    ap.add_argument("--epochs", type=int, default=4000)
    args = ap.parse_args(argv)

    rows, labels, names = build_dataset(args.human, args.machine)
    if len(rows) < 4:
        print("aidetect.calibrate: need at least 4 documents", file=sys.stderr)
        return 2

    n_h = labels.count(0)
    n_m = labels.count(1)
    print(f"loaded {len(rows)} documents ({n_h} human, {n_m} machine), "
          f"{len(names)} signals")
    if len(rows) < 50:
        print("WARNING: fewer than 50 documents. These weights are fitting noise.\n"
              "         Collect ~200 per class from your own domain before using them.")

    w, b = fit(rows, labels, names, epochs=args.epochs)
    metrics = evaluate(rows, labels, w, b)

    print("\nin-sample metrics (NOT held out - expect optimism):")
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}" if isinstance(v, float) else f"  {k:<10} {v}")

    print("\ntop weights:")
    for name, weight in sorted(w.items(), key=lambda kv: -abs(kv[1]))[:12]:
        print(f"  {name:<36} {weight:+.3f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"bias": b, "weights": w, "metrics": metrics}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
