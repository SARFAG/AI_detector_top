"""Evaluation harness: metrics that mean something.

Built because the repo could not answer a basic question - does a change
actually help? - and because the headline margin in the README was fit on the
same documents it was reported against.

Three things matter here:

1. **Out-of-sample scoring.** Leave-one-out cross-validation refits the weights
   without the document being scored. The gap between in-sample and
   out-of-sample is the overfitting, stated as a number.
2. **Uncertainty.** Bootstrap confidence intervals and a permutation test. On a
   small corpus a perfect AUROC is nearly meaningless, and the interval says so
   instead of letting the point estimate flatter us.
3. **FPR at fixed TPR**, not accuracy. "What is our false-positive rate when we
   catch 90% of machine text" is the question a deployment actually faces.

Usage:
    python3 -m aidetect.evaluate --human samples/human --machine samples/machine
    python3 -m aidetect.evaluate --human H --machine M --json --ablate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from .calibrate import build_dataset, evaluate as _fit_metrics, extract_features, fit
from .detector import analyse
from .signals import sigmoid

Scores = Sequence[float]
Labels = Sequence[int]


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------

def auroc(scores: Scores, labels: Labels) -> float:
    """Area under the ROC curve, via the rank/Mann-Whitney identity.

    Ties contribute 0.5, which matters here: a detector that abstains (returns
    0.5) on everything should score 0.5, not 1.0.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def roc_points(scores: Scores, labels: Labels) -> List[Tuple[float, float, float]]:
    """(fpr, tpr, threshold) at every distinct threshold, high to low."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    if not n_pos or not n_neg:
        return []
    out = [(0.0, 0.0, float("inf"))]
    for thr in sorted(set(scores), reverse=True):
        tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
        out.append((fp / n_neg, tp / n_pos, thr))
    return out


def fpr_at_tpr(scores: Scores, labels: Labels, target: float) -> Tuple[float, float]:
    """Lowest false-positive rate achieving at least `target` recall."""
    best = (1.0, 0.0)
    found = False
    for fpr, tpr, thr in roc_points(scores, labels):
        if tpr >= target and (not found or fpr < best[0]):
            best = (fpr, thr)
            found = True
    return best if found else (float("nan"), float("nan"))


def tpr_at_fpr(scores: Scores, labels: Labels, budget: float) -> Tuple[float, float]:
    """Highest recall available while staying within a false-positive budget."""
    best = (0.0, float("inf"))
    for fpr, tpr, thr in roc_points(scores, labels):
        if fpr <= budget and tpr > best[0]:
            best = (tpr, thr)
    return best


def confusion(scores: Scores, labels: Labels, threshold: float = 0.5) -> Dict[str, float]:
    tp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 0)
    tn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 1)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(len(labels), 1)}


def log_loss(scores: Scores, labels: Labels) -> float:
    """Mean negative log-likelihood.

    Unlike AUROC this keeps resolution when separation is already perfect, so
    it can still tell whether a change made the detector *more confident* for
    the right reasons. On a small corpus it is the more informative number.
    """
    import math
    if not labels:
        return float("nan")
    total = 0.0
    for s_, y in zip(scores, labels):
        p = min(max(s_, 1e-12), 1 - 1e-12)
        total -= math.log(p) if y else math.log(1 - p)
    return total / len(labels)


def margin(scores: Scores, labels: Labels) -> float:
    """Lowest machine score minus highest human score.

    Negative means the classes overlap. Also keeps resolution at AUROC 1.0.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    return min(pos) - max(neg)


def brier(scores: Scores, labels: Labels) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    if not labels:
        return float("nan")
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)


def ece(scores: Scores, labels: Labels, bins: int = 10) -> float:
    """Expected calibration error: does 0.8 actually mean 80%?"""
    if not labels:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores)
               if (s >= lo and s < hi) or (b == bins - 1 and s == 1.0)]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += (len(idx) / len(labels)) * abs(conf - acc)
    return total


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def bootstrap_auroc(scores: Scores, labels: Labels, n: int = 2000,
                    seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI. Replicates missing a class are discarded."""
    rng = random.Random(seed)
    k = len(labels)
    vals: List[float] = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        ls = [labels[i] for i in idx]
        if len(set(ls)) < 2:
            continue
        vals.append(auroc([scores[i] for i in idx], ls))
    if not vals:
        return {"mean": float("nan"), "lo95": float("nan"),
                "hi95": float("nan"), "usable": 0}
    vals.sort()
    return {
        "mean": sum(vals) / len(vals),
        "lo95": vals[int(0.025 * len(vals))],
        "hi95": vals[min(int(0.975 * len(vals)), len(vals) - 1)],
        "usable": len(vals),
    }


def permutation_test(scores: Scores, labels: Labels, n: int = 10000,
                     seed: int = 0) -> Dict[str, float]:
    """How often does shuffled labelling match this AUROC?

    With a small corpus the floor on p is 1/C(n, k), so the test can be
    incapable of significance no matter how clean the separation. That floor is
    reported alongside p.
    """
    rng = random.Random(seed)
    observed = auroc(scores, labels)
    shuffled = list(labels)
    hits = 0
    for _ in range(n):
        rng.shuffle(shuffled)
        if auroc(scores, shuffled) >= observed:
            hits += 1
    n_pos = sum(labels)
    k = len(labels)
    comb = 1
    for i in range(min(n_pos, k - n_pos)):
        comb = comb * (k - i) // (i + 1)
    return {"observed_auroc": observed, "p_value": (hits + 1) / (n + 1),
            "min_achievable_p": 1.0 / comb if comb else float("nan")}


# --------------------------------------------------------------------------
# Out-of-sample scoring
# --------------------------------------------------------------------------

def leave_one_out(rows: List[Dict[str, float]], labels: List[int],
                  names: List[str], epochs: int = 1500) -> List[float]:
    """Refit without each document, then score it. The honest number."""
    out: List[float] = []
    for i in range(len(rows)):
        tr_rows = [r for j, r in enumerate(rows) if j != i]
        tr_lab = [l for j, l in enumerate(labels) if j != i]
        if len(set(tr_lab)) < 2:
            out.append(0.5)
            continue
        w, b = fit(tr_rows, tr_lab, names, epochs=epochs)
        z = b + sum(w.get(k, 0.0) * v for k, v in rows[i].items())
        out.append(sigmoid(z))
    return out


def ablate(rows: List[Dict[str, float]], labels: List[int], names: List[str],
           baseline_auroc: float, baseline_loss: float):
    """Drop each signal, refit out-of-sample, report what it was worth.

    Ranked by log-loss rather than AUROC: once separation is perfect, removing
    a signal usually leaves AUROC at 1.0 and the ablation reads as a column of
    zeros. Log-loss still moves, so it says which signals are actually carrying
    the decision.
    """
    results = []
    for name in names:
        reduced_names = [n for n in names if n != name]
        if not reduced_names:
            continue
        reduced = [{k: v for k, v in r.items() if k != name} for r in rows]
        scores = leave_one_out(reduced, labels, reduced_names, epochs=800)
        results.append({
            "signal": name,
            "auroc_without": auroc(scores, labels),
            "auroc_cost": baseline_auroc - auroc(scores, labels),
            "logloss_without": log_loss(scores, labels),
            "logloss_cost": log_loss(scores, labels) - baseline_loss,
            "margin_without": margin(scores, labels),
        })
    results.sort(key=lambda r: -r["logloss_cost"])
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def evaluate_corpus(human_dir: str, machine_dir: str, do_ablate: bool = False,
                    seed: int = 0) -> Dict:
    rows, labels, names = build_dataset(human_dir, machine_dir)
    if len(rows) < 4:
        raise SystemExit("aidetect.evaluate: need at least 4 documents")

    paths = (sorted(_txt(human_dir)) + sorted(_txt(machine_dir)))

    # In-sample: the shipped hand-weighted detector, scoring documents its
    # constants were tuned on.
    heuristic = [analyse(_read(p)).probability for p in paths]
    # Out-of-sample: weights refit without each document.
    oos = leave_one_out(rows, labels, names)

    report: Dict = {
        "n": len(labels),
        "n_human": labels.count(0),
        "n_machine": labels.count(1),
        "n_signals": len(names),
        "heuristic_in_sample": _block(heuristic, labels, seed),
        "fitted_out_of_sample": _block(oos, labels, seed),
    }
    report["overfitting_gap_auroc"] = (
        report["heuristic_in_sample"]["auroc"] - report["fitted_out_of_sample"]["auroc"])

    if do_ablate:
        report["ablation"] = ablate(
            rows, labels, names,
            report["fitted_out_of_sample"]["auroc"],
            report["fitted_out_of_sample"]["log_loss"])
    report["per_document"] = [
        {"path": p, "label": "machine" if y else "human",
         "heuristic": round(h, 4), "out_of_sample": round(o, 4)}
        for p, y, h, o in zip(paths, labels, heuristic, oos)
    ]
    return report


def _block(scores: Scores, labels: Labels, seed: int) -> Dict:
    a = auroc(scores, labels)
    f90, t90 = fpr_at_tpr(scores, labels, 0.90)
    t01, _ = tpr_at_fpr(scores, labels, 0.01)
    t05, _ = tpr_at_fpr(scores, labels, 0.05)
    return {
        "auroc": a,
        "bootstrap": bootstrap_auroc(scores, labels, seed=seed),
        "permutation": permutation_test(scores, labels, seed=seed),
        "fpr_at_tpr90": f90,
        "threshold_at_tpr90": t90,
        "tpr_at_fpr01": t01,
        "tpr_at_fpr05": t05,
        "log_loss": log_loss(scores, labels),
        "margin": margin(scores, labels),
        "brier": brier(scores, labels),
        "ece": ece(scores, labels),
        "at_threshold_0.5": confusion(scores, labels, 0.5),
    }


def _txt(d: str) -> List[str]:
    import glob
    return glob.glob(os.path.join(d, "*.txt"))


def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


def format_report(r: Dict) -> str:
    L: List[str] = []
    L.append("=" * 72)
    L.append(f"  corpus: {r['n']} documents ({r['n_human']} human, "
             f"{r['n_machine']} machine), {r['n_signals']} signals")
    L.append("=" * 72)

    if r["n"] < 50:
        L.append("")
        L.append(f"  !! {r['n']} documents has essentially no statistical power.")
        L.append("     Treat every figure below as indicative only.")

    L.append("")
    L.append("  !! LEAVE-ONE-OUT HERE IS NOT FULLY HONEST.")
    L.append("     It refits the logistic weights without each document, but the")
    L.append("     FEATURE DEFINITIONS (e.g. the template_repetition midpoint)")
    L.append("     were hand-tuned on the whole corpus and are not re-derived per")
    L.append("     fold. Genuinely clean numbers need a held-out set that never")
    L.append("     informed a constant.")

    bs_in = r["heuristic_in_sample"]["bootstrap"]
    if bs_in["lo95"] == bs_in["hi95"]:
        L.append("")
        L.append("  !! The bootstrap interval is degenerate (lo == hi).")
        L.append("     With perfect separation and small n, every resample also")
        L.append("     separates perfectly, so the interval collapses. Read that")
        L.append("     as the bootstrap being unable to express uncertainty here,")
        L.append("     NOT as precision.")

    for key, title in (("heuristic_in_sample",
                        "SHIPPED HEURISTIC  (in-sample: constants were tuned on these docs)"),
                       ("fitted_out_of_sample",
                        "FITTED, LEAVE-ONE-OUT  (honest: refit without each document)")):
        b = r[key]
        L.append("")
        L.append(f"  {title}")
        L.append("  " + "-" * 68)
        bs = b["bootstrap"]
        L.append(f"    AUROC                 {b['auroc']:.4f}   "
                 f"95% CI [{bs['lo95']:.3f}, {bs['hi95']:.3f}]  "
                 f"({bs['usable']} usable replicates)")
        pm = b["permutation"]
        L.append(f"    permutation p         {pm['p_value']:.4f}   "
                 f"(floor at this n: {pm['min_achievable_p']:.4f})")
        L.append(f"    FPR @ TPR=90%         {b['fpr_at_tpr90']:.4f}   "
                 f"(threshold {b['threshold_at_tpr90']:.3f})")
        L.append(f"    TPR @ FPR<=1%         {b['tpr_at_fpr01']:.4f}")
        L.append(f"    TPR @ FPR<=5%         {b['tpr_at_fpr05']:.4f}")
        L.append(f"    log loss              {b['log_loss']:.4f}  "
                 f"(keeps resolution at AUROC 1.0)")
        L.append(f"    margin                {b['margin']:+.4f}  "
                 f"(min machine - max human)")
        L.append(f"    Brier                 {b['brier']:.4f}  (lower better)")
        L.append(f"    calibration error     {b['ece']:.4f}  (does 0.8 mean 80%?)")
        c = b["at_threshold_0.5"]
        L.append(f"    @0.5  tp={c['tp']} fp={c['fp']} tn={c['tn']} fn={c['fn']}  "
                 f"precision={c['precision']:.3f} recall={c['recall']:.3f}")

    L.append("")
    L.append(f"  OVERFITTING GAP (in-sample AUROC - out-of-sample AUROC): "
             f"{r['overfitting_gap_auroc']:+.4f}")

    if "ablation" in r:
        L.append("")
        L.append("  ABLATION  (out-of-sample cost of removing each signal, "
                 "ranked by log loss)")
        L.append("  " + "-" * 68)
        L.append(f"    {'signal':<36}{'logloss cost':>14}{'auroc cost':>12}")
        for row in r["ablation"][:12]:
            L.append(f"    {row['signal']:<36}{row['logloss_cost']:>+14.4f}"
                     f"{row['auroc_cost']:>+12.4f}")

    L.append("")
    L.append("  PER DOCUMENT")
    L.append("  " + "-" * 68)
    L.append(f"    {'document':<38}{'label':<10}{'heur':>8}{'oos':>9}")
    for d in r["per_document"]:
        name = os.path.basename(d["path"])[:36]
        flag = "  <-- miss" if ((d["label"] == "machine") != (d["out_of_sample"] >= 0.5)) else ""
        L.append(f"    {name:<38}{d['label']:<10}{d['heuristic']:>8.3f}"
                 f"{d['out_of_sample']:>9.3f}{flag}")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="aidetect.evaluate")
    ap.add_argument("--human", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--ablate", action="store_true",
                    help="measure each signal's out-of-sample contribution (slow)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    r = evaluate_corpus(args.human, args.machine, do_ablate=args.ablate, seed=args.seed)
    if args.json:
        json.dump(r, sys.stdout, indent=2, default=float)
        print()
    else:
        print(format_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
