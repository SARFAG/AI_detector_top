"""Tests for the evaluation harness.

Metrics are tested against hand-computable values. A broken metric is worse
than no metric, because it produces confident wrong numbers.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aidetect.evaluate import (auroc, bootstrap_auroc, confusion, ece,
                               evaluate_corpus, fpr_at_tpr, log_loss, margin,
                               permutation_test, roc_points, tpr_at_fpr)

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


class TestAUROC(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)

    def test_inverted(self):
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)

    def test_all_ties_is_chance(self):
        """An abstaining detector must score 0.5, not 1.0. Ties count half."""
        self.assertEqual(auroc([0.5] * 4, [1, 1, 0, 0]), 0.5)

    def test_known_intermediate(self):
        # pos {0.6, 0.4}, neg {0.5, 0.3}: 0.6>both, 0.4>0.3 only => 3/4
        self.assertAlmostEqual(auroc([0.6, 0.4, 0.5, 0.3], [1, 1, 0, 0]), 0.75)

    def test_single_class_is_nan(self):
        self.assertNotEqual(auroc([0.5, 0.6], [1, 1]), auroc([0.5, 0.6], [1, 1]))


class TestOperatingPoints(unittest.TestCase):
    def test_fpr_at_tpr_perfect(self):
        fpr, _ = fpr_at_tpr([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 0.90)
        self.assertEqual(fpr, 0.0)

    def test_fpr_at_tpr_overlapping(self):
        # To catch both positives the threshold must reach 0.4, admitting 0.5.
        fpr, _ = fpr_at_tpr([0.6, 0.4, 0.5, 0.3], [1, 1, 0, 0], 1.0)
        self.assertEqual(fpr, 0.5)

    def test_tpr_at_zero_fpr(self):
        tpr, _ = tpr_at_fpr([0.6, 0.4, 0.5, 0.3], [1, 1, 0, 0], 0.0)
        self.assertEqual(tpr, 0.5)

    def test_roc_is_monotonic(self):
        pts = roc_points([0.9, 0.7, 0.5, 0.3, 0.1], [1, 1, 0, 1, 0])
        for a, b in zip(pts, pts[1:]):
            self.assertLessEqual(a[0], b[0])
            self.assertLessEqual(a[1], b[1])


class TestResolutionMetrics(unittest.TestCase):
    """These exist because AUROC saturates at 1.0 and stops discriminating."""

    def test_log_loss_still_moves_at_perfect_auroc(self):
        confident = [0.99, 0.99, 0.01, 0.01]
        timid = [0.55, 0.55, 0.45, 0.45]
        self.assertEqual(auroc(confident, [1, 1, 0, 0]),
                         auroc(timid, [1, 1, 0, 0]))
        self.assertLess(log_loss(confident, [1, 1, 0, 0]),
                        log_loss(timid, [1, 1, 0, 0]))

    def test_log_loss_is_finite_at_extremes(self):
        self.assertLess(log_loss([1.0, 0.0], [0, 1]), 1e9)

    def test_margin_sign(self):
        self.assertGreater(margin([0.9, 0.8, 0.2], [1, 1, 0]), 0)
        self.assertLess(margin([0.4, 0.3, 0.6], [1, 1, 0]), 0)


class TestCalibration(unittest.TestCase):
    def test_ece_zero_when_perfect(self):
        self.assertAlmostEqual(ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]), 0.0)

    def test_ece_detects_overconfidence(self):
        self.assertGreater(ece([0.99, 0.99, 0.99, 0.99], [1, 1, 0, 0]), 0.4)

    def test_confusion_counts(self):
        c = confusion([0.9, 0.4, 0.8, 0.1], [1, 1, 0, 0], 0.5)
        self.assertEqual((c["tp"], c["fn"], c["fp"], c["tn"]), (1, 1, 1, 1))


class TestUncertainty(unittest.TestCase):
    def test_bootstrap_reports_usable_replicates(self):
        r = bootstrap_auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], n=200)
        self.assertGreater(r["usable"], 0)
        self.assertLessEqual(r["lo95"], r["hi95"])

    def test_permutation_reports_its_own_floor(self):
        """At small n the test may be incapable of significance; it must say so."""
        r = permutation_test([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], n=500)
        self.assertAlmostEqual(r["min_achievable_p"], 1.0 / 6.0, places=6)
        self.assertGreaterEqual(r["p_value"], r["min_achievable_p"] * 0.5)


class TestCorpusEvaluation(unittest.TestCase):
    def test_runs_and_reports_both_modes(self):
        r = evaluate_corpus(os.path.join(SAMPLES, "human"),
                            os.path.join(SAMPLES, "machine"))
        self.assertEqual(r["n"], r["n_human"] + r["n_machine"])
        self.assertGreaterEqual(r["n"], 8)
        self.assertIn("heuristic_in_sample", r)
        self.assertIn("fitted_out_of_sample", r)
        self.assertIn("overfitting_gap_auroc", r)

    def test_out_of_sample_does_not_beat_in_sample_by_much(self):
        """A large negative gap would mean the harness is leaking labels."""
        r = evaluate_corpus(os.path.join(SAMPLES, "human"),
                            os.path.join(SAMPLES, "machine"))
        self.assertLess(r["overfitting_gap_auroc"], 0.5)
        self.assertGreater(r["overfitting_gap_auroc"], -0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
