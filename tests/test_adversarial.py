"""Tests for the adversarial robustness harness."""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aidetect.adversarial import (ATTACKS, hardening_corpus, robustness_report,
                                  single_attack_effect)
from aidetect.detector import analyse

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _machine():
    return [open(f, encoding="utf-8").read()
            for f in sorted(glob.glob(os.path.join(SAMPLES, "machine", "*.txt")))]


class TestAttacks(unittest.TestCase):
    def test_every_attack_names_a_real_signal(self):
        """An attack that targets nothing measurable is untestable."""
        rep = analyse(_machine()[0])
        known = {s.name for s in rep.signals}
        for atk in ATTACKS:
            for target in atk.targets.split(" / "):
                self.assertIn(target.strip(), known,
                              f"{atk.name} targets unknown signal {target}")

    def test_attacks_do_not_destroy_the_text(self):
        """A perturbation must stay a plausible document, not mangle it."""
        text = _machine()[0]
        import random
        for atk in ATTACKS:
            out = atk.fn(text, random.Random(0))
            self.assertGreater(len(out.split()), len(text.split()) * 0.6,
                               f"{atk.name} deleted most of the text")
            self.assertLess(len(out.split()), len(text.split()) * 1.6,
                            f"{atk.name} inflated the text")

    def test_attacks_are_deterministic(self):
        import random
        text = _machine()[0]
        for atk in ATTACKS:
            a = atk.fn(text, random.Random(7))
            b = atk.fn(text, random.Random(7))
            self.assertEqual(a, b, f"{atk.name} is not reproducible")

    def test_attacks_never_raise(self):
        import random
        for atk in ATTACKS:
            for text in ("", "short", "a b c.\n\nd e f.", _machine()[0]):
                try:
                    atk.fn(text, random.Random(0))
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"{atk.name} raised on {text[:12]!r}: {exc}")


class TestRobustnessReport(unittest.TestCase):
    def test_report_shape(self):
        r = robustness_report(_machine()[0])
        for key in ("baseline", "attack_cost", "survived", "final_score",
                    "per_attack", "trajectory"):
            self.assertIn(key, r)

    def test_trajectory_starts_at_baseline(self):
        r = robustness_report(_machine()[0])
        self.assertEqual(r["trajectory"][0]["score"], r["baseline"])

    def test_attack_cost_is_consistent_with_survival(self):
        r = robustness_report(_machine()[0])
        self.assertEqual(r["survived"], r["attack_cost"] is None)

    def test_single_attacks_ranked_by_effect(self):
        rows = single_attack_effect(_machine()[0])
        for a, b in zip(rows, rows[1:]):
            self.assertGreaterEqual(a["drop"], b["drop"])


class TestHardeningCorpus(unittest.TestCase):
    def test_variants_keep_the_machine_label(self):
        """Roughening the surface of generated text does not make it human."""
        corpus = hardening_corpus(_machine()[:2])
        self.assertTrue(corpus)
        for _, label in corpus:
            self.assertTrue(label.startswith("machine_attacked_"))

    def test_current_recall_on_hard_negatives(self):
        """Regression floor. If a change drops this, the change weakened the
        detector against exactly the perturbations an evader would try."""
        corpus = hardening_corpus(_machine())
        caught = sum(1 for text, _ in corpus if analyse(text).probability >= 0.5)
        self.assertGreaterEqual(caught / len(corpus), 0.90,
                                f"hard-negative recall fell to "
                                f"{caught}/{len(corpus)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
