"""Tests for the greedy adversarial rewriter."""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aidetect.bypass import bypass, compare_to_fixed_order
from aidetect.detector import analyse

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _machine():
    return [open(f, encoding="utf-8").read()
            for f in sorted(glob.glob(os.path.join(SAMPLES, "machine", "*.txt")))]


class TestSearch(unittest.TestCase):
    def test_score_never_increases(self):
        """Greedy must reject a move that makes things worse."""
        for t in _machine():
            r = bypass(t)
            self.assertLessEqual(r.final_score, r.original_score + 1e-9)
            for s in r.steps:
                self.assertLess(s.score_after, s.score_before)

    def test_reported_text_matches_reported_score(self):
        for t in _machine()[:2]:
            r = bypass(t)
            self.assertAlmostEqual(analyse(r.text).probability, r.final_score,
                                   places=9)

    def test_deterministic(self):
        t = _machine()[0]
        self.assertEqual(bypass(t, seed=3).text, bypass(t, seed=3).text)

    def test_respects_round_limit(self):
        r = bypass(_machine()[0], target=0.0, rounds=3)
        self.assertLessEqual(len(r.steps), 3)

    def test_stops_at_target(self):
        """A document already at or below target needs no moves.

        Uses target=1.0 rather than 0.999: the first machine sample scores a
        full 100%, so 0.999 is below it and the search correctly runs.
        """
        t = _machine()[0]
        r = bypass(t, target=1.0)
        self.assertTrue(r.reached_target)
        self.assertEqual(len(r.steps), 0)

    def test_no_repeats_by_default(self):
        names = [s.attack for s in bypass(_machine()[3], rounds=12).steps]
        self.assertEqual(len(names), len(set(names)))

    def test_repeats_allowed_when_asked(self):
        r = bypass(_machine()[3], target=0.0, rounds=20, allow_repeats=True)
        self.assertGreaterEqual(len(r.steps), 1)

    def test_greedy_is_at_least_as_good_as_fixed_order(self):
        """The reason this module exists rather than reusing robustness_report."""
        for t in _machine():
            c = compare_to_fixed_order(t)
            self.assertGreaterEqual(c["greedy_advantage"], -0.02,
                                    "greedy search did worse than fixed order")

    def test_handles_degenerate_input(self):
        for t in ("", "short", "a b c.\n\nd e f."):
            try:
                bypass(t, rounds=3)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"crashed on {t!r}: {exc}")


class TestMeasuredOutcome(unittest.TestCase):
    """Pins the measured result so a change to either side is visible."""

    def test_does_not_currently_bypass_the_detector(self):
        """No machine document crosses 0.5 under the full attack space.

        This is the headline result of the adversarial evaluation. If a change
        to the detector or the attack suite breaks it, this test fails and the
        evaluation needs rerunning and rewriting - in either direction.
        """
        for t in _machine():
            r = bypass(t, target=0.0, rounds=30, allow_repeats=True)
            self.assertFalse(r.crossed_threshold,
                             f"a document now bypasses: {r.final_score:.3f}")

    def test_attack_space_is_not_trivially_exhausted(self):
        """At least one document should admit several productive moves,
        otherwise the evaluation is not testing much."""
        best = max(len(bypass(t, target=0.0, rounds=30,
                              allow_repeats=True).steps) for t in _machine())
        self.assertGreaterEqual(best, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
