"""Tests for three-class triage and span localisation."""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aidetect.synth import make_corpus, make_mixed
from aidetect.triage import classify, span_metrics

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _texts(kind):
    return [open(f, encoding="utf-8").read()
            for f in sorted(glob.glob(os.path.join(SAMPLES, kind, "*.txt")))]


class TestSynth(unittest.TestCase):
    def setUp(self):
        self.H, self.M = _texts("human"), _texts("machine")

    def test_ratio_is_approximately_honoured(self):
        d = make_mixed(self.H, self.M, 0.5, total_words=900)
        self.assertAlmostEqual(d.machine_fraction, 0.5, delta=0.12)

    def test_pure_cases_labelled(self):
        self.assertEqual(make_mixed(self.H, self.M, 0.0).label, "human")
        self.assertEqual(make_mixed(self.H, self.M, 1.0).label, "ai")

    def test_word_labels_match_spans(self):
        d = make_mixed(self.H, self.M, 0.4, total_words=600)
        self.assertEqual(sum(d.word_labels()),
                         sum(b - a for a, b in d.machine_spans))

    def test_spans_do_not_overlap(self):
        d = make_mixed(self.H, self.M, 0.6, total_words=900)
        for (a1, b1), (a2, b2) in zip(d.machine_spans, d.machine_spans[1:]):
            self.assertLessEqual(b1, a2)

    def test_deterministic(self):
        a = make_mixed(self.H, self.M, 0.5, seed=7).text
        b = make_mixed(self.H, self.M, 0.5, seed=7).text
        self.assertEqual(a, b)


class TestTriage(unittest.TestCase):
    def setUp(self):
        self.H, self.M = _texts("human"), _texts("machine")

    def test_pure_human_is_human(self):
        for t in self.H:
            self.assertEqual(classify(t).prediction, "human",
                             "false positive on pure human text")

    def test_pure_machine_is_ai(self):
        for t in self.M:
            self.assertEqual(classify(t).prediction, "ai")

    def test_pure_human_yields_no_spans(self):
        for t in self.H:
            self.assertEqual(classify(t).machine_spans, [])

    def test_short_text_abstains(self):
        r = classify("far too short to judge at all")
        self.assertEqual(r.prediction, "human")
        self.assertTrue(r.note)

    def test_mixed_document_is_assisted_not_pure(self):
        d = make_mixed(self.H, self.M, 0.5, total_words=900)
        self.assertEqual(classify(d.text).prediction, "ai_assisted")

    def test_spans_are_within_bounds(self):
        d = make_mixed(self.H, self.M, 0.5, total_words=900)
        r = classify(d.text)
        for a, b in r.machine_spans:
            self.assertGreaterEqual(a, 0)
            self.assertLessEqual(b, r.word_count)
            self.assertLess(a, b)

    def test_span_quality_on_synthetic_corpus(self):
        """Regression floor for localisation, measured not asserted blindly."""
        docs = make_corpus(self.H, self.M, per_ratio=2)
        scored = []
        for d in docs:
            if d.machine_fraction <= 0:
                continue
            m = span_metrics(d.word_labels(),
                             classify(d.text).predicted_word_labels())
            if m["precision"] == m["precision"]:
                scored.append(m)
        self.assertGreaterEqual(len(scored), 8)
        mean_f1 = sum(m["f1"] for m in scored) / len(scored)
        mean_iou = sum(m["iou"] for m in scored) / len(scored)
        self.assertGreater(mean_f1, 0.65, f"span F1 regressed to {mean_f1:.2f}")
        self.assertGreater(mean_iou, 0.55, f"span IoU regressed to {mean_iou:.2f}")

    def test_document_class_agreement(self):
        docs = make_corpus(self.H, self.M, per_ratio=2)
        agree = sum(1 for d in docs if classify(d.text).prediction == d.label)
        self.assertGreaterEqual(agree, int(0.75 * len(docs)))


class TestSpanMetrics(unittest.TestCase):
    def test_perfect(self):
        m = span_metrics([1, 1, 0, 0], [1, 1, 0, 0])
        self.assertEqual((m["precision"], m["recall"], m["iou"]), (1.0, 1.0, 1.0))

    def test_disjoint(self):
        m = span_metrics([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertEqual(m["iou"], 0.0)

    def test_half_overlap(self):
        m = span_metrics([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
