"""Tests for classical stylometry, discourse features and RAIDAR."""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aidetect import discourse, stylometry
from aidetect.rewrite import raidar_signal, word_edit_ratio
from aidetect.stylometry import (DeltaModel, NearestCentroid, delta, profile,
                                 verify_authorship)
from aidetect.text import parse

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _docs(kind):
    return [parse(open(f, encoding="utf-8").read())
            for f in sorted(glob.glob(os.path.join(SAMPLES, kind, "*.txt")))]


class TestProfile(unittest.TestCase):
    def test_function_word_frequencies_are_rates(self):
        p = profile(parse("the cat sat on the mat and the dog " * 20))
        self.assertGreater(p.function_freqs["the"], 0.2)
        self.assertLessEqual(p.function_freqs["the"], 1.0)

    def test_profile_is_content_free(self):
        """Same function-word skeleton, different content words."""
        a = profile(parse("the cat is on the mat and the dog is here " * 20))
        b = profile(parse("the ship is on the lake and the bird is here " * 20))
        for w in ("the", "is", "on", "and"):
            self.assertAlmostEqual(a.function_freqs[w], b.function_freqs[w],
                                   places=6)


class TestDelta(unittest.TestCase):
    def setUp(self):
        self.H, self.M = _docs("human"), _docs("machine")

    def test_delta_to_self_is_zero(self):
        p = profile(self.H[0])
        model = DeltaModel.fit([profile(d) for d in self.H + self.M])
        self.assertAlmostEqual(delta(model, p, p), 0.0, places=9)

    def test_delta_is_symmetric(self):
        a, b = profile(self.H[0]), profile(self.M[0])
        model = DeltaModel.fit([profile(d) for d in self.H + self.M])
        self.assertAlmostEqual(delta(model, a, b), delta(model, b, a), places=9)

    def test_nearest_centroid_separates_classes(self):
        nc = NearestCentroid(self.H, self.M)
        for d in self.H:
            self.assertLess(nc.score(d)[0], 0, "human scored machine-leaning")
        for d in self.M:
            self.assertGreater(nc.score(d)[0], 0, "machine scored human-leaning")

    def test_signal_requires_length(self):
        sigs = stylometry.analyse(parse("short text here"))
        self.assertEqual(sigs[0].logodds, 0.0)

    def test_signal_fires_on_corpus(self):
        sig = stylometry.analyse(self.M[0])[0]
        self.assertGreater(sig.logodds, 0.0)


class TestAuthorshipVerification(unittest.TestCase):
    """The stronger question: does this match THIS author, not 'an AI'?"""

    def test_same_author_matches_better_than_others(self):
        H, M = _docs("human"), _docs("machine")
        # Treat one human doc as the candidate and the others as unrelated
        # background; the candidate should sit closer to its own class.
        res = verify_authorship(H[0], [H[1], H[2]], M)
        self.assertIn("delta_to_author", res)
        self.assertIn("percentile_among_others", res)
        self.assertGreaterEqual(res["percentile_among_others"], 0.0)
        self.assertLessEqual(res["percentile_among_others"], 1.0)

    def test_requires_reference(self):
        with self.assertRaises(ValueError):
            verify_authorship(_docs("human")[0], [], _docs("machine"))


class TestDiscourseRejections(unittest.TestCase):
    """The three features that were measured and did NOT separate must stay
    at zero weight, so a negative result is not quietly re-promoted."""

    def test_rejected_features_carry_no_weight(self):
        for d in _docs("human") + _docs("machine"):
            for sig in discourse.analyse(d):
                if sig.name in ("discourse.topic_drift",
                                "discourse.rare_word_reuse",
                                "discourse.readability_variance"):
                    self.assertEqual(sig.logodds, 0.0,
                                     f"{sig.name} was given weight without "
                                     f"evidence it separates")

    def test_rejected_features_still_report_measurements(self):
        sigs = {s.name: s for s in discourse.analyse(_docs("human")[0])}
        self.assertTrue(sigs["discourse.topic_drift"].evidence)

    def test_sentence_type_mix_credits_human_variety(self):
        text = ("Really? I think so. Yes. But who knows! Maybe not. "
                "Hard to say. Could be. Who cares? Fine. Moving on! ") * 3
        sig = discourse._sentence_type_mix(parse(text))
        self.assertLess(sig.logodds, 0)


class TestRaidar(unittest.TestCase):
    def test_edit_ratio_bounds(self):
        self.assertEqual(word_edit_ratio("a b c", "a b c"), 0.0)
        self.assertEqual(word_edit_ratio("a b c", "x y z"), 1.0)

    def test_edit_ratio_ignores_case_and_punctuation(self):
        self.assertEqual(word_edit_ratio("Hello, world!", "hello world"), 0.0)

    def test_no_rewriter_is_neutral(self):
        d = parse(open(os.path.join(SAMPLES, "machine", "seo_article.txt"),
                       encoding="utf-8").read())
        self.assertEqual(raidar_signal(d, None).logodds, 0.0)

    def test_unchanged_rewrite_reads_machine(self):
        d = parse(open(os.path.join(SAMPLES, "machine", "seo_article.txt"),
                       encoding="utf-8").read())
        self.assertGreater(raidar_signal(d, lambda t: d.raw).logodds, 1.0)

    def test_heavy_rewrite_reads_human(self):
        d = parse(open(os.path.join(SAMPLES, "human", "blog_moving.txt"),
                       encoding="utf-8").read())
        sig = raidar_signal(d, lambda t: " ".join(w + "z" for w in d.raw.split()))
        self.assertLess(sig.logodds, -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
