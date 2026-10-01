"""Test suite. Run with: python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aidetect
from aidetect import code as code_layer
from aidetect import forensics, lexical, statistical
from aidetect.detector import analyse, analyse_segments
from aidetect.lexical import _RX_HUMAN
from aidetect.text import parse, split_sentences

HERE = os.path.dirname(__file__)
SAMPLES = os.path.join(HERE, "..", "samples")


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestTokenisation(unittest.TestCase):
    def test_abbreviations_do_not_split(self):
        s = split_sentences("Dr. Smith arrived. He was late.")
        self.assertEqual(len(s), 2)

    def test_initials_do_not_split(self):
        s = split_sentences("It was J. R. Tolkien. He wrote books.")
        self.assertEqual(len(s), 2)

    def test_empty_input(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n  "), [])

    def test_code_fences_stripped_from_prose(self):
        doc = parse("Some prose here.\n```\nnot prose at all\n```\nMore prose.")
        self.assertNotIn("not prose", doc.prose)
        self.assertIn("More prose", doc.prose)


class TestRegressionWordBoundaries(unittest.TestCase):
    """Regression: 'ugh' used to match inside 'throughput' and 'thoughtfully',
    poisoning the human-marker score with phantom evidence."""

    def test_human_marker_needs_word_boundary(self):
        for word in ("throughput", "thoughtfully", "enough", "roughly"):
            self.assertEqual(_RX_HUMAN.findall(word), [], f"false match in {word}")

    def test_human_marker_still_matches_standalone(self):
        self.assertTrue(_RX_HUMAN.findall("ugh, that was bad"))
        self.assertTrue(_RX_HUMAN.findall("tbh I have no idea"))

    def test_markers_ending_in_punctuation_still_match(self):
        self.assertTrue(_RX_HUMAN.findall("edit: fixed a typo"))


class TestRegressionTrailingWhitespace(unittest.TestCase):
    """Regression: `\\s+$` in MULTILINE matched the newline of every blank line,
    so paragraph breaks were miscounted as human 'trailing whitespace'."""

    def test_blank_lines_are_not_irregularities(self):
        text = ("This is a clean paragraph of prose with no problems at all. " * 12
                + "\n\n" + "Another clean paragraph follows the first one here. " * 12)
        sig = forensics._typo_absence(parse(text))
        self.assertGreaterEqual(
            sig.logodds, 0.0,
            "blank-line separators must not count as human irregularities")


class TestForensics(unittest.TestCase):
    def test_invisible_chars_detected(self):
        sig = forensics._invisible_chars("hello​world")
        self.assertGreater(sig.logodds, 0)

    def test_leading_bom_is_not_evidence(self):
        sig = forensics._invisible_chars("﻿hello world")
        self.assertEqual(sig.logodds, 0.0)

    def test_mixed_quotes_flagged_higher_than_uniform(self):
        mixed = forensics._smart_quotes('He said “yes” and she said "no".')
        uniform = forensics._smart_quotes('He said “yes” today.')
        self.assertGreater(mixed.logodds, uniform.logodds)
        self.assertTrue(any("MIXED" in e for e in mixed.evidence))

    def test_narrow_nbsp_weighted(self):
        sig = forensics._exotic_spaces("hello world")
        self.assertGreater(sig.logodds, 0.5)

    def test_clean_ascii_produces_nothing(self):
        doc = parse("A perfectly ordinary ASCII sentence with nothing unusual.")
        for sig in forensics.analyse(doc):
            self.assertEqual(sig.logodds, 0.0, f"{sig.name} fired on clean text")


class TestLexical(unittest.TestCase):
    def test_tier4_leakage_is_decisive(self):
        doc = parse("As an AI language model, I don't have personal opinions. " * 3)
        sigs = {s.name: s for s in lexical.analyse(doc)}
        self.assertGreater(sigs["lexical.tier4_assistant_leakage"].logodds, 1.5)

    def test_negative_parallelism(self):
        doc = parse("It's not just a tool, it's a complete paradigm shift here.")
        sigs = {s.name: s for s in lexical.analyse(doc)}
        self.assertGreater(sigs["lexical.negative_parallelism"].logodds, 0)

    def test_human_markers_push_negative(self):
        doc = parse("tbh idk, my boss said whatever and anyway I kinda gave up lol")
        sigs = {s.name: s for s in lexical.analyse(doc)}
        self.assertLess(sigs["lexical.human_markers"].logodds, 0)

    def test_tier2_alone_cannot_convict(self):
        """Ordinary business English must not produce a machine verdict."""
        doc = parse(" ".join(["We leverage robust and comprehensive systems to "
                              "enhance and facilitate significant outcomes."] * 8))
        sigs = {s.name: s for s in lexical.analyse(doc)}
        self.assertLess(sigs["lexical.tier2_register"].logodds, 1.3)

    def test_attribution_identifies_claude_code(self):
        text = ("Fix the parser bug\n\n"
                "\U0001F916 Generated with [Claude Code](https://claude.com/claude-code)\n"
                "Co-Authored-By: Claude <noreply@anthropic.com>")
        attrib = lexical.attribute(parse(text))
        self.assertTrue(attrib)
        self.assertEqual(attrib[0][0], "Claude Code")


class TestStatistical(unittest.TestCase):
    def test_uniform_sentences_flagged(self):
        text = " ".join(["The system works well here." for _ in range(20)])
        sigs = {s.name: s for s in statistical.analyse(parse(text))}
        self.assertGreater(sigs["statistical.burstiness"].logodds, 0.5)

    def test_bursty_sentences_credited(self):
        text = ("Yes. " "I spent the entire afternoon convinced that the problem lay "
                "somewhere deep inside the connection pooling layer, which it did not. "
                "No. Wrong again. "
                "The actual cause turned out to be an unbounded timeout on an external "
                "HTTP client that nobody had thought about in two years. "
                "Huh. Anyway. ") * 3
        sigs = {s.name: s for s in statistical.analyse(parse(text))}
        self.assertLess(sigs["statistical.burstiness"].logodds, 0)

    def test_ngram_rate_is_a_proportion(self):
        """Regression: rate was divided by unique gram count, exceeding 1.0."""
        text = " ".join(["the system is designed for scale and speed"] * 40)
        sig = statistical._ngram_repetition(parse(text))
        self.assertLessEqual(sig.detail, 1.0)
        self.assertGreaterEqual(sig.detail, 0.0)

    def test_short_text_yields_no_statistical_signal(self):
        for sig in statistical.analyse(parse("Too short to say anything.")):
            self.assertEqual(sig.logodds, 0.0)


class TestCodeLayer(unittest.TestCase):
    def test_prose_is_not_treated_as_code(self):
        self.assertFalse(code_layer.looks_like_code(
            "This is an ordinary paragraph.\nIt has several lines.\n"
            "None of them are code.\nNot one.\n"))

    def test_python_is_detected(self):
        self.assertTrue(code_layer.looks_like_code(
            "import os\n\ndef main():\n    return os.getcwd()\n\nmain()\n"))

    def test_emoji_in_output_flagged(self):
        src = ('import sys\n\ndef a():\n    print("✅ Done!")\n\n'
               'def b():\n    return 1\n\ndef c():\n    return 2\n')
        sigs = {s.name: s for s in code_layer.analyse(parse(src))}
        self.assertGreater(sigs["code.emoji_output"].logodds, 0)


class TestDetectorContract(unittest.TestCase):
    def test_probability_is_bounded(self):
        for text in ["", "x", "hello world", "As an AI language model " * 200]:
            p = analyse(text).probability
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_empty_input_does_not_crash(self):
        rep = analyse("")
        self.assertEqual(rep.confidence, "none")

    def test_short_text_is_withheld(self):
        rep = analyse("This text is far too short to judge fairly at all.")
        self.assertEqual(rep.verdict, "insufficient text")
        self.assertEqual(rep.probability, 0.5)

    def test_long_human_text_not_convicted_by_length(self):
        """Length alone must never accumulate a conviction.

        Uses the real human samples concatenated rather than one sentence
        repeated: repeated text genuinely *is* uniform and low-burstiness, so a
        degenerate fixture would be testing the wrong thing.
        """
        combined = "\n\n".join(
            read(p) for p in sorted(glob.glob(os.path.join(SAMPLES, "human", "*.txt"))))
        rep = analyse(combined)
        self.assertGreater(rep.word_count, 800, "fixture should be long")
        self.assertLess(rep.probability, 0.5,
                        f"long human text convicted by length: {rep.probability:.3f}")

    def test_deterministic(self):
        text = read(os.path.join(SAMPLES, "machine", "seo_article.txt"))
        self.assertEqual(analyse(text).probability, analyse(text).probability)

    def test_unicode_input_does_not_crash(self):
        for text in ["你好世界 " * 50, "\U0001F600" * 100,
                     "\x00\x01 weird \udcff"]:
            try:
                analyse(text)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"crashed on {text[:20]!r}: {exc}")


class TestEndToEndSeparation(unittest.TestCase):
    """The real test: does it actually separate the two classes?"""

    def setUp(self):
        self.human = sorted(glob.glob(os.path.join(SAMPLES, "human", "*.txt")))
        self.machine = sorted(glob.glob(os.path.join(SAMPLES, "machine", "*.txt")))
        self.assertTrue(self.human and self.machine, "samples missing")

    def test_human_samples_score_low(self):
        for path in self.human:
            p = analyse(read(path)).probability
            self.assertLess(p, 0.5, f"false positive on {os.path.basename(path)}: {p:.3f}")

    def test_machine_samples_score_high(self):
        for path in self.machine:
            p = analyse(read(path)).probability
            self.assertGreater(p, 0.5, f"false negative on {os.path.basename(path)}: {p:.3f}")

    def test_separation_margin(self):
        hi = max(analyse(read(p)).probability for p in self.human)
        lo = min(analyse(read(p)).probability for p in self.machine)
        self.assertGreater(lo - hi, 0.25,
                           f"margin too thin: human max {hi:.3f}, machine min {lo:.3f}")


class TestSegmentation(unittest.TestCase):
    def test_locates_pasted_machine_paragraph(self):
        human = ("ok so I spent all weekend on this and I still dont really get it. "
                 "tried three things, none worked, gave up around midnight. "
                 "anyway heres where I got to. ") * 4
        machine = ("It's worth noting that this approach plays a crucial role in "
                   "fostering robust outcomes. Let's delve into the intricate "
                   "tapestry of considerations that underscore its pivotal nature "
                   "within the ever-evolving landscape of modern practice. ") * 4
        segs = analyse_segments(human + "\n\n" + machine)
        self.assertGreaterEqual(len(segs), 2)
        self.assertGreater(max(p for _, p in segs), min(p for _, p in segs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
