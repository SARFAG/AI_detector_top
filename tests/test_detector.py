"""Test suite. Run with: python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import glob
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aidetect
from aidetect import code as code_layer
from aidetect import forensics, lexical, stance, statistical, syntax
from aidetect.detector import analyse, analyse_segments, fraction_ai
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


class TestSyntaxLayer(unittest.TestCase):
    """Added after the detector missed a machine-written technical spec."""

    def test_template_repetition_is_length_normalised(self):
        """Raw repeat-share grows with length; the signal must not just be a
        length detector."""
        unit = ("The handler validates the payload before the worker commits "
                "the record. A reader fetches the entry while a writer locks "
                "the table. ")
        short = syntax._template_repetition(parse(unit * 12))
        long_ = syntax._template_repetition(parse(unit * 48))
        self.assertLess(abs(short.detail - long_.detail), 12.0,
                        "repeat-share should be roughly length-invariant")

    def test_parallel_templates_flagged(self):
        """Same syntactic shape, different vocabulary - invisible to word
        n-grams, which is the whole point of this layer."""
        text = ("The parser rejects the token when the buffer exceeds the limit. "
                "The loader discards the record when the cursor passes the bound. "
                "The writer blocks the commit when the journal reaches the cap. "
                "The reader drops the frame when the window crosses the edge. "
                "The worker halts the batch when the counter breaks the ceiling. "
                "The monitor flags the span when the latency tops the target. ") * 5
        sig = syntax._template_repetition(parse(text))
        self.assertGreater(sig.logodds, 0.4)

    def test_short_text_yields_nothing(self):
        for sig in syntax.analyse(parse("Three short words here.")):
            self.assertEqual(sig.logodds, 0.0)


class TestStanceLayer(unittest.TestCase):
    def test_authorial_absence_needs_length(self):
        sig = stance._authorial_absence(parse("No pronouns appear in this line."))
        self.assertEqual(sig.logodds, 0.0)

    def test_authorial_presence_credits_human(self):
        text = ("I spent the whole day on this and I am still not sure it is "
                "right. You might disagree, and honestly maybe you should. "
                "TODO: check with the team before we ship it. ") * 10
        sig = stance._authorial_absence(parse(text))
        self.assertLess(sig.logodds, 0)

    def test_authorless_long_text_flagged(self):
        text = ("The handler validates each payload. The worker commits the "
                "record once validation passes. Entries resolve against the "
                "registry. Unresolved entries fail the batch. ") * 14
        sig = stance._authorial_absence(parse(text))
        self.assertGreater(sig.logodds, 0.6)


class TestRegressionPronounCliff(unittest.TestCase):
    """Regression: authorial_absence used to be a cliff.

    Exactly-zero presence scored strongly; anything else scored 0.00. A
    rewritten spec containing a single "I" in 188 words (5.3 markers per 1k,
    over 12x below the lowest human sample) therefore scored nothing. One
    pronoun defeated the signal - not a property worth keeping in a detector
    anyone might try to evade.
    """

    BASE = ("The handler validates each payload before the worker commits it. "
            "Entries resolve against the registry in declaration order. "
            "A malformed entry fails the batch without a partial write. "
            "Digest and size are checked before the record is parsed. ")

    def test_one_pronoun_does_not_cancel_the_signal(self):
        clean = self.BASE * 13
        with_pronoun = "I opened the checkout. " + clean
        a = stance._authorial_absence(parse(clean))
        b = stance._authorial_absence(parse(with_pronoun))
        self.assertGreater(a.logodds, 0.4)
        self.assertGreater(b.logodds, 0.3,
                           "a single pronoun flattened the signal to nothing")
        self.assertLess(a.logodds - b.logodds, 0.25,
                        "one pronoun should taper the score, not cancel it")

    def test_signal_is_monotonic_in_presence(self):
        """More authorial presence must never increase the machine score."""
        prev = None
        for extra in (0, 1, 3, 6, 12):
            text = ("I think you should check this, though I am not sure. " * extra
                    + self.BASE * 13)
            lo = stance._authorial_absence(parse(text)).logodds
            if prev is not None:
                self.assertLessEqual(lo, prev + 1e-9,
                                     "signal is not monotonic in presence")
            prev = lo

    def test_human_density_still_scores_human(self):
        text = ("I spent all day on this and honestly I am not sure you would "
                "agree with me, but here is what I found. Maybe it helps? ") * 10
        self.assertLess(stance._authorial_absence(parse(text)).logodds, 0)


class TestWindowedFraction(unittest.TestCase):
    def test_fraction_bounded_and_consistent(self):
        text = read(os.path.join(SAMPLES, "machine", "seo_article.txt"))
        frac, scores = fraction_ai(text)
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)
        self.assertTrue(scores)
        self.assertAlmostEqual(
            frac, sum(p >= 0.5 for p in scores) / len(scores), places=6)

    def test_short_text_returns_empty(self):
        frac, scores = fraction_ai("too short to window")
        self.assertEqual(frac, 0.0)
        self.assertEqual(scores, [])


class TestRegressionMarkerFreeSpec(unittest.TestCase):
    """Regression for a real miss.

    A 658-word machine-written technical specification scored 42% -
    'inconclusive' - because it had zero register markers, zero Unicode
    artifacts and zero structural tells. Every vocabulary-based layer was
    blind to it. The syntax and stance layers exist because of this document.
    """

    def setUp(self):
        self.text = read(os.path.join(SAMPLES, "machine", "spec_cross_mission.txt"))

    def test_now_detected(self):
        rep = analyse(self.text)
        self.assertGreater(rep.probability, 0.75,
                           f"regressed on the marker-free spec: {rep.probability:.3f}")

    def test_lexical_layer_is_still_blind_to_it(self):
        """Documents why the new layers were needed: the old ones find nothing."""
        rep = analyse(self.text)
        self.assertEqual(rep.layer_totals.get("lexical", 0.0), 0.0)
        self.assertEqual(rep.layer_totals.get("structural", 0.0), 0.0)

    def test_carried_by_syntax_and_stance(self):
        rep = analyse(self.text)
        self.assertGreater(rep.layer_totals.get("syntax", 0.0), 0.5)
        self.assertGreater(rep.layer_totals.get("stance", 0.0), 0.5)

    def test_all_windows_flagged(self):
        frac, _ = fraction_ai(self.text)
        self.assertEqual(frac, 1.0)


class TestFormattingInvariance(unittest.TestCase):
    """Reformatting must not change the verdict.

    The same machine-written spec was submitted twice, once as plain lines and
    once with markdown bullets and backticked identifiers. The surface layers
    moved (markdown_density, section_density, paragraph_uniformity); the
    form-based layers were bit-identical. That is the property worth enforcing:
    anything measuring syntactic form should be invariant to presentation,
    because presentation is the easiest thing in the world to change.
    """

    @staticmethod
    def _markdownify(text):
        out = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                out.append("")
                continue
            # Backtick bare identifiers, bullet anything sentence-shaped.
            line = re.sub(r"\b(\w+_\w+)\b", r"`\1`", line)
            out.append("* " + line if stripped.endswith(".") else line)
        return "\n".join(out)

    def setUp(self):
        self.plain = read(os.path.join(SAMPLES, "machine", "spec_cross_mission.txt"))
        self.formatted = self._markdownify(self.plain)

    def test_verdict_does_not_flip(self):
        a = analyse(self.plain).probability
        b = analyse(self.formatted).probability
        self.assertGreater(a, 0.5)
        self.assertGreater(b, 0.5, "reformatting flipped the verdict")

    def test_form_based_layers_are_invariant(self):
        a = analyse(self.plain)
        b = analyse(self.formatted)
        for layer in ("syntax", "stance"):
            self.assertAlmostEqual(
                a.layer_totals.get(layer, 0.0), b.layer_totals.get(layer, 0.0),
                places=6, msg=f"{layer} layer is sensitive to formatting")

    def test_human_text_also_survives_markdownification(self):
        """The invariant must not be a one-way ratchet toward 'machine'."""
        human = read(os.path.join(SAMPLES, "human", "forum_debugging.txt"))
        self.assertLess(analyse(self._markdownify(human)).probability, 0.5)


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
