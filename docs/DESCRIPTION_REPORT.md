# Description report — cross-mission `blocked_by` specification

Analysis of a single submitted description. Run with the detector in this
repository at commit `fc3f0c6` plus the correction noted in section 5.

---

## Verdict

**Machine-generated. 86–98% depending on one contested signal. High confidence
either way.**

All four scoring surfaces agree — the first document in this corpus where they
do:

| surface | result |
|---|---|
| document level | **97.6%** machine |
| `fraction_ai` windows (250w) | **100%** machine — 79%, 85%, 78%, 73% |
| triage (200w windows) | **`ai`**, 10/10 windows machine, range 60.7–78.5% |
| segment scan | all machine |
| Burrows's Delta alone | +0.45, machine |

No surface, and no window of any size, read human.

---

## Which parts are machine-written

**All of it, uniformly.** That is the finding, and it is worth stating
precisely because it is not the same as "we checked each part".

Per-section scoring is not available here. The seven numbered sections run
54–139 words each, all below the 150-word floor where the form signals
activate and well below the 250-word threshold for a reliable verdict. Scoring
them individually returns 48–55% for every section — noise, not evidence, and
reporting those numbers as per-section findings would be fabrication.

What *is* measurable is that the document is **internally uniform**. Ten
overlapping 200-word windows span 60.7% to 78.5%, a range of 18 points with no
dip toward human anywhere. For comparison, the confirmed human-written
specification in this corpus swings from 19% to 69% across its windows. This
document has no human-textured passage in it.

---

## What marks it

Measured, strongest first, after the correction in section 5:

| signal | value | |
|---|---|---|
| `forensics.unicode_symbols` | +0.58 | **7 × U+2192 `→`, zero ASCII `->`** |
| `prompt.engineered_shape` | +0.55 | 7 numbered constraint blocks |
| `forensics.markdown_density` | +0.49 | 32 bullets, 7 numbered sections |
| `stylometry.burrows_delta` | +0.45 | function-word profile, content-free |
| `stance.term_invariance` | +0.43 | "mission" 28× at 42/1k, no variation |
| `stance.enumeration_density` | +0.38 | 89 commas/1k, no open-ended markers |
| `stance.authorial_absence` | +0.36 | one first-person marker in 666 words |

Against it: `statistical.burstiness` −0.62 (CV 0.64), which is the bullet
structure producing 16 short fragments rather than prose rhythm.

### The arrows

Seven `→` (U+2192) and zero ASCII `->`. Earlier specifications in this series
used ASCII throughout. U+2192 is not reachable from a keyboard without a
compose key, a snippet expansion, or a paste.

### The framing echo

```
confirmed-human spec : "Rough idea of what I want, open to ideas:-"
this document        : "Rough idea, open to better ones:"
```

Same opening move, same incident-then-numbered-proposal shape, same register.
The characteristic *phrasing* is reproduced; the characteristic *unevenness*
is not. The human specifications in this corpus sit at 8.6% and 10.7% template
repetition — this one at 20.1%.

Surface voice is copyable. Construction rhythm is much less so.

---

## 5. A correction, and the caveat it produced

The first version of this analysis led with `syntax.template_repetition` at
**20.1%** — the highest value in the corpus, above every confirmed machine
document. That was presented as the headline evidence.

Checking the matched templates showed they were mostly **identifier field
lists**, not prose parallelism:

```
x77   <w> <w> <w> <w> <w>
      "wp id ref reason malformed"
      "mission blockers wp id ref"
      "ref mission slug blocking wp"
```

The tokenizer splits `cross_mission_blockers` into three content words, so a
document that repeats field lists across sections manufactures identical
templates. Masking this document's 74 inline code spans drops it from 20.1% to
**7.7% — inside the human range.** Every other document in the corpus moves by
at most 1.2 points.

**Two fixes were attempted and both reverted**, because each was worse on net:

- *Masking backticked spans* fixed this document but made the signal depend on
  whether identifiers happen to be formatted as code, breaking the formatting-
  invariance property the test suite enforces.
- *Keeping snake_case whole in the tokenizer* fixed it at the root but dropped
  a confirmed machine document from 17.9% to 6.5%, into the human range, and
  made an existing false positive worse. Five tests failed.

So the signal keeps its current behaviour, with a **known confound documented
here**: on documents dense in repeated identifier lists, `template_repetition`
is inflated and should not be read as prose parallelism.

**Effect on the verdict.** Excluding that signal entirely, the document scores
**86.8%** instead of 97.6%, still "likely machine-generated" at high
confidence, carried by seven independent signals none of which depend on
identifiers. The conclusion does not rest on the contested measurement — but
the headline number was overstated, and 86.8% is the figure to quote.

---

## Confidence and limits

**What supports the verdict:** seven independent signals agree, four scoring
surfaces agree unanimously, and no window of any size reads human. The
register damping applied to technical prose is already in effect, so the
verdict is not an artifact of the genre bias documented in `LIMITS.md` §3c.

**What limits it:** this corpus contains twelve labelled documents. The
detector's measured accuracy is 10/12, and both of its errors are on technical
specifications — the same register as this document. A confident verdict here
sits in the category where the tool has demonstrably failed before, in both
directions.

**Not a basis for consequences.** See `docs/LIMITS.md` §7.
