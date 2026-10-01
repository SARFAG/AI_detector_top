# aidetect

A layered detector for machine-generated text that **reports evidence, not
verdicts**. Zero dependencies, pure Python 3.8+.

It answers "which signals point which way, and how strongly" — then shows you
every one of them, so a human can judge. Read [docs/LIMITS.md](docs/LIMITS.md)
before you act on any score; the failure modes are specific and they matter.

```
$ python3 -m aidetect samples/machine/chatgpt_remote_work.txt

  machine-generated probability :  99.9%
  [########################################]
  verdict                       : very likely machine-generated
  confidence                    : high
  words analysed                : 362

  evidence FOR machine authorship:
    lexical.tier4_assistant_leakage    + 2.40  'certainly!'; 'i hope this helps'
    lexical.tier1_register             + 1.74  'delve'; 'multifaceted'; 'pivotal'
    lexical.tier3_phrases              + 1.67  'a comprehensive overview'; 'let's delve'
    structural.answer_shape            + 1.15  opens by restating the prompt; headed sections; closes with a synthesis
    structural.mandatory_conclusion    + 0.88  'In conclusion'; in final 20% of document
    statistical.burstiness             + 0.69  CV=0.38 (mean 13.4 words, sd 5.1, n=27)

  evidence FOR human authorship:
    statistical.lexical_diversity      -0.35  MATTR-50=0.889 (high lexical variety)

  possible source model (weak evidence, drifts per release):
    ChatGPT        score=3.0    'certainly!'; 'let me know if you'd like me to'
```

## Install

Nothing to install.

```bash
git clone https://github.com/SARFAG/AI_detector_top
cd AI_detector_top
python3 -m aidetect your_file.txt
```

## Usage

```bash
python3 -m aidetect essay.txt              # full evidence report
python3 -m aidetect --json essay.txt       # machine-readable
python3 -m aidetect --all essay.txt        # every signal, including neutral ones
python3 -m aidetect --segments mixed.txt   # locate pasted spans (see below)
cat draft.md | python3 -m aidetect         # stdin

python3 -m aidetect --threshold 0.8 *.txt  # exit 1 if any file scores >= 0.8
```

As a library:

```python
import aidetect

report = aidetect.analyse(text)
report.probability        # 0.0 - 1.0
report.verdict            # 'likely machine-generated'
report.confidence         # none | low | medium | high
report.attribution        # [('ChatGPT', 4.0, ["'certainly!'", ...])]

for signal in report.top_signals:
    print(signal.name, signal.logodds, signal.evidence)
```

## The feature that actually matters: `--segments`

Whole-document scoring is weak on mixed writing, which is most real writing — a
person writes an essay and pastes two paragraphs from a chat. The document
averages out to "inconclusive" and tells you nothing.

Segment scanning scores a sliding window of paragraphs, so the paste localises:

```
$ python3 -m aidetect --segments mixed.txt

WHOLE DOCUMENT: 44.8% (inconclusive)

   31.3%  ok so I finally figured this out after like 6 hours and I want...
   12.1%  The fix was two lines. Set an open_timeout and a read_timeout...
   19.9%  Anyway. Hope this helps someone. If you're seeing random pool...
   79.9%  # The Ultimate Guide to Productivity in the Modern Workplace...  <-- MACHINE
   86.3%  In today's ever-evolving business landscape, productivity has...  <-- MACHINE
```

This is the strongest practical use of the tool: it points at a specific span a
human can read and evaluate, rather than producing a number about a whole
document.

## How it works

Six layers, each emitting `Signal` objects that carry a **log-odds
contribution** plus the evidence that produced it. The detector sums log-odds
and applies a sigmoid, so every point of the final probability traces back to a
quoted span of input.

| Layer | Measures | Precision / Recall |
|---|---|---|
| `forensics` | Unicode, smart quotes, em dashes, invisible chars, markdown leakage | high / low |
| `lexical` | Tiered register lexicon, negative parallelism, model attribution | medium / medium |
| `statistical` | Burstiness, paragraph uniformity, repetition, connective density | medium / high |
| `structural` | Mandatory conclusions, answer shape, section balance | low / high |
| `code` | Docstring uniformity, generic names, absent human traces | medium / medium |
| `syntax` | Syntactic template repetition, clause uniformity | **high / high** |
| `stance` | Authorial absence, term invariance, enumeration density | high / medium |
| `stylometry` | Burrows's Delta over function words, nearest-centroid | high / medium |
| `discourse` | Sentence-type mix (3 further features measured and rejected) | low / medium |
| `rewrite` | RAIDAR rewriting distance — needs a model callable | high / high |
| `probe` *(optional)* | Binoculars / true perplexity — needs PyTorch | high / high |

Design rules that are enforced in code:

1. **Short text cannot convict.** Evidence is shrunk below ~250 words; below 40
   words the detector refuses and returns 0.5.
2. **Bounded evidence.** Every signal saturates, so length alone can never
   accumulate a conviction.
3. **A human prior.** Base log-odds start negative — a false accusation costs
   far more than a miss.
4. **Evidence always travels with the score.**

The reasoning behind every signal is in [docs/RESEARCH.md](docs/RESEARCH.md).

## Classical stylometry and authorship verification

`stylometry.py` implements **Burrows's Delta** — the canonical authorship
method, which reads only function-word rates and ignores content entirely.
That makes it near-orthogonal to every other layer, which is why ablation
ranks it second overall despite centroids fitted on seven documents.

It also exposes the stronger and fairer question:

```python
from aidetect.stylometry import verify_authorship
verify_authorship(candidate_doc, their_past_writing, other_peoples_writing)
```

**Authorship verification** asks "does this match *this person's* known
writing?" rather than "does this look machine-written?". Comparing someone
against their own baseline sidesteps the fairness problem in `LIMITS.md`
entirely: writing simply, or in a second language, is no longer evidence of
anything, because the comparison is to themselves. If you have a person's
prior work, prefer this over AI detection.

## RAIDAR (rewriting distance)

Ask a model to rewrite the text and measure how much changed. It rewrites
*human* text heavily and *machine* text barely, because machine text already
looks like its own output. Needs no corpus, lexicon or register assumption —
the weakness that defeated this detector on terse technical prose.

```python
from aidetect.rewrite import raidar_signal
raidar_signal(doc, lambda t: call_your_model(t))
```

The model call is yours to supply; this module never touches the network.

## Optional: the model-backed layer

Layers 0–4 are proxies for what a real language model measures directly. For
production accuracy, add **Binoculars** — perplexity normalised by
cross-perplexity, the best zero-shot method, and specifically the one with a
tolerable false-positive rate on non-native English:

```bash
pip install torch transformers
```

```python
from aidetect.probe import Binoculars
b = Binoculars()                  # ~14GB on first use
print(b.score(text))              # LOW score => machine-generated
print(b.as_signal(text))          # slots into the normal report
```

## Calibrating to your domain

The shipped weights are hand-set priors from the literature, **not** a fit to
your data. Detector accuracy is dominated by domain. Fit your own:

```bash
python3 -m aidetect.calibrate --human corpus/human --machine corpus/machine
```

Collect ~200 documents per class from your own domain first. Below that you are
fitting noise, and the tool will say so.

## Three-class triage (`--triage`)

Matches the output shape of production detectors: `human` / `ai_assisted` /
`ai`, plus the **word offsets** of the machine-looking spans.

```bash
python3 -m aidetect --triage mixed.txt
```

```
  prediction          : ai_assisted
  fraction_assisted   : 0.47
  words               : 878
  machine spans       : 2
    words     0-100   admin. Council tax, the energy switch, the post redirect ...
    words   500-878   convinced it was pgbouncer. It wasn't pgbouncer. I want ...
```

Spans come from **per-word probability aggregation**, not binary window flags:
each window contributes its probability to every word it covers, and each word
takes the mean. Measured on synthetic mixed documents this lifts span recall
from ~0.45 to 0.83 at similar precision, because a window straddling a seam is
diluted below threshold and binary flagging loses its machine half entirely.

Validated against `aidetect/synth.py`, which splices known-human and
known-machine text into documents whose machine spans are known *exactly*:

| | |
|---|---|
| document class agreement | 12/14 |
| span precision | 0.78 |
| span recall | 0.83 |
| span F1 | 0.80 |
| span IoU | 0.70 |

**Read those numbers with two caveats.** Spliced text has hard seams where real
assisted writing blends, so this is an upper bound on real performance. And
localisation is only stride-accurate — spans smear by up to ~50 words at the
edges, short inserts are missed, and an excerpt printed at a span boundary will
often show human text.

## Evaluation

```bash
python3 -m aidetect.evaluate --human samples/human --machine samples/machine --ablate
```

Reports AUROC with bootstrap CI, a permutation test **and its own floor**,
FPR at fixed TPR, log loss, separation margin, calibration error, and a
leave-one-out ablation. It also prints what is wrong with its own numbers:

```
!! LEAVE-ONE-OUT HERE IS NOT FULLY HONEST.
   It refits the logistic weights without each document, but the FEATURE
   DEFINITIONS were hand-tuned on the whole corpus and are not re-derived
   per fold.

!! The bootstrap interval is degenerate (lo == hi). With perfect separation
   and small n, every resample also separates perfectly, so the interval
   collapses. Read that as the bootstrap being unable to express uncertainty
   here, NOT as precision.
```

Current result on 7 documents: AUROC 1.000 in-sample and out-of-sample,
overfitting gap 0.0000, permutation p=0.0288 against a floor of 0.0286 — i.e.
the best p-value this sample size can produce, which is a statement about the
corpus, not the detector.

Ablation ranks by log loss, because AUROC saturates at 1.0 and returns a column
of zeros:

| signal | log-loss cost |
|---|---|
| `syntax.template_repetition` | **+0.0396** |
| `stylometry.burrows_delta` | **+0.0205** |
| `stance.authorial_absence` | +0.0073 |
| `statistical.burstiness` | +0.0064 |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

69 tests: layer unit tests, metric tests against hand-computed values,
crash-resistance on hostile Unicode, regression tests for every bug found
during development, formatting-invariance tests, and an end-to-end separation
test that asserts a margin between the human and machine sample sets.

```
human max      18.6%
machine min    97.7%
margin         79.0 points   (in-sample - see below)
```

Eight samples is a sanity check, not an evaluation, and that margin is
**in-sample**: the `syntax.template_repetition` midpoint and the stylometry
centroids were fit to these documents. For real numbers, build a corpus and
use the calibrator.

### The matched pair

The most informative validation here is two specifications of the *same
feature*, same domain, same length — one AI-written, one human-written and
confirmed by passing an independent production detector:

| | score |
|---|---|
| AI-written version | **78.9%** likely machine-generated |
| human-written version | **18.6%** likely human-written (29.1% before it joined the corpus) |

A 60-point separation on matched content is worth more than the headline
margin, because topic, genre, length and format are all held constant.

### The feature that matters most

`syntax.template_repetition` abstracts every token to a shape class and measures
what share of 5-token *templates* recur — catching parallelism that word-level
n-grams cannot see, because the vocabulary differs while the construction
repeats. Measured on fixed 200-word windows so it is not a length detector in
disguise.

| | windowed template repeat-share |
|---|---|
| human samples | 2.6–6.4% |
| machine samples | 13.8–19.9% |

Over 2x separation with no overlap. It was added after the detector missed a
machine-written technical specification that had zero register markers, zero
Unicode artifacts and zero structural tells — see
[docs/RESEARCH.md §11–14](docs/RESEARCH.md) for that failure case and what it
taught.

## Honest limitations

Paraphrasing tools defeat this. So does ten minutes of editing, or asking the
model to vary its sentence length. Non-native English speakers draw false
positives at a much higher rate — this is a structural property of every
stylometric detector, not a bug to be fixed later. The research doc in this repo
scores 99% machine-generated because it quotes every marker in the lexicon.

Use it for triage, localisation and self-review. Do not use it to make decisions
about people. [docs/LIMITS.md](docs/LIMITS.md) spells out why.

## Layout

```
aidetect/
  signals.py      Signal type, sigmoid, saturating evidence bound
  text.py         tokenisation, sentence/paragraph splitting
  lexicon.py      marker DATA - add new models here, not in code
  forensics.py    layer 0  unicode & typography
  lexical.py      layer 1  register markers + attribution
  statistical.py  layer 2  burstiness & distribution
  structural.py   layer 3  document shape
  code.py         layer 4  source-code tells
  prompts.py      layer 5  prompt authorship + injection surface
  probe/          layer 6  optional, model-backed (Binoculars)
  detector.py     the ensemble
  calibrate.py    fit weights from labelled data
  cli.py          command line
docs/
  RESEARCH.md     why each signal exists
  LIMITS.md       when not to trust it
  CLOUD_SESSION.md  developing this in a Claude Code cloud session
```

## License

MIT
