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

## Tests

```bash
python3 -m unittest discover -s tests -v
```

35 tests: layer unit tests, crash-resistance on hostile Unicode, two regression
tests for bugs found during development, and an end-to-end separation test that
asserts a margin between the human and machine sample sets.

```
human max      22.1%
machine min    84.6%
margin         62.6 points
```

Six samples is a sanity check, not an evaluation. For real numbers, build a
corpus and use the calibrator.

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
