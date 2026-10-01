# Making this actually good

Honest assessment of where this detector stands and what moves it, ordered by
payoff per unit of effort. The short version: we are at the end of what
hand-written features can do. Everything above Tier 2 here is worth more than
any further heuristic.

## Where we actually are

| | status |
|---|---|
| Signals | 30, hand-weighted |
| Corpus | 7 documents |
| Validation | **in-sample** — constants fit on the same 7 docs we report margins on |
| Out-of-domain accuracy | **unknown** |
| Adversarial robustness | **untested**; paraphrasing almost certainly defeats it |
| AI-assisted detection | **not supported** |

That "in-sample" row is the problem. We cannot currently tell whether a change
helps. Every number in the README is optimistic by an unknown amount.

---

## Tier 0 — the blocker: build a corpus

Nothing else can be measured until this exists. It is unglamorous and it is the
single highest-value work item.

**Target:** 500+ documents per class minimum, 2000+ preferred, matched to the
domain you actually care about. Essay detectors do not transfer to spec prose;
we proved that on ourselves.

**Human side — must be provably pre-2022** to avoid contamination:
- Project Gutenberg, pre-2021 Stack Exchange and Reddit dumps, old arXiv
- Internal documents with known authorship and timestamps (best, if available)

**Machine side — generate paired data:** same prompts, multiple models
(Claude, GPT, Gemini, Llama, Qwen, DeepSeek), multiple temperatures, multiple
register instructions including "write tersely", "write like a human", "vary
your sentence length". If the corpus only contains default-voice output, the
detector only catches default-voice output.

**Split discipline:**
- train / validation / **held-out test**
- The test set is opened **once**, at the end. Not for tuning. Not "just to
  check". The moment it informs a constant it stops being a test set.
- Split by *document source*, not randomly, or near-duplicates leak across.

**Label the third class** while you are at it: `human`, `ai`, `ai_assisted`,
with span-level marks where a document is mixed. You cannot build the
AI-assisted classifier later without this.

## Tier 1 — model-backed scoring (the single biggest accuracy jump)

Everything in layers 0-8 is a proxy for what a language model measures
directly. Published zero-shot AUROC for the methods below runs ~0.95+; our
feature heuristics out of domain are likely closer to 0.75-0.85.

1. **Binoculars** — already stubbed in `aidetect/probe/`. Perplexity normalised
   by cross-perplexity between two related models. Best zero-shot method, and
   specifically the one with a tolerable false-positive rate on non-native
   English, which is the failure mode that matters most. Needs ~14GB and a GPU.
2. **Fast-DetectGPT** — conditional probability curvature, one forward pass.
   ~340x faster than DetectGPT and usually more accurate. Good second signal.
3. **Ghostbuster-style** — run text through several *weak* LMs, combine
   per-token probability features, train a classifier on the combination.
   Robust because it never needs access to the generating model.

Wire the probe score in as one more `Signal` and the existing ensemble machinery
handles it unchanged.

## Tier 2 — learn the combination instead of hand-weighting it

`aidetect/calibrate.py` already fits logistic regression over the signal vector.
With a real corpus:

1. Upgrade to gradient boosting over `[all 30 signal logodds] + [probe scores]`.
2. **Fine-tune a small transformer** (DeBERTa-v3-base, ModernBERT) directly on
   the corpus. A few thousand examples, 1-2 GPU-hours. This is most likely what
   the production detector we were compared against is doing.
3. Ensemble: learned classifier + Binoculars + the interpretable features. Keep
   the features even when the model wins — they are what makes a score
   explainable to the person it is about.

## Tier 3 — robustness, or it dies on contact with reality

1. **Adversarial training.** Paraphrase every machine sample (DIPPER, or just
   ask a model to rewrite) and add those as positive examples. Without this,
   one paraphrasing pass defeats the whole system.
2. **RAIDAR.** Ask a model to rewrite the input and measure edit distance. It
   rewrites *human* text heavily and *machine* text barely, because machine text
   already looks like its own output. Cheap with API access and fully orthogonal
   to everything we currently measure.
3. **Register/genre conditioning.** "Zero connectives" means opposite things in
   narrative and in spec prose — we hit exactly this bug. Detect genre first,
   then apply a genre-specific weight profile.
4. **Mixed-document sequence labelling.** Reframe from document classification
   to span labelling. This is what `--segments` gestures at and what
   `fractionAiAssisted` requires.

## Tier 4 — evaluation and deployment discipline

1. **Metrics that mean something:** AUROC, and **FPR at fixed TPR** (e.g. what
   is our false-positive rate at 90% detection?). "Accuracy on 7 documents" is
   not a metric.
2. **Per-domain breakdown.** One number hides everything.
3. **Fairness evaluation — this should gate deployment.** Measure FPR
   separately on non-native-English writing. If it is materially higher than on
   native-speaker writing, that is the headline result, not a footnote. Detectors
   have historically flagged >50% of non-native essays as AI.
4. **Calibrated abstention.** Conformal prediction gives a guaranteed error rate
   with an explicit "I don't know" band. For any use touching people, abstaining
   is a feature.
5. **Keep the evidence output.** A score nobody can interrogate is unusable in
   any process where the subject gets to respond.

## Tier 5 — narrow but free when applicable

- **Watermark detection.** SynthID-Text is deployed in Gemini. Statistically
  airtight when present; useless otherwise, and you generally need the key.
- **Provenance/metadata forensics.** Document version history, editing time,
  paste-shaped revisions. *Stronger than any stylometric signal* when you have
  it, and far more defensible to the person being assessed.
- **Multilingual.** Every constant here is English-specific.

---

## What NOT to do

- **Add more regexes.** We are past the point where that helps; it now mostly
  adds overfit risk and noise.
- **Tune against a single example.** Including the ones in this repo.
- **Report in-sample numbers as performance.** We currently do, and the README
  says so explicitly.
- **Ship it as a decision-maker about people.** See `LIMITS.md`.

## Suggested order

```
1. Corpus + held-out split              <- blocks everything
2. Evaluation harness (AUROC, FPR@TPR)  <- unblocked, buildable now
3. Binoculars wired in                  <- biggest single accuracy jump
4. Learned combination over features
5. Adversarial + paraphrase training
6. Fairness evaluation                  <- gates any deployment
7. AI-assisted span labelling
```

Steps 1 and 3 need resources this container does not have (data, a GPU). Step 2
needs neither and should come first, because until it exists we cannot tell
whether any later change helped.
