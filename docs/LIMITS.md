# Limits — read this before acting on any score

This detector outputs **evidence for a human to weigh**, never a verdict to act
on automatically. That is not legal boilerplate; it is the honest description of
what the technology can do.

---

## 1. The fairness problem is the most important thing on this page

Liang et al. (2023) tested seven GPT detectors on TOEFL essays written by
non-native English speakers. **Over half were misclassified as AI-generated.**
Essays by native speakers were classified near-perfectly.

The cause is structural, not a bug in any one product. Detectors reward low
perplexity and low lexical variety. A person writing competently in their second
language produces exactly that profile — simpler syntax, a smaller working
vocabulary, more predictable phrasing. The detector cannot distinguish "written
by a model" from "written carefully by someone with a smaller English
vocabulary", because on every measurable axis they look the same.

**Consequence:** any process that penalises people based on this score will
penalise non-native speakers, disabled writers using assistive tools, and people
taught to write formulaically — at a far higher rate than everyone else. That is
a product and policy decision, and it belongs to whoever deploys this, not to
the model.

The signals most implicated are `statistical.lexical_diversity` and
`statistical.burstiness`. They are deliberately down-weighted in this
implementation, which costs accuracy and is the right trade.

## 2. What defeats this detector completely

| Attack | Effort | Effect |
|---|---|---|
| Paraphrasing tools (DIPPER and similar) | seconds | Destroys lexical + most statistical signals |
| "Write like a human, vary sentence length, include a typo" | one prompt | Moves burstiness into the human range |
| Ten minutes of human editing | minutes | Kills the forensic layer outright |
| Retyping instead of pasting | minutes | Removes every Unicode signal |
| Translation round-trip | seconds | Normalises the register away |

Anyone who wants to evade this will evade it. The detector is useful against
**unmodified** machine output, which in practice is most of it — but a
determined adversary wins, always, and no future version changes that.

## 3. False positives you should expect

- **Non-native English speakers** (see §1 — the big one).
- **Technical and legal writing**, which is legitimately formulaic, hedged and
  uniformly structured.
- **Corporate/marketing copy**, written by humans trained into the exact
  register LLMs imitate.
- **Heavily edited prose.** Copy-editing removes typos, regularises sentence
  length and standardises punctuation — it makes human text look machine-made.
- **Anything pasted through Word, Google Docs or a CMS**, which silently
  converts straight quotes to curly ones and hyphens to em dashes. The entire
  forensic layer is then measuring the word processor, not the author.
- **Writing about AI writing.** This repo's own `docs/RESEARCH.md` scores above
  99% machine-generated, because it quotes every marker in the lexicon. Any
  document discussing these markers will trip them.

## 3a. A documented false negative

The detector originally scored a 658-word machine-written technical
specification at 42% — "inconclusive". It had no register markers, no Unicode
artifacts and no structural tells; terse identifier-dense spec prose strips out
everything the vocabulary-based layers measure.

The `syntax` and `stance` layers were added in response and it now scores 96%.
But the general lesson stands and is not fixed: **compressed technical registers
defeat surface-feature detection**, and there are certainly other registers that
do the same which are not yet in the sample corpus. A null result from this tool
means "no applicable evidence found", never "human".

## 4. False negatives you should expect

- Short text. Below ~250 words the detector withholds judgement on purpose;
  below 40 words it refuses outright and returns 0.5.
- Models prompted for an unusual voice or persona.
- Mixed documents where machine text is a small fraction — use
  `--segments` for these, not the whole-document score.
- Newer models. Register markers drift with every release; `delve` was a strong
  2023–2024 signal and is already weakening as vendors tune it out.

## 5. The score is not a probability of guilt

`probability = 0.85` means *the accumulated evidence is consistent with machine
generation at this weighting*. It does **not** mean "85% chance this person
cheated". The weights shipped here are hand-set priors from the literature, not
a fit to your population. Until you run `aidetect.calibrate` on several hundred
labelled documents from **your own domain**, the number is ordinal at best: it
ranks documents sensibly, but its absolute value means little.

## 6. Use it for these things

- **Triage.** Rank a large pile so a human reads the most suspicious first.
- **Localisation.** `--segments` finds *which paragraphs* were pasted into an
  otherwise human document. This is the strongest practical use, because the
  output points at a specific span a human can evaluate.
- **Self-review.** Check your own drafts for register markers before publishing.
- **Corpus measurement.** Estimate machine-generated share of a dataset in
  aggregate, where individual errors wash out.
- **Provenance forensics.** The Unicode layer answers "was this pasted from a
  chat UI", which is a narrower and much better-evidenced question than "was
  this written by AI".

## 7. Do not use it for these things

- Automated disciplinary or academic-misconduct decisions.
- Hiring, admissions, or any gate on a person's opportunities.
- Any decision where the subject cannot see the evidence and respond to it.
- Any single-document high-stakes call, at all.

If you need defensible provenance, invest in **process evidence** instead:
document version history, editor telemetry, draft timestamps. A Google Doc that
appeared in two paste-shaped revisions with ninety seconds of editing time is
far stronger evidence than any stylometric score, and it is evidence the subject
can actually contest on the merits.
