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

## 3b. The corpus has a blind spot that matters

**There is no human-written technical writing in `samples/`.** The human
samples are a blog post, a forum post and a product review - all informal. Every
specification in the corpus is machine-written.

This was discovered while testing whether two gated signals
(`enumeration_density`, `term_invariance`) could safely fire on shorter text.
Measured at 190 words, comma density runs 37-44 per 1k for the human samples,
27-58 for machine prose, and **75-99 for the specs**. That looks like a clean
separation, and it is - but it separates by *genre*, not by authorship. A
human-written specification would sit at 75-99 too.

Those gates were therefore **left in place**. Lowering them would have improved
the score on the machine-written specs in the corpus while baking in the rule
"technical register implies machine", which is precisely the failure mode in
section 1 and section 3: it would misfire on exactly the people least able to
contest it.

**Update: one human-written technical specification has since been added**
(`samples/human/spec_goose_request.txt`, confirmed by passing an independent
production detector). It gives the first real measurement on this register,
and the result is mixed:

- **Document level: correct.** 29.1% out-of-sample, "leans human-written". Its
  AI-written counterpart on the same feature, same domain and same length
  scores 78.9%. A 60-point separation on a matched pair is the strongest
  single validation in this repo.
- **Triage level: a false positive.** `--triage` calls the same document
  `ai_assisted` and flags its API-naming section as machine. Window size is
  not the cause - 200/50 through 400/100 all do it. The prose signals have
  nothing to read in several hundred words of bare identifier lists, so local
  scores drift up even though the document as a whole is clearly human.

So: the document-level verdict is now validated on one human technical
document, and the span-level verdict is known to over-flag identifier-dense
regions. A test pins that false positive so it stays visible.

One document is not a validation set. **Still add more human-written technical
prose before trusting either verdict on that register.**

## 3c. A confirmed false positive, and what it cost

A second human-written technical specification (`samples/human/spec_dbc_linter.txt`,
confirmed by passing an independent production detector) scored **88.1%,
"likely machine-generated", high confidence**. That is a false positive on
exactly the register section 3b warned about, and it is the clearest evidence
in this repo that the warning was not theoretical.

**Cause.** Three signals measure *polish and impersonality* rather than
authorship - absence of first person, absence of questions, absence of typos.
A technical specification has all three whoever writes it:

```
                      identifiers/1k   presence/1k   authorial signal
human blog / review          0            66-89          -0.4 to -0.7
human goose spec            62            12             +0.41
human DBC spec              23             0.0           +1.14
```

**Fix.** `aidetect/register.py` detects technical register by identifier
density and damps those three signals to 35% there. The document went
88.1% -> 71.3% -> 60.4%, informal human samples moved by less than 0.3
points, and machine samples by less than 2.

**It is still wrong.** 60.4% is above the threshold. The remaining evidence is
diffuse - nothing above +0.40 - and the margin-aware confidence rule now
reports it as "low" rather than "high", but a human document still scores
machine-leaning. A test pins both the ceiling and the low confidence so the
defect stays visible.

**What this cost elsewhere**, recorded honestly: Burrows's Delta no longer
separates the corpus perfectly (one human document sits at +0.07), and triage
now calls the machine-written spec `ai_assisted` too. Both were artifacts of a
corpus with no hard cases in it.

**The windowed views were right and the document-level score was wrong.**
Triage scored 6 of 11 windows human on this document, `fraction_ai` 1 of 4,
and the segment scan 1 of 5, while the document score said 88.1%. Every one of
the damped signals has a length gate and switches off in a 200-word window, so
windowing had been avoiding the bias by accident. That is worth remembering
when a view disagrees with the headline number.

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
