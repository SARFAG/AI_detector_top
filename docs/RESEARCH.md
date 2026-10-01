# How LLM Writing Differs From Human Writing

Research notes backing the detector in this repo. Every claim here maps to a
signal in `aidetect/`. Read `docs/LIMITS.md` before trusting any score.

---

## 0. The one-paragraph summary

LLMs sample from a probability distribution that has been flattened by RLHF
toward a "helpful, safe, well-organised" register. The result is text that is
**too likely**: each next word is closer to the most predictable choice than a
human's would be, variance between sentences is compressed, structure is
imposed where humans would ramble, and a small set of register-marking words is
massively over-represented. Humans are *bursty* — they mix a nine-word sentence
with a forty-word one, drop a typo, change their mind mid-paragraph, and leave
the ending unresolved. Nearly every usable detection signal is a measurement of
one of those two facts.

---

## 1. Statistical / information-theoretic signals (the strongest family)

### 1.1 Perplexity
Perplexity = how surprised a reference language model is by the text. Machine
text has systematically **lower** perplexity, because it was literally sampled
from a similar distribution. This was the original GPTZero signal.

*Weakness:* a human writing plainly about a common topic also scores low. Raw
perplexity alone has a terrible false-positive rate.

### 1.2 Burstiness (variance of perplexity and of sentence length)
Humans vary wildly sentence to sentence. LLMs regress to the mean. Measuring
the **standard deviation** of per-sentence perplexity, or just the coefficient
of variation of sentence lengths, separates the classes better than the mean
does. Implemented without a model in `aidetect/statistical.py`
(`sentence_length_burstiness`), and with a model in the optional probe.

### 1.3 DetectGPT / Fast-DetectGPT (probability curvature)
Insight: machine text sits on a **local maximum** of the model's log-probability
surface. Perturb it slightly (mask-and-refill with T5) and log-prob drops a lot;
do the same to human text and it drops much less. Score = that gap.

DetectGPT needs ~100 perturbations per sample (slow). **Fast-DetectGPT**
replaces perturbation with an analytic *conditional probability curvature* —
one forward pass, ~340x faster, usually better AUROC. This is the method to
implement if you add a GPU.

### 1.4 Binoculars (best zero-shot method as of this writing)
Ratio of two quantities computed from a pair of closely related models (e.g.
Falcon-7B and Falcon-7B-Instruct):

```
Binoculars score = perplexity(text | observer) / cross-perplexity(observer, performer)
```

The denominator normalises away "this text is just inherently predictable",
which is exactly the failure mode of raw perplexity. It is the main reason
Binoculars holds a low false-positive rate on non-native-English and technical
text where other zero-shot detectors collapse. **Low score ⇒ machine.**

Needs two open-weights models in memory. See `aidetect/probe/` for the hook.

### 1.5 Ghostbuster
Rather than one model, run text through several **weak** language models, take
the per-token probability vectors, then search over a space of combined
features (sums, ratios, differences) and train a classifier on the best ones.
Robust because it never depends on access to the *generating* model.

### 1.6 RAIDAR (rewriting distance)
Ask an LLM to "rewrite this text". It edits **human** text heavily and
**machine** text barely at all — the machine text already looks like its own
output, so there is nothing to fix. Edit distance before/after is the feature.
Cheap to implement if you already have API access, and surprisingly strong.

### 1.7 Watermarking
Kirchenbauer et al.: at generation time, hash the previous token to seed a
pseudorandom split of the vocabulary into a "green list" and "red list", then
bias sampling toward green. A detector counts green tokens and runs a z-test.
Statistically airtight **when present**.

Reality check:
- Google deployed **SynthID-Text** in Gemini; it is the only widely shipped one.
- OpenAI built a watermark and did not ship it broadly.
- Anthropic does not watermark Claude output.
- Watermarks survive light editing, die under heavy paraphrase or translation.
- You usually need the detector key, which vendors do not hand out.

**Conclusion: you cannot build a general detector on watermarks.** Treat it as a
bonus confirmation channel, not a foundation.

---

## 2. Lexical signals — the "LLM register"

Post-2023 corpus studies of academic abstracts, Wikipedia edits and product
reviews found sharp frequency spikes in a consistent vocabulary. These words
are not *wrong*; they are over-produced by 10–50x relative to pre-2022
baselines.

**Tier 1 (strongest, near-diagnostic in volume):**
`delve`, `intricate`, `tapestry`, `realm`, `underscore(s)`, `pivotal`,
`testament to`, `multifaceted`, `nuanced`, `meticulous`, `garner`,
`showcasing`, `ever-evolving`, `landscape of`, `navigate the complexities`,
`in the realm of`, `plays a crucial role`, `it's worth noting`,
`a comprehensive overview`

**Tier 2 (common in business English too — only count in aggregate):**
`leverage`, `robust`, `foster`, `holistic`, `seamless`, `streamline`,
`empower`, `harness`, `crucial`, `vital`, `significant`, `enhance`,
`facilitate`, `paradigm`, `synergy`, `cutting-edge`, `game-changer`

**Tier 3 — structural phrases:**
- `It's not just X, it's Y` / `This isn't about X — it's about Y`
  (**negative parallelism**; one of the single strongest phrase-level tells)
- `While X, it's important to note that Y` (symmetric both-sides hedging)
- Rule-of-three triads: `fast, reliable, and secure`
- `In conclusion,` / `Ultimately,` / `In summary,` / `The key takeaway is`
- `Let's dive in` / `Let's explore` / `Here's a breakdown`
- `I hope this helps!` / `Feel free to` / `Let me know if you'd like`

**Tier 4 — assistant-frame leakage (a paste gave it away):**
`As an AI language model`, `I don't have personal opinions`,
`As of my last knowledge update`, `I cannot browse the internet`,
`Certainly!`, `Great question!`, `Of course!`

Implemented in `aidetect/lexicon.py` with tiered weights so Tier 2 words never
convict on their own.

---

## 3. Per-model attribution

These separate *which* assistant, given that text is already machine-flagged.
They are stylistic tendencies, not guarantees, and they drift with every
release — treat as weak evidence and re-measure against fresh samples.

### ChatGPT (OpenAI)
- Bulleted lists where **every** bullet opens with a bolded lead-in: `- **Speed:** ...`
- `Here's a breakdown` / `Here's a quick rundown` openers
- Emoji used as section markers (🚀 ✅ 🔑 💡) in otherwise formal text
- Comparison tables with an `Aspect | Description` shape
- `Certainly!` openers; `Let me know if you'd like me to expand on any of these!` closers
- Narrow no-break space (U+202F) before punctuation in some locales — a **web-UI
  rendering artifact**, strong paste evidence
- Heavy em-dash use

### Claude (Anthropic)
- Longer uninterrupted prose; reaches for lists less eagerly than ChatGPT
- Visible hedging and self-qualification: `I should note`, `That said,`,
  `Worth noting:`, `to be fair`, `I'd push back gently on`
- Restates/reframes the question before answering
- `Here's the thing:` as a pivot
- Sparse emoji; `##` headers plus bold rather than emoji headers
- Explicit uncertainty calibration (`I'm not certain, but`) more than peers

### Claude Code (Anthropic, agentic CLI)
The most *forensically* detectable because it leaves artifacts in repos:
- Commit trailers: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
  and `Co-Authored-By: Claude <noreply@anthropic.com>`
- PR bodies ending in a session URL under `https://claude.ai/code/...`
- `CLAUDE.md` at repo root
- `.claude/` directory (settings.json, skills/, agents/, commands/)
- Commit messages with a summary line plus a bulleted body of what changed
- Occasional `<thinking>` / tool-block leakage into committed text

### Gemini (Google)
- `Of course.` as a standalone opener line
- Very deep nested bullets with aggressive bolding
- `It's important to remember that` / `Here's a comprehensive overview`
- SynthID-Text watermark present (not publicly verifiable)

### Note on product names
The user asked about "ChatGPT Astra" among others. Product/codename churn is
fast and I will not invent fingerprints for a product I cannot verify. The
right move is in §7: collect your own labelled samples and let
`aidetect/calibrate.py` learn the fingerprint empirically. The architecture
treats per-model profiles as **data**, not code, exactly so new models can be
added without touching the detector.

---

## 4. Typographic & Unicode forensics (highest precision, lowest recall)

These rarely fire, but when they do they are close to proof of a copy-paste out
of a web chat UI rather than a keyboard.

| Codepoint | Name | Why it matters |
|---|---|---|
| U+2014 `—` | EM DASH | Assistants emit it constantly; most people never type it on a phone or a plain keyboard |
| U+2013 `–` | EN DASH | Same, in ranges |
| U+2018/2019 `' '` | curly quotes | Keyboards make straight `'`; curly means a rich-text pipeline or an LLM |
| U+201C/201D `" "` | curly double quotes | Same |
| U+00A0 | NBSP | Rendering artifact from a web UI |
| U+202F | NARROW NBSP | Strong ChatGPT web-UI paste signal |
| U+200B/200C/200D | zero-width chars | Invisible; sometimes deliberate watermarking |
| U+FEFF | BOM mid-document | Paste artifact |
| `•` `→` `≈` `×` | Unicode symbols in prose | Markdown-ish rendering leaking into plain text |

**The mixed-signal rule:** curly quotes mixed with straight quotes in the same
document means a human pasted an LLM paragraph into their own writing. That
mixture is a stronger signal than either state alone, and it localises *which
paragraph* came from the model. `aidetect/forensics.py` reports the split.

Also under this heading:
- Zero typos across 1500+ informal words is itself anomalous.
- No double spaces, no missing apostrophes, perfect capitalisation of `I`.
- Markdown (`**bold**`, `### head`, `---`) surviving into a context that does
  not render markdown (an email, a forum post, a Word doc).

---

## 5. Structural signals

- **Uniform paragraph lengths.** Humans write a one-line paragraph then a
  twelve-line one. LLMs produce 3–5 sentence blocks over and over.
- **The mandatory conclusion.** LLMs almost always synthesise at the end, even
  when nothing needs synthesising. Humans often just stop.
- **Header/bullet density** far above the base rate for the genre.
- **Answer-shaped structure:** intro sentence restating the question → 3–5
  headed sections → conclusion. The shape of an assignment answer.
- **Balanced coverage.** Equal space to each sub-topic. Humans are lopsided and
  spend most of their words on the part they actually care about.
- **No unresolved threads.** Humans leave a tangent dangling; LLMs tie off.

---

## 6. Code-specific signals

Writing is not the only output. For source code:

- Docstrings on **every** function, in one consistent style, with the parameter
  list restating the signature.
- Comments that restate the code (`# increment the counter`) rather than
  explain *why*.
- Generic identifiers: `result`, `data`, `items`, `processed_data`, `temp`.
- An appended `if __name__ == "__main__":` demo block nobody asked for.
- Defensive `try/except` around operations that cannot fail.
- Type hints applied with total consistency (humans are patchy).
- **Absence** of human traces: no commented-out experiments, no `XXX`/`HACK`,
  no personal or in-joke names, no dead code, no inconsistent indentation.
- Emoji in `print()` / log strings (`print("✅ Done!")`).
- Hallucinated APIs — functions that read plausibly but do not exist.

Git-level: a single enormous initial commit, commit messages in perfect
imperative mood with bulleted bodies, zero `wip`/`fix typo` commits.

---

## 7. Prompt detection (asked about explicitly)

Detecting *AI-written prompts* (as opposed to prose) is a different problem.
Machine-authored prompts show:
- Role framing: `You are a world-class ...`, `Act as a ...`
- `Your task is to` / `Your goal is to`
- Numbered constraint stacks, often with `Do NOT` in caps
- An explicit `Output format:` / `Respond in JSON` section
- Delimiter theatre: `###`, `---`, `<instructions>` XML tags
- Politeness-as-incentive artifacts (`This is very important to my career`)
- Over-specified persona plus over-specified format, together

Handled by `aidetect/prompts.py`. This also doubles as a lightweight
prompt-injection surface scan, since injections share the imperative-stacking
shape.

---

## 8. Metadata forensics (outside the text)

When you have more than a text blob, these beat every textual signal:
- **Google Docs version history** — a document that appears in 2–3 giant
  revisions with no editing time was pasted.
- **DOCX** `app.xml` `TotalTime` near zero for a long document.
- **Keystroke dynamics / paste events** if you control the editor.
- **Clipboard-sized insertions** in any revision-tracked system.

If you are building this for a real institution, invest here first. It is more
defensible than any statistical score and far harder to argue with.

---

## 9. What defeats all of the above

Be honest about this with anyone who uses the tool.

1. **Paraphrasing attacks.** DIPPER and similar paraphrasers destroy lexical
   and most statistical signals while preserving meaning.
2. **"Write like a human" prompting.** Asking for typos, varied sentence
   length and informal register moves burstiness into the human range.
3. **Light human editing.** Ten minutes of a person rewriting kills the
   forensic layer entirely.
4. **Translation round-trips.** Normalises away the register.
5. **Domain shift.** A detector tuned on essays misfires badly on technical
   documentation, legal text, or non-native English.

And the critical fairness finding: Liang et al. (2023) showed GPT detectors
flagged **over half** of TOEFL essays by non-native English speakers as AI,
versus near-zero for native-speaker essays — because low lexical variety and
simple syntax read as "low perplexity". Any deployment that punishes people
based on this class of score will disproportionately punish non-native
speakers, and that is a product decision, not a modelling detail.

---

## 10. Therefore: the architecture

No single signal is good enough. The design in this repo is a **layered
ensemble that reports evidence, not a verdict**:

```
Layer 0  forensics    unicode, typography, paste artifacts   high precision / low recall
Layer 1  lexical      tiered marker lexicon + attribution    medium / medium
Layer 2  statistical  burstiness, TTR, repetition, hedging   medium / high
Layer 3  structural   paragraph/section/closing shape        low / high
Layer 4  code         source-code specific tells             medium / medium
Layer 5  probe        Binoculars / Fast-DetectGPT (optional) high / high  [needs GPU]
```

Each layer emits `Signal` objects carrying a **log-odds contribution** and a
human-readable piece of evidence. The detector sums log-odds and passes them
through a sigmoid, so the final number is a calibrated probability and every
point of it can be traced to a quoted span of the input.

Layers 0–4 are pure Python with zero dependencies and run anywhere. Layer 5 is
where real accuracy comes from and is intentionally pluggable.

---

## 11. Addendum: the marker-free failure case

The detector described above **missed** a 658-word machine-written technical
specification, scoring it 42% — "inconclusive". An independent production
detector scored the same text `fractionAi: 1.0`, prediction `AI`. The miss is
worth documenting because the reason generalises.

### Why every original layer was blind

| Layer | Found | Why |
|---|---|---|
| lexical | **nothing** | Not one register marker. No *delve/crucial/robust*, no "it's worth noting", no assistant leakage, no negative parallelism. |
| forensics | almost nothing | ASCII `->` not `→`. Straight apostrophes. No NBSP, no markdown — headers were bare lines. |
| structural | **nothing** | No `#` headings to count, no "In conclusion", no answer shape. |
| statistical | net *human* | Sentence burstiness sat on the neutral midpoint; zero connectives was being scored as human evidence. |

The text was terse, identifier-dense specification prose. That register strips
out every surface feature the original ensemble measured. **The failure mode is
genre, not model quality** — any sufficiently compressed technical register
defeats vocabulary-based detection.

A secondary lesson: a human reading it (including the author of this file)
judged it *more* likely human, reasoning that its high information density and
lack of redundancy were un-model-like. That reasoning was wrong. Low redundancy
indicates a human author in **narrative or argumentative** prose, where a model's
padding instinct shows. It indicates nothing in **reference/spec** prose, where
the model is transcribing a structured requirement set and compression is the
genre norm. Applying a narrative-genre intuition to reference-genre text is what
produced the wrong call.

### What actually separates it (§12, §13)

Two new families, both measuring *form* rather than *vocabulary*, so they
survive a marker-free register.

## 12. Syntactic template repetition — the strongest single feature found

Abstract every token to a shape class (function words kept literal, everything
else → `<w>`, `<ID>`, `<Cap>`, `<CAPS>`), then measure what share of 5-token
templates recur. Word-level n-grams cannot see this: the vocabulary differs
while the construction repeats.

Measured on fixed 200-word windows, because raw repeat-share grows with document
length and would otherwise be a length detector in disguise.

| | windowed 5-gram template repeat-share |
|---|---|
| human samples | 2.6%, 4.1%, 6.4% |
| machine samples | 13.8%, 14.8%, 19.9% |
| the missed spec | **17.9%** |

Over 2x separation, no overlap, and the widest margin of any single feature in
this package. The mechanism is principled: models reuse construction patterns
even when deliberately varying word choice.

Variants tested and rejected — collapsing function words to a single class
(gap 5.9), 4-grams (gap 4.0), 6-grams (gap 7.3). Keeping function words literal
at n=5 gave the cleanest split.

## 13. Authorial presence

A human writing 650 words of anything almost always leaves themselves in it: a
pronoun, a hedge, an aside, an unresolved question, a TODO. Machine output is
authorless by default.

| | 1st/2nd person per 1k | meta-commentary |
|---|---|---|
| human samples | 39–78 | 0–6.6 |
| machine samples | 11–38 | 2.7–5.4 |
| the missed spec | **0.0** | **0.0** |

Measured as an *absence*, which is unusual here and needs care: absence is only
evidence once the text is long enough for it to be surprising. Gated at 250
words and scaled with length.

Related, weaker: **term invariance** (people drift between "work package", "WP"
and "the package"; models lock onto one surface form) and **enumeration
density** (people write "etc." and trail off; models enumerate the closed set).

## 14. Result and the honest caveat

The missed spec went **42% → 96%**, and every other sample improved; the
corpus margin went from 62.6 to 90.8 points.

That 90.8 is **in-sample**. The template-repetition midpoint was fit to seven
documents, one of which is the spec itself. The *direction* and *mechanism* of
both new features are principled and I would expect them to hold; the exact
constants are not trustworthy and should be re-fit with `aidetect.calibrate` on
a real corpus before anyone relies on the magnitudes.


---

## 15. The matched pair

The most informative document in this corpus is a *pair*: the same feature
request for the `goose` migration tool, written once by a model and once by a
person, the human one confirmed by passing an independent production detector.
They share **35 of 36 identifiers**. Same four states, same result struct, same
option functions, same storage scheme. The content is not in dispute, so every
difference is in the writing.

### What separated them, measured

| | AI | human |
|---|---|---|
| paragraph length CV | 0.33 | **0.70** |
| paragraph word counts | 68,117,108,79,69,78,67,34 | 64,**21**,**9**,84,121,42,**160**,80 |
| typographic irregularities | **0** | `it's/its` confusion, trailing whitespace |
| sentence length CV | 0.49 | 0.60 |
| authorial presence per 1k | 6.5 | 12.0 |

The human has a 9-word paragraph next to a 160-word one - a throwaway line,
then the part they actually cared about. And `it's applied status` is a
grammar confusion rather than a typo, which models essentially never produce.
That error type was added to the irregularity list as a result.

### The structural difference, which was new

Distinct identifiers per paragraph:

```
AI     [0,  2, 14,  2,  5,  1,  8,  3]    spread across 7 of 8
human  [0,  0,  0,  0,  0,  0, 22,  9]    quarantined into the last 2
```

The person wrote six paragraphs of plain prose about the problem, then put
every API name into one bolt-on section introduced as *"to make this concrete,
the names I'd go with"*. The model interleaved naming through nearly every
paragraph.

People separate **what I want** from **what to call it**. Models treat naming
as part of each requirement. Measured across every technical document in the
corpus:

```
human technical prose     0.25, 0.29       fraction of paragraphs with identifiers
machine technical prose   0.88, 1.00, 1.00, 1.00, 1.00, 1.00
```

Shipped as `structural.identifier_dispersion`, gated to documents with at
least 15 identifiers per 1k words and four substantial paragraphs, and
weighted modestly because only two human documents support the low end.

### Two candidates from the same analysis that were rejected

**Declarative-vs-request stance.** The model writes specification
(*"`MigrationStatus` gains `AppliedChecksum`"*), the person writes request
(*"I would really like goose to be able to..."*). Striking on the pair - 24.2
vs 10.3 declaratives per 1k - and it does not survive the corpus: one machine
document scores -10.0, below every human. Rejected.

**Backtick/markup consistency.** The AI version wrapped all 136 identifier
occurrences in backticks and used no lists; the human used lists and not one
backtick. Rejected on principle rather than measurement: it is pure
presentation, and `TestFormattingInvariance` exists precisely to keep
presentation out of the scoring. The same document reformatted must not score
differently.

### Effect

Matched-pair separation went from 60 to 71 points (AI 84.2%, human 13.1%), and
the corpus margin from 79.0 to 85.4.

### Why pairs are worth more than documents

Every other comparison in this corpus confounds authorship with topic, genre
and length - a blog post against an SEO article differs in far more than who
wrote it. A matched pair holds all of that constant, so the remaining
difference is the thing being measured. **One matched pair taught this project
more than the preceding six documents.**
