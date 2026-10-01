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
