# How machine-written text differs from human-written text

Measured findings from 13 documents, 12 with confirmed labels. Every number
here comes from running `aidetect` over the corpus in this repository; nothing
is quoted from literature without being re-measured.

**Read section 6 before using any of this.** It contains a result that
undermines most of what comes before it.

---

## 1. The headline

Machine text is **too even**. Not more formal, not more verbose, not more
fond of the word "delve" — those are symptoms with poor recall. The durable
difference is that a model allocates attention uniformly: paragraphs come out
the same size, sentences cluster around one length, constructions repeat,
and nothing is left unresolved. People are lumpy. They write a nine-word
paragraph then a hundred-and-sixty-word one, because one part mattered to them
and the other did not.

Everything below is a way of measuring unevenness, and the main limitation is
that **unevenness is also a property of formatting and editing**, not only of
authorship.

## 2. The corpus

| | documents |
|---|---|
| human | 5 — a blog post, a forum post, a product review, two technical specifications |
| machine | 7 — two articles, an explainer, four technical specifications |
| unlabelled | 1 |

Two confirmed **matched pairs** — the same request written once by a person
and once by a model — are the most valuable items in it, because they hold
topic, genre, length and format constant.

## 3. Results

```
document               truth    score   verdict
goose spec (prose)     AI       69.4%   leans machine-generated
goose spec (rough)     HUMAN    10.8%   likely human-written
graphify (polished)    AI       82.7%   likely machine-generated
DBC (damaged)          AI       29.3%   leans human-written          WRONG
DBC (clean)            HUMAN    60.4%   leans machine-generated      WRONG
cross-mission spec     AI       95.5%   very likely machine-generated
chatgpt article        AI      100.0%   very likely machine-generated
claude explainer       AI       98.6%   very likely machine-generated
seo article            AI      100.0%   very likely machine-generated
blog (moving)          HUMAN     0.6%   very likely human-written
forum (debugging)      HUMAN     1.0%   very likely human-written
review (keyboard)      HUMAN     3.5%   very likely human-written

accuracy 10/12
```

Note the shape of the errors: both are on one matched pair, and they are
**inverted** — the machine document scores human and the human document scores
machine. Section 6.

## 4. What actually separates the classes

Per-signal reliability, measured as "fired, and pointed the right way":

| signal | accuracy | fired on |
|---|---|---|
| lexical register markers (tiers 1–4, rule-of-three) | **1.00** | 3–4 of 12 |
| structural answer-shape, mandatory conclusion | **1.00** | 2 of 12 |
| `syntax.clause_uniformity` | 0.90 | 10 of 12 |
| `stance.authorial_absence` | 0.83 | 12 of 12 |
| `structural.identifier_dispersion` | 0.83 | 6 of 12 |
| `stylometry.burrows_delta` | 0.82 | 11 of 12 |
| `statistical.paragraph_uniformity` | 0.80 | 10 of 12 |
| `syntax.template_repetition` | 0.75 | 12 of 12 |
| `statistical.burstiness` | 0.73 | 11 of 12 |

### 4.0 Accuracy is not importance

The table above answers "when this signal fires, is it right?" It does not
answer "how much does it move a score?" — and those come apart badly.

Mean absolute contribution across the same 12 documents, in log-odds:

| signal | mean \|lo\| | max | accuracy |
|---|---|---|---|
| `syntax.template_repetition` | **0.743** | 1.425 | 0.75 |
| `statistical.burstiness` | 0.389 | 1.632 | 0.73 |
| `lexical.tier1_register` | 0.360 | 1.738 | 1.00 |
| `stylometry.burrows_delta` | 0.349 | 0.677 | 0.82 |
| `statistical.paragraph_uniformity` | 0.334 | 1.022 | 0.80 |
| `stance.authorial_absence` | 0.310 | 0.703 | 0.83 |
| `syntax.clause_uniformity` | 0.240 | 0.465 | **0.90** |
| `structural.identifier_dispersion` | 0.193 | 0.420 | 0.83 |
| `discourse.sentence_type_mix` | 0.088 | 0.327 | 0.71 |
| `forensics.em_dash` | 0.029 | 0.138 | 0.25 |

`clause_uniformity` is the most *accurate* signal in the package and seventh by
weight. The register lexicon is perfectly accurate and contributes little,
because it rarely fires. `template_repetition` carries roughly twice the weight
of anything else despite ranking eighth on accuracy.

**A reliable signal and an important one are different things, and the earlier
table only measured the first.**

### 4.0b Case study: the one document pair that crossed the line

Two prose versions of the same cross-mission specification, same author
request, scoring 96.5% and 20.7%. Decomposing the 4.66 log-odds swing:

```
syntax.template_repetition        +1.71   construction variety
statistical.paragraph_uniformity  +1.60   paragraph length variance
stance.enumeration_density        +0.30   comma / list density
statistical.burstiness            +0.29   sentence length variance
stance.authorial_absence          +0.23   pronouns, hedges, asides
syntax.clause_uniformity          +0.21   clause length variance
stance.term_invariance            +0.18
stylometry.burrows_delta          +0.15
structural.identifier_dispersion  +0.07
```

The second version reads conspicuously more casual — fourteen first-person
pronouns against two, two rhetorical questions against none, asides like
*"why would we yank it?"*. **Those casual markers account for 0.23 of 4.66.
Five percent.**

Seventy-one percent is two structural measures: how varied the sentence
constructions are, and how unevenly the document is proportioned. The rest is
a tail of further variance measures — comma density, sentence length, clause
length — which are the same underlying property counted four ways.

A controlled test confirms the direction. Merging the second document's
fragmented paragraphs to match the first's structure moved it 20.7% -> 25.4%,
not toward 96.5%. The difference is not formatting and not voice. The sentences
were built differently.

**Consequence.** Casual voice is the visible difference and the cheap one;
`aidetect/bypass.py`'s `authorial_presence` attack injects exactly those
markers and barely moves a score. The expensive differences are structural, and
they are what nine mechanical transforms could not fake (section 6.4).

### 4.1 The register lexicon is precise and nearly useless

*delve, intricate, tapestry, underscores, it's worth noting, in conclusion* —
**100% accurate when they fire, and they fire on a quarter of the corpus.**
They catch default-voice assistant output and nothing else. Every technical
specification in this corpus, machine-written or not, contains none of them.

A detector built only on this, which is what most popular tools are, sees
nothing at all in the documents that matter.

### 4.2 Syntactic template repetition

Abstract every token to a shape class — function words kept literal,
everything else to `<w>`, `<ID>`, `<Cap>` — then measure what share of 5-token
*templates* recur. Word-level n-grams cannot see this: the vocabulary differs
while the construction repeats.

```
human     2.6%  4.1%  6.4%
machine  13.8% 14.8% 17.9% 19.9%
```

Over 2x separation. Measured on fixed 200-word windows, because raw
repeat-share grows with length and would otherwise be a length detector.

### 4.3 Authorial presence

A person writing 650 words almost always leaves themselves in it — a pronoun,
a hedge, an aside, an unresolved question. Machine output is authorless by
default.

```
human informal    64–89 presence markers per 1k words
machine prose     19–47
marker-free spec   0–5
```

**Caveat that cost us a false positive:** human-written *technical* prose also
runs at 0–12. See 6.2.

### 4.4 Paragraph and clause rhythm

The matched-pair measurement, same request, same length:

```
AI     paragraphs  68, 117, 108, 79, 69, 78, 67, 34     CV 0.33
human  paragraphs  64,  21,   9, 84, 121, 42, 160, 80   CV 0.70
```

A 9-word paragraph beside a 160-word one. The model's shortest and longest
differ by a factor of three; the person's by a factor of eighteen.

Sentences show the same thing. In the first fourteen sentences of that pair,
the model never exceeded 18 words; the person wrote a 43, a 37, and a 5.

### 4.5 Error texture

Not error *presence* — error *character*.

- `it's applied status` where `its` was meant. A grammar confusion, not a
  typo. Models essentially never produce it.
- `Rough idea of what I want, open to ideas:-` — a personal punctuation habit.
- An empty numbered list item left in the document.

Against which: one machine document contained six dropped words
(`2 for makes a signal`). That is not a human error pattern either — people do
not drop a word mid-clause and continue coherently. It is transmission damage.

### 4.6 Concept placement

From the matched pair, 35 of 36 identifiers shared:

```
AI     identifiers per paragraph  [0,  2, 14,  2,  5,  1,  8,  3]
human                             [0,  0,  0,  0,  0,  0, 22,  9]
```

The person wrote six paragraphs of prose about the problem, then put every API
name in one section at the end, introduced as *"to make this concrete, the
names I'd go with"*. The model interleaved naming through every paragraph.

**People separate what they want from what to call it.** Models treat naming
as part of each requirement.

### 4.7 Function-word distribution

Burrows's Delta — the canonical authorship method, which reads only
function-word rates and ignores content entirely. Content-free, so it survives
a domain swap: the same spec rewritten with gardening vocabulary (22% content
overlap) still scored +0.79.

## 5. What does NOT separate them

Each of these was measured and rejected. They are listed because they are
widely believed, and because re-proposing them should require new data.

| candidate | result |
|---|---|
| **em dash frequency** | **1 right / 3 wrong — below chance.** The most-cited signal in popular AI detection is anti-correlated here. |
| **lexical diversity (MATTR)** | **3 right / 5 wrong — below chance.** |
| declarative-vs-request stance | striking on one pair, overlaps across the corpus (one machine document below every human) |
| sentence-length range | human 25–50, machine 19–32 — overlaps |
| modal verb density | human 3.5–23, machine 8–22 — overlaps |
| topic drift between paragraphs | human .008–.043, machine .014–.083 — overlaps |
| rare-word reuse | one of three humans inside the machine range |
| readability variance | human 7.4–18.4, machine 9.6–13.6 — total overlap |
| backtick/markup consistency | rejected on principle: pure presentation, trivially changed |

Both below-chance signals have been cut to near-zero weight rather than
inverted, because n is small — but they can no longer move a verdict.

### 6.4 Mechanical transformation does not close the gap

`aidetect/bypass.py` implements the findings above as nine transforms, one per
signal, and searches greedily over them — every attack tried each round, best
move kept, repeats allowed, thirty rounds.

```
chatgpt_remote_work   100.0% -> 99.8%
claude_explainer       98.6% -> 94.3%
seo_article           100.0% -> 97.9%
spec_cross_mission     95.5% -> 55.0%    attack space exhausted at 10 rounds
spec11 (bulleted)      97.6% -> 80.8%
spec12 (prose)         96.5% -> 83.3%
```

Nothing crosses 0.5. The best case moves 40 points and stops, with
`paragraph_variance` applied three times contributing more than half of it.

So the findings describe the gap accurately and do not constitute a procedure
for closing it. The only document in this corpus that crossed the threshold
did so by being written differently, not transformed (section 4.0b).

## 6. The result that undermines the rest

### 6.1 A matched pair the detector gets backwards

Two versions of one DBC-linter specification. One machine-written, one
human-written. **The detector scores them inverted**, and 8 of 14 signals
point the wrong way:

| signal | AI doc | human doc | |
|---|---|---|---|
| `paragraph_uniformity` | −1.20 | −0.05 | backwards |
| `template_repetition` | −0.50 | −0.25 | backwards |
| `typo_absence` | −0.25 | +0.18 | backwards |
| `identifier_dispersion` | −0.41 | 0.00 | backwards |
| `burstiness` | −0.32 | −0.03 | backwards |
| **score** | **29.3%** | **60.4%** | both wrong |

The machine document was formatted as a bulleted rule list and had arrived
with transmission damage: uneven paragraphs, visible irregularities, clustered
identifiers. The human document was written as careful, polished, flowing
prose: even paragraphs, zero typos, dispersed identifiers.

**Every signal measured the formatting and the editing, and none measured the
author.**

### 6.2 Technical register is confounded with machine authorship

Three signals measure *polish and impersonality* rather than authorship:
absence of first person, absence of questions, absence of typos. A technical
specification has all three whoever writes it.

```
                   identifiers/1k   presence/1k   authorial signal
human blog/review        0             66–89        −0.4 to −0.7
human goose spec        62             12           +0.41
human DBC spec          23              0.0         +1.14
```

`aidetect/register.py` now damps those signals to 35% on technical prose. The
false positive fell from 88.1% to 60.4% — still wrong, and pinned by a test so
it stays visible.

### 6.3 Windowed views were right when the document score was wrong

On the human DBC spec, the document score said 88.1% machine while triage
scored 6 of 11 windows human, `fraction_ai` 1 of 4, and the segment scan 1 of
5. The windowed views were correct.

The mechanism: every genre-correlated signal has a length gate and switches
off in a 200-word window, so windowing was avoiding the bias by accident.
**When a view disagrees sharply with the headline number, the disagreement is
information.**

## 7. What this means

**The reliable conclusion.** On unmodified, default-voice model output —
articles, explainers, marketing copy — the difference is large and easy to
measure. Four such documents in this corpus score 98.6% to 100%, and the three
informal human documents score 0.6% to 3.5%. That gap is real.

**The unreliable conclusion.** On edited, formatted, or domain-specific
writing, the measured difference is substantially **formatting and polish**,
not authorship. A carefully written human spec and a roughly formatted machine
spec can invert the detector completely, and did.

**Therefore:** these signals measure *how a document was produced and
finished*, which correlates with authorship in the common case and detaches
from it in exactly the cases where the question is interesting. The honest
framing is not "is this AI?" but "does this document have the texture of
unedited generated output?" — a narrower and more defensible question.

## 8. What would actually settle it

1. **Model-backed scoring.** Everything here is a proxy for what a language
   model measures directly. Binoculars — perplexity normalised by
   cross-perplexity — is stubbed in `aidetect/probe/` and needs a GPU.
2. **Authorship verification instead of AI detection.** Compare a document to
   *that person's* prior writing rather than to a population average. Stronger,
   and it sidesteps the fairness problem entirely.
3. **Provenance, not stylometry.** Editing history, revision timestamps, paste
   events. Stronger than every signal in this document combined, and the only
   kind of evidence a person can contest on the merits.
4. **More matched pairs.** Two taught this project more than the other eleven
   documents put together.
