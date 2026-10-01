# Developing this in a Claude Code cloud session

Notes on the workflow that produced this repo, since the environment has real
constraints worth knowing about.

## What a cloud session is

The session runs in an **ephemeral container** in the cloud, not on your
machine. The repo was cloned fresh when the container started, and the container
is reclaimed after inactivity. Anything not committed and pushed is lost.

That single fact drives everything below.

## Constraints that shaped this design

**1. Commit early, commit often.** There is no "I'll clean it up tomorrow" —
tomorrow the container is gone. Each layer here was committed as it passed its
tests.

**2. Assume no packages.** This container had Python 3.11 and *not even numpy*.
Rather than fight it, the detector is pure-stdlib, which turned out to be the
right call anyway: it runs anywhere, installs in zero seconds, and the heavy
model-backed layer is isolated in `probe/` behind a lazy import that degrades
gracefully when torch is missing.

If you do need packages, outbound HTTPS goes through a proxy that is already
configured — `pip install` works. Just do not assume a GPU.

**3. Network access is governed by a policy** chosen when the environment was
created. If a host is blocked you will see it as a connection failure, not a
clear error. Check the environment's network settings rather than debugging the
code.

**4. Disk is a fixed per-session allowance.** `df` is misleading: "Avail" at 0
with low "Used" means the allowance is spent, not that the machine is broken.
Relevant here because `probe/` would download ~14GB of model weights — do that
on a real machine, not in a session you want to keep using.

## A workflow that works

```bash
git checkout -b feature/thing          # never work on the default branch
# ... build one layer ...
python3 -m unittest discover -s tests  # prove it before committing
git add -A && git commit -m "..."
git push -u origin feature/thing       # push often; the container is temporary
```

Build in **testable slices**. Each layer in this repo is independently
importable and independently tested, so a broken layer never blocks the rest.
That matters more in an ephemeral environment than in a local checkout, because
you cannot leave a half-finished tree lying around overnight.

## What actually found the bugs

Not reading the code — *running it on real input*. Two real bugs surfaced only
by scoring the sample corpus and reading the per-signal breakdown:

1. `'ugh'` was matching inside **"throughput"** and **"thoughtfully"**. The
   human-marker regex had no word boundaries, so a technical document about
   database throughput accumulated phantom "human" evidence. It produced a
   -1.89 log-odds swing and caused a false negative.

2. `\s+$` in MULTILINE mode matches the newline of every **blank line**, so
   paragraph breaks were being counted as human "trailing whitespace"
   irregularities. Every multi-paragraph document got unearned human credit.

Both are invisible in code review and obvious the moment you print the evidence
list. This is the argument for making a scorer explain itself: an opaque
`0.73` hides both bugs forever, while `human_markers -1.89 'ugh'` on a document
that contains no "ugh" is impossible to miss.

Both now have named regression tests in `tests/test_detector.py`.

## Getting the work out

The session can push to the repo directly. Branch, commit, push — then open a PR
from the GitHub UI, or ask the session to open one. Do not rely on anything
living in `/tmp` or in the container's filesystem after you close the tab.
