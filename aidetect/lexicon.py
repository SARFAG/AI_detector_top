"""Marker data: the LLM register, structural phrases and per-model fingerprints.

This is DATA, not logic, on purpose. New models appear constantly; adding one
should mean appending to `MODEL_PROFILES`, never touching the detector.

Tiers encode confidence. A Tier-2 word like "leverage" is ordinary business
English and must never convict on its own, so it carries a small weight and only
matters in aggregate. Tier-4 phrases are assistant-frame leakage and are close
to proof that a chat response was pasted.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Tiered single-word / short-phrase register markers  (research §2)
# --------------------------------------------------------------------------

# Near-diagnostic in volume. Frequency in academic abstracts spiked 10-50x
# after 2022.
TIER1_WORDS = [
    "delve", "delves", "delving", "intricate", "intricacies", "tapestry",
    "realm", "realms", "underscore", "underscores", "underscoring", "pivotal",
    "multifaceted", "nuanced", "meticulous", "meticulously", "garner",
    "garnered", "showcasing", "ever-evolving", "ever-changing", "unwavering",
    "indelible", "burgeoning", "myriad", "paramount", "quintessential",
]

# Ordinary business/academic English. Only meaningful in aggregate.
TIER2_WORDS = [
    "leverage", "leveraging", "robust", "foster", "fostering", "holistic",
    "seamless", "seamlessly", "streamline", "streamlined", "empower",
    "empowering", "harness", "harnessing", "crucial", "vital", "enhance",
    "enhancing", "facilitate", "paradigm", "synergy", "cutting-edge",
    "game-changer", "transformative", "comprehensive", "invaluable",
    "profound", "vibrant", "dynamic", "innovative", "significant",
]

# Multi-word structural phrases. Strong because the *construction* is the tell.
TIER3_PHRASES = [
    "it's worth noting", "it is worth noting", "it's important to note",
    "it is important to note", "it's important to remember",
    "plays a crucial role", "plays a vital role", "plays a pivotal role",
    "a testament to", "navigate the complexities", "navigating the complexities",
    "in the realm of", "in today's world", "in today's fast-paced",
    "the landscape of", "a comprehensive overview", "a comprehensive guide",
    "let's dive in", "let's delve", "let's explore", "dive deep into",
    "here's a breakdown", "here's a quick rundown", "at the end of the day",
    "when it comes to", "the key takeaway", "in conclusion", "in summary",
    "to sum up", "ultimately,", "that being said", "first and foremost",
    "it's no secret that", "more than just", "not only that",
    "unlock the potential", "take your", "to new heights", "shed light on",
    "pave the way", "stand the test of time", "a double-edged sword",
    "the ever-evolving landscape", "a wide range of", "a myriad of",
    "it's crucial to understand", "by understanding",
]

# Assistant-frame leakage: someone pasted a chat response verbatim.
TIER4_LEAKAGE = [
    "as an ai language model", "as an ai assistant", "as a large language model",
    "i don't have personal opinions", "i do not have personal opinions",
    "i don't have personal experiences", "i cannot browse the internet",
    "as of my last knowledge update", "as of my knowledge cutoff",
    "my training data", "i hope this helps", "feel free to ask",
    "feel free to let me know", "let me know if you'd like",
    "let me know if you have any", "is there anything else",
    "i'd be happy to help", "i'd be happy to assist",
    "certainly!", "great question!", "of course!", "absolutely!",
    "here's a summary of", "i apologize for the confusion",
    "i cannot provide", "it's essential to consult",
    "consult a qualified professional", "this is not financial advice",
]

TIER_WEIGHTS = {1: 0.42, 2: 0.11, 3: 0.30, 4: 1.25}
# Max log-odds any single tier may contribute, so a long document cannot
# accumulate unbounded evidence just by being long.
TIER_CAPS = {1: 2.2, 2: 1.1, 3: 2.0, 4: 3.2}
# Saturation midpoints: hit count at which a tier reaches half its cap.
TIER_MIDPOINTS = {1: 2.5, 2: 6.0, 3: 3.0, 4: 1.0}


# --------------------------------------------------------------------------
# Negative evidence: markers that point toward a human author  (research §5)
# --------------------------------------------------------------------------

HUMAN_MARKERS = [
    # Hesitation, self-interruption, informal repair
    "idk", "iirc", "afaik", "tbh", "imo", "imho", "fwiw", "ymmv", "lol", "lmao",
    "meh", "ugh", "huh", "yeah", "yep", "nope", "gonna", "wanna", "kinda",
    "sorta", "dunno", "ain't", "y'all", "bruh", "wtf", "damn", "shit",
    # Personal, situated, unverifiable-by-a-model detail
    "my boss", "my wife", "my husband", "my kid", "my mom", "my dad",
    "last night", "this morning", "yesterday i", "i remember when",
    "back in college", "at my old job", "my coworker",
    # Explicit uncertainty that an assistant would smooth away
    "i could be wrong", "no idea why", "not sure if", "someone correct me",
    "i might be misremembering", "don't quote me",
    # Anti-structure
    "anyway", "anyways", "long story short", "sorry for the rant",
    "edit:", "update:", "tl;dr",
]

# Markers whose human reading depends on context. A plain word-boundary match
# is wrong for these: "Whatever remains produces a warning" is a relative
# pronoun in formal prose, not the dismissive interjection, and it was costing
# a machine-written specification 0.84 log-odds of phantom human evidence.
HUMAN_MARKER_PATTERNS = [
    # Dismissive "whatever" only: standalone, or closing a clause.
    r"\bwhatever\s*(?=[.!?,;]|$)",
    # "right" / "sure" as sentence-initial concessives, not adjectives.
    r"(?:^|[.!?]\s+)(?:right|sure|fine)\s*[,.]",
    # Trailing "or something" / "or whatever".
    r"\bor (?:something|whatever)\b",
]

# --------------------------------------------------------------------------
# Per-model fingerprints  (research §3)
# Weak evidence only. These drift with every model release — recalibrate
# against fresh samples rather than trusting them indefinitely.
# --------------------------------------------------------------------------

MODEL_PROFILES = {
    "ChatGPT": {
        "phrases": [
            "certainly!", "here's a breakdown", "here's a quick rundown",
            "let me know if you'd like me to", "i hope this helps!",
            "let's dive in", "great question!", "sure thing!",
            "here's how you can", "in a nutshell",
        ],
        "patterns": [
            # "- **Speed:** ..." — bolded lead-in on every bullet
            (r"^\s*[-*]\s+\*\*[^*\n]{2,40}:?\*\*", "bolded bullet lead-ins", 3),
            # Emoji used as a section marker in formal text
            (r"^\s*#{1,4}\s*[\U0001F300-\U0001FAFF]", "emoji section headers", 1),
            (r"\|\s*Aspect\s*\|", "Aspect/Description comparison table", 1),
        ],
        "unicode": [" "],  # narrow NBSP: ChatGPT web-UI paste artifact
    },
    "Claude": {
        "phrases": [
            "i should note", "that said,", "worth noting:", "here's the thing",
            "i'd push back", "to be fair,", "i'm not certain, but",
            "the honest answer is", "a few things stand out",
            "let me reframe", "the short version:", "one caveat:",
        ],
        "patterns": [
            # NB: plain '##' headers are universal across assistants and are
            # deliberately NOT a fingerprint here.
            (r"\b(?:That said|To be fair|Worth noting),\s", "hedged pivots", 2),
        ],
        "unicode": [],
    },
    "Claude Code": {
        "phrases": [
            "generated with [claude code]", "co-authored-by: claude",
            "claude.ai/code", "claude.com/claude-code",
        ],
        "patterns": [
            (r"\U0001F916\s*Generated with", "Claude Code commit trailer", 1),
            (r"Co-Authored-By:\s*Claude", "Claude co-author trailer", 1),
            (r"^\s*<thinking>", "leaked thinking block", 1),
        ],
        "unicode": [],
    },
    "Gemini": {
        "phrases": [
            "of course.", "it's important to remember that",
            "here's a comprehensive overview", "i can certainly help with that",
            "absolutely! here",
        ],
        "patterns": [
            (r"^\s{2,}[-*]\s+\*\*", "deeply nested bolded bullets", 3),
        ],
        "unicode": [],
    },
}


# --------------------------------------------------------------------------
# Prompt-authorship markers  (research §7)
# --------------------------------------------------------------------------

PROMPT_MARKERS = [
    "you are a", "you are an", "act as a", "act as an", "your task is to",
    "your goal is to", "your job is to", "you will be given",
    "respond only with", "respond in json", "output format:", "output only",
    "do not include", "do not explain", "step by step", "think step by step",
    "take a deep breath", "this is very important to my career",
    "you must", "you should never", "under no circumstances",
    "i will tip you", "world-class", "expert in the field",
    "<instructions>", "</instructions>", "###instruction", "### instruction",
]
