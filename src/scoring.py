"""Transcription scoring for the legibility diagnostic.

Character-level similarity between a model's transcription and the source text,
used to separate *perception* (can the model read the render?) from
*comprehension* (can it use what it read?).

Metric: `difflib.SequenceMatcher(...).ratio()` (Ratcliff/Obershelp), on a
whitespace- and case-normalized pair. Range [0, 1], 1 = identical.

**autojunk=False is load-bearing.** difflib's default `autojunk=True` treats any
character occurring in more than 1% of a sequence longer than 200 chars as
"junk" and excludes it from matching. On BoolQ passages (typically 300-700 chars)
this collapses the score of a *correct* transcription toward zero -- e.g. a
verbatim transcription of a 460-char passage scored 0.009 under the default and
0.98 with it off. Always pass autojunk=False here.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize(s: str) -> str:
    """Lowercase, strip ends, collapse internal whitespace runs to one space."""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def gold_text(item: dict) -> str:
    """The reference string a render should reproduce: passage then question."""
    return f"{item['passage']} {item['question']}"


def transcription_similarity(pred: str, gold: str) -> float:
    """Normalized character similarity in [0, 1]. See module docstring for why
    autojunk=False matters on long passages."""
    return SequenceMatcher(None, normalize(pred), normalize(gold),
                           autojunk=False).ratio()
