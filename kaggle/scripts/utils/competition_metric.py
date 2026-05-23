"""Scoring for the NVIDIA Nemotron Model Reasoning Challenge.

Mirrors the public reference implementation `compare_answer` confirmed in the
2026 Progress-Prize winning submission (tonghuikang/nemotron). The rules are:

1. Pure binary strings (`[01]+` after strip) compare **strictly as strings**
   (case-insensitive). Leading-zero-pad sensitivity means `"00011011" != "11011"`.
2. Otherwise, attempt `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` on float casts.
3. Otherwise, fall back to **case-insensitive** stripped string comparison.

The previous placeholder used a relative tolerance `1e-2 * max(|gold|, 1.0)`
that was lenient for |gold|>1, and did **not** handle case-insensitive strings
or binary strings specifically. Local accuracy was therefore biased upward
versus the leaderboard for several categories (notably `numeral`, `cipher`,
and any binary-string-output `bit_manipulation` rows where leading zeros
matter).
"""

from __future__ import annotations

import math
import re

_BINARY_RE = re.compile(r"[01]+")


def scores(gold: str, pred: str) -> bool:
    """Return True if prediction matches gold per competition rules."""
    return answers_match_default(gold, pred)


def answers_match_default(gold: str, pred: str) -> bool:
    g = (gold or "").strip()
    p = (pred or "").strip()

    if _BINARY_RE.fullmatch(g):
        return p.lower() == g.lower()

    try:
        gv = float(g)
        pv = float(p)
        if not math.isfinite(gv) or not math.isfinite(pv):
            return False
        return math.isclose(gv, pv, rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass

    return p.lower() == g.lower()
