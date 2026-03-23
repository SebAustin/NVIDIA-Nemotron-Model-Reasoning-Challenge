"""
Default scoring aligned with common Kaggle reasoning metrics (exact string OR numeric tolerance).

Replace this module with the competition's public `metric.py` logic when available.
Implement `scores(gold: str, pred: str) -> bool` if the official API differs.
"""

from __future__ import annotations

import math


def scores(gold: str, pred: str) -> bool:
    """Return True if prediction matches gold per competition rules."""
    return answers_match_default(gold, pred)


def answers_match_default(gold: str, pred: str) -> bool:
    g = (gold or "").strip()
    p = (pred or "").strip()
    if g == p:
        return True
    try:
        gv = float(g)
        pv = float(p)
        if not math.isfinite(gv) or not math.isfinite(pv):
            return False
        tol = 1e-6 * max(abs(gv), 1.0)
        return abs(pv - gv) <= tol
    except ValueError:
        return False
