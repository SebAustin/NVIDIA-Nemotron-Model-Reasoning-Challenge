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


def answers_match_default(gold: str, pred: str, rel_tol: float = 1e-2) -> bool:
    """Competition rule: correct if exact string match OR within relative tolerance 1e-2.

    Numeric comparison is symmetric relative tolerance (``math.isclose`` semantics)
    with a small absolute floor so values near zero still compare sanely.
    """
    g = (gold or "").strip()
    p = (pred or "").strip()
    if g == p:
        return True
    try:
        gv = float(g)
        pv = float(p)
    except ValueError:
        return False
    if not (math.isfinite(gv) and math.isfinite(pv)):
        return False
    return math.isclose(gv, pv, rel_tol=rel_tol, abs_tol=rel_tol)
