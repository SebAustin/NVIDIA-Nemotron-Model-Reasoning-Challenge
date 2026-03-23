"""Extract \\boxed{} answers and compare to gold using competition metric hook."""

from __future__ import annotations

import re
from typing import Optional

from scripts.utils.competition_metric import scores as default_scores

_BOXED_PATTERN = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.DOTALL,
)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the contents of the last \\boxed{...} in text (supports one nested brace level)."""
    if not text:
        return None
    matches = _BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def answers_match(gold: str, pred: str) -> bool:
    """Use competition `scores(gold, pred)` if provided elsewhere; else default implementation."""
    return default_scores(gold, pred)
