"""Answer extraction + scoring, mirroring the competition metric.

Extraction order (matches the competition's described behaviour: prefer boxed,
then heuristic patterns, then the last numeric value):
    1. the LAST ``\\boxed{...}`` in the text
    2. heuristic phrases ("the answer is X", "answer: X", "result is X")
    3. the last numeric value found
Scoring is exact-string OR relative tolerance 1e-2 (see competition_metric).
"""

from __future__ import annotations

import re
from typing import Optional

from scripts.utils.competition_metric import scores as default_scores

_HEURISTIC = re.compile(
    r"(?:final answer|the answer is|answer\s*[:=]|result is|result\s*[:=])\s*"
    r"\$?\\?\(?\s*([^\n.$]+?)\s*\$?\)?\s*[.\n]?$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed_answer(text: str) -> Optional[str]:
    """Contents of the last ``\\boxed{...}``.

    Greedy "up to the last closing brace": the answer itself may contain ``{`` or
    ``}`` (e.g. symbol-alphabet equation answers), so we take everything after the
    final ``\\boxed{`` up to the last ``}`` in the text. Falls back to the rest of
    the string if the brace is never closed.
    """
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    rest = text[idx + len("\\boxed{"):]
    close = rest.rfind("}")
    return (rest[:close] if close != -1 else rest).strip()


def extract_final_answer(text: str) -> Optional[str]:
    """Full extraction chain used to score model output."""
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    heur = _HEURISTIC.findall(text)
    if heur:
        cand = heur[-1].strip().strip("`'\"")
        if cand:
            return cand
    nums = _NUMBER.findall(text)
    if nums:
        return nums[-1]
    return None


def answers_match(gold: str, pred: str) -> bool:
    return default_scores(gold, pred)
