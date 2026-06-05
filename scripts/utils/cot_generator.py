"""Deterministic, solver-grounded chain-of-thought for real train rows.

No external API: for the families we can solve exactly we articulate the inferred
rule, apply it, and arrive at the (already-known) gold answer. For the hard
families (bit_manipulation, equation) we return ``None`` so the caller emits a
direct-answer example instead of fabricating reasoning — the reasoning for those
families is taught via synthetic puzzles with *known* rules (puzzle_generator).

Every reasoning string is verified to be consistent with the gold answer before
use (the caller checks via the competition metric).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from scripts.utils import solvers


def _pairs(prompt: str) -> List[Tuple[str, str]]:
    return solvers._arrow_pairs(prompt)


def _gravity_reasoning(prompt: str, answer: str) -> Optional[str]:
    obs = re.findall(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)", prompt)
    q = re.search(r"determine the falling distance for t\s*=\s*([\d.]+)", prompt)
    if not obs or not q:
        return None
    xs = [0.5 * float(t) ** 2 for t, _ in obs]
    ys = [float(d) for _, d in obs]
    g = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    tt = float(q.group(1))
    return (
        f"The model is d = 0.5*g*t^2, so g = 2*d/t^2 is constant across the "
        f"observations. Fitting the examples gives g ≈ {g:.2f}. "
        f"For t = {tt:g}: d = 0.5*{g:.2f}*{tt:g}^2 ≈ {answer}."
    )


def _unit_reasoning(prompt: str, answer: str) -> Optional[str]:
    pairs = re.findall(r"([\d.]+)\s*m\s*becomes\s*([\d.]+)", prompt)
    q = re.search(r"convert the following measurement:\s*([\d.]+)", prompt)
    if not pairs or not q:
        return None
    xs = [float(a) for a, _ in pairs]
    ys = [float(b) for _, b in pairs]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    den = n * sum(x * x for x in xs) - sx * sx
    if abs(den) > 1e-12:
        k = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / den
        b = (sy - k * sx) / n
    else:
        k, b = sy / sx, 0.0
    xt = float(q.group(1))
    off = "" if abs(b) < 5e-3 else f" + {b:.2f}"
    return (
        f"Each output is a linear function of the input: output ≈ {k:.4f}*input"
        f"{off}. Applying it to {xt:g}: {k:.4f}*{xt:g}{off} ≈ {answer}."
    )


def _numeral_reasoning(prompt: str, answer: str) -> Optional[str]:
    q = re.search(r"write the number (\d+)", prompt)
    if not q:
        return None
    n = int(q.group(1))
    return (
        f"The examples map each integer to its Roman numeral. "
        f"Converting {n}: {n} = {answer}."
    )


def _encryption_reasoning(prompt: str, answer: str) -> Optional[str]:
    pairs = _pairs(prompt)
    cmap = solvers._enc_letter_map(pairs)
    q = re.search(r"decrypt the following text:\s*(.+)$", prompt, re.M)
    if cmap is None or not q:
        return None
    sample = ", ".join(f"{c}->{p}" for c, p in sorted(cmap.items())[:6])
    return (
        f"Aligning each ciphertext with its plaintext gives a fixed letter "
        f"substitution (e.g. {sample}, ...). Decoding '{q.group(1).strip()}' "
        f"with this mapping yields: {answer}."
    )


_BUILDERS = {
    "gravity": _gravity_reasoning,
    "unit_conversion": _unit_reasoning,
    "numeral": _numeral_reasoning,
    "encryption": _encryption_reasoning,
}


def build_reasoning(prompt: str, family: str, answer: str) -> Optional[str]:
    """Return concise reasoning for a solvable family, else None (→ direct answer)."""
    fn = _BUILDERS.get(family)
    if fn is None:
        return None
    try:
        return fn(str(prompt), str(answer).strip())
    except Exception:
        return None
