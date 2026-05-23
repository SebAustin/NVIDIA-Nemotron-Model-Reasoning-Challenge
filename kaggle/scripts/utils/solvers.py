"""Programmatic solvers for the four puzzle families.

Each solver enumerates a small rule family, finds the rule matching ALL example pairs,
and returns the answer for the test input plus a discovery-style chain-of-thought.
"""

from __future__ import annotations

import re
import string
from typing import Callable, List, Optional, Tuple

PuzzleType = str  # "bit_manipulation" | "text_cipher" | "algebraic" | "sequence"


# ---------------------------------------------------------------------------
# Prompt parsing helpers
# ---------------------------------------------------------------------------

_ARROW_RE = re.compile(r"\s*(?:→|->|=>)\s*")
_INPUT_PREFIX = re.compile(r"^(?:input)\s*[:=]?\s*", re.IGNORECASE)
_OUTPUT_PREFIX = re.compile(r"^(?:output)\s*[:=]?\s*", re.IGNORECASE)
_TEST_PREFIX = re.compile(
    r"^(?:this\s+input|the\s+test\s+input|test\s+input|input)\s*[:=]?\s*",
    re.IGNORECASE,
)


def _strip_input_prefix(s: str) -> str:
    return _INPUT_PREFIX.sub("", s).strip()


def _strip_output_prefix(s: str) -> str:
    return _OUTPUT_PREFIX.sub("", s).strip()


def _strip_test_prefix(s: str) -> str:
    return _TEST_PREFIX.sub("", s).strip()


def _parse_io_pairs(prompt: str) -> List[Tuple[str, str]]:
    """Extract input/output example pairs from a prompt."""
    pairs: List[Tuple[str, str]] = []
    for raw_line in prompt.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith((
            "hint", "rule", "given", "compute", "apply", "what",
            "now ", "respond", "examples", "structure", "numeric puzzle",
            "sequence puzzle", "in ", "here are",
        )):
            continue
        parts = _ARROW_RE.split(line)
        if len(parts) != 2:
            continue
        inp = _strip_input_prefix(parts[0]).strip()
        out = _strip_output_prefix(parts[1]).strip()
        if not inp or not out:
            continue
        if inp.lower().startswith(("example", "note", "hint", "rule")):
            continue
        if "(" in inp and ")" in inp and any(ch.isalpha() for ch in inp):
            continue
        pairs.append((inp, out))
    return pairs


def _find_test_input(prompt: str, kind: PuzzleType) -> Optional[str]:
    """Locate the test input in the prompt (the value to apply the rule to)."""
    patterns = [
        r"apply (?:the same |the )?(?:rule|mapping|transformation|function)?\s*(?:to[:\s]?)\s*(.+)",
        r"now apply (?:the same |the )?rule (?:to[:\s]?)\s*(.+)",
        r"compute\s+f\(\s*(-?\d+)\s*\)",
        r"compute\s+(.+)",
    ]
    text = prompt.strip()
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.lastindex:
            cand = m.group(m.lastindex).strip().rstrip(".").rstrip("?")
            cand = cand.split("\n")[0].strip()
            cand = _strip_test_prefix(cand)
            if cand:
                return cand
    return None


# ---------------------------------------------------------------------------
# Bit manipulation solver
# ---------------------------------------------------------------------------

def _bits_xor(a: str, b: str) -> str:
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def _bits_rotl(b: str, k: int) -> str:
    k %= len(b)
    return b[k:] + b[:k]


def _bits_complement(b: str) -> str:
    return "".join("0" if c == "1" else "1" for c in b)


def _bits_swap_nibbles(b: str) -> str:
    if len(b) != 8:
        return b
    return b[4:] + b[:4]


def _bits_flip(b: str, idx: int) -> str:
    arr = list(b)
    arr[idx] = "0" if arr[idx] == "1" else "1"
    return "".join(arr)


def _is_bit_string(s: str) -> bool:
    return bool(s) and all(c in "01" for c in s)


def _bit_pairs(prompt: str) -> List[Tuple[str, str]]:
    pairs = _parse_io_pairs(prompt)
    return [(a, b) for a, b in pairs if _is_bit_string(a) and _is_bit_string(b) and len(a) == len(b)]


def _bits_shl(b: str, k: int) -> str:
    k %= len(b)
    return b[k:] + ("0" * k)


def _bits_shr(b: str, k: int) -> str:
    k %= len(b)
    return ("0" * k) + b[: len(b) - k]


def _solve_bit_inner(
    pairs: List[Tuple[str, str]],
    test_inp: str,
) -> Optional[Tuple[str, str, str]]:
    """Returns (rule_desc, applied_answer, applied_steps) or None."""
    if not pairs or not _is_bit_string(test_inp):
        return None
    n = len(pairs[0][0])
    if any(len(a) != n or len(b) != n for a, b in pairs):
        return None
    if len(test_inp) != n:
        return None

    # XOR with fixed mask
    mask_candidates = set()
    for a, b in pairs:
        mask_candidates.add(_bits_xor(a, b))
    if len(mask_candidates) == 1:
        mask = mask_candidates.pop()
        if all(_bits_xor(a, mask) == b for a, b in pairs):
            return f"XOR with fixed mask {mask}", _bits_xor(test_inp, mask), f"XOR each bit of {test_inp} with {mask}"

    # Rotate left by k
    for k in range(1, n):
        if all(_bits_rotl(a, k) == b for a, b in pairs):
            return f"Rotate left by {k} bits", _bits_rotl(test_inp, k), f"Shift each bit of {test_inp} left by {k} positions, wrapping around"

    # Reverse all bits
    if all(a[::-1] == b for a, b in pairs):
        return "Reverse all bits", test_inp[::-1], f"Reverse the order of bits in {test_inp}"

    # Swap nibbles (8-bit only)
    if n == 8 and all(_bits_swap_nibbles(a) == b for a, b in pairs):
        return "Swap upper and lower 4 bits", _bits_swap_nibbles(test_inp), f"Swap nibbles of {test_inp}"

    # Swap halves (16-bit: upper 8 <-> lower 8)
    if n == 16 and all(a[8:] + a[:8] == b for a, b in pairs):
        return "Swap upper and lower 8-bit halves", test_inp[8:] + test_inp[:8], f"Swap the two halves of {test_inp}"

    # Flip bit at fixed position
    for idx in range(n):
        if all(_bits_flip(a, idx) == b for a, b in pairs):
            return f"Flip bit at position {idx} (0=leftmost)", _bits_flip(test_inp, idx), f"Flip bit {idx} of {test_inp}"

    # Add m mod 2^n
    if n <= 16:
        diffs = set()
        modulus = 1 << n
        for a, b in pairs:
            d = (int(b, 2) - int(a, 2)) % modulus
            diffs.add(d)
        if len(diffs) == 1:
            m = diffs.pop()
            if all((int(a, 2) + m) % modulus == int(b, 2) for a, b in pairs):
                ans_int = (int(test_inp, 2) + m) % modulus
                return (
                    f"Add {m} mod {modulus} (interpret as unsigned int)",
                    format(ans_int, f"0{n}b"),
                    f"Convert {test_inp} to int ({int(test_inp, 2)}), add {m}, mod {modulus}, convert back",
                )

    # Complement
    if all(_bits_complement(a) == b for a, b in pairs):
        return "Bitwise complement", _bits_complement(test_inp), f"Flip every bit in {test_inp}"

    # Reverse then complement
    if all(_bits_complement(a[::-1]) == b for a, b in pairs):
        ans = _bits_complement(test_inp[::-1])
        return (
            "Reverse all bits then take bitwise complement",
            ans,
            f"Reverse {test_inp} -> {test_inp[::-1]}, then complement -> {ans}",
        )

    # Logical left shift by k (zero-fill on right)
    for k in range(1, n):
        if all(_bits_shl(a, k) == b for a, b in pairs):
            return f"Logical left shift by {k} bits", _bits_shl(test_inp, k), f"Shift {test_inp} left by {k}, fill right with zeros"

    # Logical right shift by k (zero-fill on left)
    for k in range(1, n):
        if all(_bits_shr(a, k) == b for a, b in pairs):
            return f"Logical right shift by {k} bits", _bits_shr(test_inp, k), f"Shift {test_inp} right by {k}, fill left with zeros"

    # Rotate left k then XOR mask
    for k in range(1, n):
        masks_after_rot = set()
        for a, b in pairs:
            rotated = _bits_rotl(a, k)
            masks_after_rot.add(_bits_xor(rotated, b))
        if len(masks_after_rot) == 1:
            mask = masks_after_rot.pop()
            if all(_bits_xor(_bits_rotl(a, k), mask) == b for a, b in pairs):
                ans = _bits_xor(_bits_rotl(test_inp, k), mask)
                return (
                    f"Rotate left by {k} then XOR with {mask}",
                    ans,
                    f"Rotate {test_inp} left by {k}, then XOR with {mask}",
                )

    # Add m mod 2^n then XOR mask
    if n <= 16:
        modulus = 1 << n
        for a, b in pairs[:1]:
            ai, bi = int(a, 2), int(b, 2)
            for m in range(modulus):
                shifted = (ai + m) % modulus
                shifted_bits = format(shifted, f"0{n}b")
                mask = _bits_xor(shifted_bits, b)
                # Verify on all pairs
                if all(
                    _bits_xor(format((int(x, 2) + m) % modulus, f"0{n}b"), mask) == y
                    for x, y in pairs
                ):
                    ans_int = (int(test_inp, 2) + m) % modulus
                    ans = _bits_xor(format(ans_int, f"0{n}b"), mask)
                    return (
                        f"Add {m} mod {modulus} then XOR with {mask}",
                        ans,
                        f"Add {m}, mod {modulus}, then XOR with {mask}",
                    )

    return None


def solve_bit(prompt: str) -> Optional[Tuple[str, str]]:
    pairs = _bit_pairs(prompt)
    if len(pairs) < 2:
        return None
    test_inp = _find_test_input(prompt, "bit_manipulation")
    if test_inp is None or not _is_bit_string(test_inp):
        # Try last "solo" bit string in the prompt
        candidates = re.findall(r"\b([01]{4,16})\b", prompt)
        candidates = [c for c in candidates if not any(c == p[0] or c == p[1] for p in pairs)]
        if not candidates:
            return None
        test_inp = candidates[-1]

    res = _solve_bit_inner(pairs, test_inp)
    if res is None:
        return None
    rule_desc, answer, steps = res

    lines = [
        "I will analyze the input/output pairs to find a consistent bit transformation.",
        "",
        "The example pairs are:",
    ]
    for a, b in pairs:
        lines.append(f"  {a} -> {b}")
    lines.append("")
    lines.append(f"Testing rule: {rule_desc}.")
    lines.append("Verifying against every example:")
    for a, b in pairs:
        lines.append(f"  {a} -> {b} matches.")
    lines.append("")
    lines.append(f"Applying the rule to the test input {test_inp}:")
    lines.append(f"  {steps}")
    lines.append(f"Result: {answer}")
    return answer, "\n".join(lines)


# ---------------------------------------------------------------------------
# Cipher solver
# ---------------------------------------------------------------------------

def _caesar(text: str, k: int) -> str:
    out = []
    for c in text:
        if c.isupper():
            out.append(chr((ord(c) - ord("A") + k) % 26 + ord("A")))
        elif c.islower():
            out.append(chr((ord(c) - ord("a") + k) % 26 + ord("a")))
        else:
            out.append(c)
    return "".join(out)


def _vigenere(text: str, key: str) -> str:
    key = key.upper()
    ki = 0
    out = []
    for c in text:
        if not c.isalpha():
            out.append(c)
            continue
        shift = ord(key[ki % len(key)]) - ord("A")
        ki += 1
        if c.isupper():
            out.append(chr((ord(c) - ord("A") + shift) % 26 + ord("A")))
        else:
            out.append(chr((ord(c) - ord("a") + shift) % 26 + ord("a")))
    return "".join(out)


def _build_substitution(pairs: List[Tuple[str, str]]) -> Optional[dict]:
    sub: dict = {}
    for a, b in pairs:
        if len(a) != len(b):
            return None
        for x, y in zip(a, b):
            if not x.isalpha() or not y.isalpha():
                if x != y:
                    return None
                continue
            xu, yu = x.upper(), y.upper()
            if xu in sub and sub[xu] != yu:
                return None
            sub[xu] = yu
    return sub


def _apply_substitution(text: str, sub: dict) -> str:
    out = []
    for c in text:
        if c.isupper() and c in sub:
            out.append(sub[c])
        elif c.islower() and c.upper() in sub:
            out.append(sub[c.upper()].lower())
        else:
            out.append(c)
    return "".join(out)


def _word_pairs(prompt: str) -> List[Tuple[str, str]]:
    pairs = _parse_io_pairs(prompt)
    return [(a, b) for a, b in pairs if any(c.isalpha() for c in a) and len(a) == len(b)]


def _atbash(s: str) -> str:
    out = []
    for c in s:
        if "A" <= c <= "Z":
            out.append(chr(ord("Z") - (ord(c) - ord("A"))))
        elif "a" <= c <= "z":
            out.append(chr(ord("z") - (ord(c) - ord("a"))))
        else:
            out.append(c)
    return "".join(out)


def _affine(s: str, a: int, b: int) -> str:
    out = []
    for c in s:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr(((a * (ord(c) - base) + b) % 26) + base))
        else:
            out.append(c)
    return "".join(out)


def solve_cipher(prompt: str) -> Optional[Tuple[str, str]]:
    pairs = _word_pairs(prompt)
    if len(pairs) < 2:
        return None
    test_word = _find_test_input(prompt, "text_cipher")
    if not test_word:
        return None
    test_word = test_word.split()[0].strip(string.punctuation)
    if not any(c.isalpha() for c in test_word):
        return None

    rule_desc: Optional[str] = None
    answer: Optional[str] = None
    explain: Optional[str] = None

    # Caesar shift forward
    for k in range(1, 26):
        if all(_caesar(a, k) == b for a, b in pairs):
            rule_desc = f"Caesar shift forward by {k}"
            answer = _caesar(test_word, k)
            explain = f"Shift each letter of {test_word} forward by {k} positions in the alphabet"
            break

    # Reverse string
    if rule_desc is None and all(a[::-1] == b for a, b in pairs):
        rule_desc = "Reverse the string"
        answer = test_word[::-1]
        explain = f"Reverse the letters of {test_word}"

    # Atbash
    if rule_desc is None and all(_atbash(a) == b for a, b in pairs):
        rule_desc = "Atbash cipher (A<->Z, B<->Y, ...)"
        answer = _atbash(test_word)
        explain = f"Apply Atbash mapping to each letter of {test_word}"

    # Caesar then reverse (composition)
    if rule_desc is None:
        for k in range(1, 26):
            if all(_caesar(a, k)[::-1] == b for a, b in pairs):
                rule_desc = f"Caesar shift +{k} then reverse the string"
                answer = _caesar(test_word, k)[::-1]
                explain = f"Shift {test_word} by +{k} then reverse to get {answer}"
                break

    # Reverse then Caesar
    if rule_desc is None:
        for k in range(1, 26):
            if all(_caesar(a[::-1], k) == b for a, b in pairs):
                rule_desc = f"Reverse the string then Caesar shift +{k}"
                answer = _caesar(test_word[::-1], k)
                explain = f"Reverse {test_word} then shift by +{k} to get {answer}"
                break

    # Affine cipher (a coprime with 26)
    if rule_desc is None:
        for a in [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]:
            for b in range(0, 26):
                if all(_affine(p, a, b) == q for p, q in pairs):
                    rule_desc = f"Affine cipher y = {a}*x + {b} mod 26"
                    answer = _affine(test_word, a, b)
                    explain = f"Apply affine y = {a}*x + {b} (mod 26) to each letter of {test_word}"
                    break
            if rule_desc is not None:
                break

    # Vigenere with common keywords
    if rule_desc is None:
        for key in ["KEY", "CODE", "NEMO", "BYTE", "CUDA", "DEEP", "PASS", "GOLD",
                    "LOGIC", "RULE", "DATA", "TEXT", "TURN"]:
            if all(_vigenere(a, key) == b for a, b in pairs):
                rule_desc = f"Vigenere with keyword {key}"
                answer = _vigenere(test_word, key)
                explain = f"Apply Vigenere cipher to {test_word} using keyword {key}"
                break

    # Simple substitution
    if rule_desc is None:
        sub = _build_substitution(pairs)
        if sub is not None and all(c.upper() in sub or not c.isalpha() for c in test_word):
            rule_desc = "Simple substitution (fixed letter permutation)"
            answer = _apply_substitution(test_word, sub)
            mappings_used = sorted({c.upper() for c in test_word if c.isalpha() and c.upper() in sub})
            mapping_str = ", ".join(f"{m}->{sub[m]}" for m in mappings_used)
            explain = f"Using mapping {mapping_str} on each letter of {test_word}"

    if rule_desc is None or answer is None or explain is None:
        return None

    lines = [
        "I will examine the letter mappings to identify the cipher rule.",
        "",
        "Example pairs:",
    ]
    for a, b in pairs:
        lines.append(f"  {a} -> {b}")
    lines.append("")
    p0_in, p0_out = pairs[0]
    if len(p0_in) == len(p0_out):
        lines.append(f"For the first pair {p0_in} -> {p0_out}, letter shifts:")
        for x, y in zip(p0_in.upper(), p0_out.upper()):
            if x.isalpha() and y.isalpha():
                shift = (ord(y) - ord(x)) % 26
                lines.append(f"  {x} -> {y} (shift +{shift})")
    lines.append("")
    lines.append(f"The consistent rule is: {rule_desc}.")
    lines.append("Verifying against all examples:")
    for a, b in pairs:
        lines.append(f"  {a} -> {b} matches.")
    lines.append("")
    lines.append(f"Applying to the test input {test_word}:")
    lines.append(f"  {explain}")
    lines.append(f"Result: {answer}")
    return answer, "\n".join(lines)


# ---------------------------------------------------------------------------
# Algebraic solver
# ---------------------------------------------------------------------------

def _algebra_pairs(prompt: str) -> List[Tuple[int, int]]:
    pairs = _parse_io_pairs(prompt)
    out: List[Tuple[int, int]] = []
    for a, b in pairs:
        try:
            xi = int(a)
            yi = int(b)
            out.append((xi, yi))
        except ValueError:
            continue
    return out


def _digit_sum(x: int) -> int:
    return sum(int(d) for d in str(abs(x)))


def _try_linear(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for mul in range(-9, 10):
        for add in range(-50, 51):
            if all(mul * x + add == y for x, y in points):
                return mul, add
    return None


def _try_quadratic_no_lin(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for a in range(-5, 6):
        for c in range(-50, 51):
            if all(a * x * x + c == y for x, y in points):
                return a, c
    return None


def _try_quadratic(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int]]:
    for a in range(-5, 6):
        for b in range(-20, 21):
            for c in range(-50, 51):
                if all(a * x * x + b * x + c == y for x, y in points):
                    return a, b, c
    return None


def _try_modular(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for n in range(2, 20):
        for k in range(0, n):
            if all((x % n) + k == y for x, y in points):
                return n, k
    return None


def _try_modular_linear(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int]]:
    for n in range(2, 25):
        for a in range(1, 10):
            for b in range(0, 15):
                if all(((a * x) + b) % n == y for x, y in points):
                    return a, b, n
    return None


def _try_abs_mul(points: List[Tuple[int, int]]) -> Optional[int]:
    for mul in range(2, 10):
        if all(abs(x) * mul == y for x, y in points):
            return mul
    return None


def _try_cubic(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """f(x) = a*x^3 + b*x"""
    for a in range(1, 5):
        for b in range(-10, 11):
            if all(a * (x ** 3) + b * x == y for x, y in points):
                return a, b
    return None


def _try_piecewise(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """f(x) = pos_m*x if x>=0 else x^2 + neg_add"""
    pos_pts = [(x, y) for x, y in points if x >= 0]
    neg_pts = [(x, y) for x, y in points if x < 0]
    if not pos_pts or not neg_pts:
        return None
    pos_m: Optional[int] = None
    for m in range(1, 10):
        if all(m * x == y for x, y in pos_pts):
            pos_m = m
            break
    if pos_m is None:
        return None
    for add in range(0, 20):
        if all(x * x + add == y for x, y in neg_pts):
            return pos_m, add
    return None


def _digit_product(x: int) -> int:
    if x == 0:
        return 0
    p = 1
    for d in str(abs(x)):
        p *= int(d)
    return p


def _find_function_test_input(prompt: str) -> Optional[int]:
    m = re.search(r"f\(\s*(-?\d+)\s*\)", prompt)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m = re.search(r"compute\s+(-?\d+)", prompt, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def solve_algebraic(prompt: str) -> Optional[Tuple[str, str]]:
    points = _algebra_pairs(prompt)
    if len(points) < 3:
        return None
    test_x = _find_function_test_input(prompt)
    if test_x is None:
        return None
    if any(x == test_x for x, _ in points):
        return None

    rule_desc: Optional[str] = None
    answer: Optional[int] = None
    steps: Optional[str] = None

    lin = _try_linear(points)
    if lin is not None:
        m, c = lin
        rule_desc = f"f(x) = {m}*x + {c}"
        answer = m * test_x + c
        steps = f"f({test_x}) = {m} * {test_x} + {c} = {answer}"

    if rule_desc is None:
        mod = _try_modular(points)
        if mod is not None:
            n, k = mod
            rule_desc = f"f(x) = (x mod {n}) + {k}"
            answer = (test_x % n) + k
            steps = f"f({test_x}) = ({test_x} mod {n}) + {k} = {test_x % n} + {k} = {answer}"

    if rule_desc is None and all(_digit_sum(x) == y for x, y in points):
        rule_desc = "f(x) = sum of decimal digits of |x|"
        answer = _digit_sum(test_x)
        steps = f"digits of |{test_x}| = {list(str(abs(test_x)))}, sum = {answer}"

    if rule_desc is None and all(_digit_product(x) == y for x, y in points):
        rule_desc = "f(x) = product of decimal digits of |x|"
        answer = _digit_product(test_x)
        steps = f"digits of |{test_x}| = {list(str(abs(test_x)))}, product = {answer}"

    if rule_desc is None:
        am = _try_abs_mul(points)
        if am is not None:
            rule_desc = f"f(x) = |x| * {am}"
            answer = abs(test_x) * am
            steps = f"f({test_x}) = |{test_x}| * {am} = {abs(test_x)} * {am} = {answer}"

    if rule_desc is None:
        qnl = _try_quadratic_no_lin(points)
        if qnl is not None:
            a, c = qnl
            rule_desc = f"f(x) = {a}*x^2 + {c}"
            answer = a * test_x * test_x + c
            steps = f"f({test_x}) = {a} * {test_x}^2 + {c} = {a * test_x * test_x} + {c} = {answer}"

    if rule_desc is None:
        cub = _try_cubic(points)
        if cub is not None:
            a, b = cub
            rule_desc = f"f(x) = {a}*x^3 + {b}*x"
            answer = a * (test_x ** 3) + b * test_x
            steps = f"f({test_x}) = {a} * {test_x}^3 + {b} * {test_x} = {a * (test_x ** 3)} + {b * test_x} = {answer}"

    if rule_desc is None:
        ml = _try_modular_linear(points)
        if ml is not None:
            a, b, n = ml
            rule_desc = f"f(x) = ({a}*x + {b}) mod {n}"
            answer = ((a * test_x) + b) % n
            steps = f"f({test_x}) = ({a} * {test_x} + {b}) mod {n} = {a * test_x + b} mod {n} = {answer}"

    if rule_desc is None:
        pw = _try_piecewise(points)
        if pw is not None:
            pos_m, neg_add = pw
            rule_desc = f"f(x) = {pos_m}*x if x>=0 else x^2 + {neg_add}"
            if test_x >= 0:
                answer = pos_m * test_x
                steps = f"Test x = {test_x} >= 0, so f({test_x}) = {pos_m} * {test_x} = {answer}"
            else:
                answer = test_x * test_x + neg_add
                steps = f"Test x = {test_x} < 0, so f({test_x}) = {test_x}^2 + {neg_add} = {test_x * test_x} + {neg_add} = {answer}"

    if rule_desc is None:
        quad = _try_quadratic(points)
        if quad is not None:
            a, b, c = quad
            rule_desc = f"f(x) = {a}*x^2 + {b}*x + {c}"
            answer = a * test_x * test_x + b * test_x + c
            steps = (
                f"f({test_x}) = {a} * {test_x}^2 + {b} * {test_x} + {c} "
                f"= {a * test_x * test_x} + {b * test_x} + {c} = {answer}"
            )

    if rule_desc is None or answer is None or steps is None:
        return None

    lines = [
        "I need to identify the function f from the input/output pairs.",
        "",
        "Given pairs:",
    ]
    for x, y in points:
        lines.append(f"  f({x}) = {y}")
    lines.append("")
    if len(points) >= 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        if x1 != x0:
            slope = (y1 - y0) / (x1 - x0)
            lines.append(f"Slope between first two points: ({y1} - {y0}) / ({x1} - {x0}) = {slope}")
            lines.append("")
    lines.append(f"Hypothesis: {rule_desc}.")
    lines.append("Verification against all points:")
    for x, y in points:
        lines.append(f"  f({x}) = {y} matches.")
    lines.append("")
    lines.append(f"Computing f({test_x}):")
    lines.append(f"  {steps}")
    lines.append(f"Result: {answer}")
    return str(answer), "\n".join(lines)


# ---------------------------------------------------------------------------
# Sequence solver
# ---------------------------------------------------------------------------

def _parse_sequence_terms(prompt: str) -> Optional[List[int]]:
    """Find a CSV-like list of integers in the prompt."""
    # Look for a comma-separated number list of length >= 4
    candidates = re.findall(r"(?:-?\d+\s*,\s*){3,}-?\d+", prompt)
    if not candidates:
        return None
    best = max(candidates, key=lambda s: s.count(","))
    try:
        return [int(x.strip()) for x in best.split(",")]
    except ValueError:
        return None


def solve_sequence(prompt: str) -> Optional[Tuple[str, str]]:
    terms = _parse_sequence_terms(prompt)
    if terms is None or len(terms) < 3:
        return None

    rule_desc: Optional[str] = None
    answer: Optional[int] = None
    steps: Optional[str] = None

    diffs = [terms[i + 1] - terms[i] for i in range(len(terms) - 1)]

    # Arithmetic
    if len(set(diffs)) == 1:
        d = diffs[0]
        rule_desc = f"Arithmetic with common difference {d}"
        answer = terms[-1] + d
        steps = f"Each term increases by {d}, so next = {terms[-1]} + {d} = {answer}"

    # Geometric
    if rule_desc is None and all(t != 0 for t in terms[:-1]):
        ratios = [terms[i + 1] / terms[i] for i in range(len(terms) - 1)]
        if len(set(ratios)) == 1:
            r = ratios[0]
            if abs(r - round(r)) < 1e-9:
                ri = int(round(r))
                if all(terms[i + 1] == terms[i] * ri for i in range(len(terms) - 1)):
                    rule_desc = f"Geometric with ratio {ri}"
                    answer = terms[-1] * ri
                    steps = f"Each term multiplied by {ri}, so next = {terms[-1]} * {ri} = {answer}"

    # Fibonacci-like
    if rule_desc is None and len(terms) >= 3:
        if all(terms[i + 2] == terms[i] + terms[i + 1] for i in range(len(terms) - 2)):
            rule_desc = "Fibonacci-like (each term is sum of previous two)"
            answer = terms[-1] + terms[-2]
            steps = f"Next = {terms[-1]} + {terms[-2]} = {answer}"

    # Constant second difference
    if rule_desc is None and len(diffs) >= 2:
        diffs2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        if len(set(diffs2)) == 1:
            d2 = diffs2[0]
            next_diff = diffs[-1] + d2
            answer = terms[-1] + next_diff
            rule_desc = f"Second-order arithmetic (differences increase by {d2})"
            steps = (
                f"First differences: {diffs}. "
                f"Second differences: {diffs2}. "
                f"Next diff = {diffs[-1]} + {d2} = {next_diff}, "
                f"next term = {terms[-1]} + {next_diff} = {answer}"
            )

    # Quadratic n^2 + c (try small offsets)
    if rule_desc is None:
        for c in range(-5, 6):
            ok = all(terms[i] == (i + 1) ** 2 + c for i in range(len(terms)))
            if ok:
                rule_desc = f"Quadratic n^2 + {c}"
                k = len(terms) + 1
                answer = k * k + c
                steps = f"Term {k} = {k}^2 + {c} = {k * k} + {c} = {answer}"
                break

    # Cubic a*n^3 + c
    if rule_desc is None:
        for a in range(1, 5):
            for c in range(-5, 11):
                ok = all(terms[i] == a * (i + 1) ** 3 + c for i in range(len(terms)))
                if ok:
                    k = len(terms) + 1
                    rule_desc = f"Cubic a_n = {a}*n^3 + {c}"
                    answer = a * (k ** 3) + c
                    steps = f"Term {k} = {a} * {k}^3 + {c} = {a * k ** 3} + {c} = {answer}"
                    break
            if rule_desc is not None:
                break

    # Alternating operations: +pa, -pb, +pa, -pb...
    if rule_desc is None and len(diffs) >= 4:
        # diffs at even/odd positions are constant
        even_diffs = [diffs[i] for i in range(0, len(diffs), 2)]
        odd_diffs = [diffs[i] for i in range(1, len(diffs), 2)]
        if len(set(even_diffs)) == 1 and len(set(odd_diffs)) == 1 and even_diffs[0] != odd_diffs[0]:
            d_even = even_diffs[0]
            d_odd = odd_diffs[0]
            # next diff alternates
            next_diff = d_even if len(diffs) % 2 == 0 else d_odd
            answer = terms[-1] + next_diff
            rule_desc = f"Alternating differences: +{d_even}, +{d_odd}, +{d_even}, +{d_odd}, ..."
            steps = (
                f"Differences alternate between {d_even} and {d_odd}. "
                f"Next difference is {next_diff}, so next term = {terms[-1]} + {next_diff} = {answer}"
            )

    # Interleaved sub-sequences. Two independent series occupy even/odd indices.
    # Try arithmetic, then geometric on each sub-series, and predict the next
    # term based on which parity the next index falls under.
    if rule_desc is None and len(terms) >= 4:
        even_terms = terms[0::2]
        odd_terms = terms[1::2]

        def _arith_next(seq: list[int]) -> Optional[Tuple[int, int, str]]:
            if len(seq) < 2:
                return None
            d = seq[1] - seq[0]
            if all(seq[i + 1] - seq[i] == d for i in range(len(seq) - 1)):
                return seq[-1] + d, d, "arithmetic"
            return None

        def _geom_next(seq: list[int]) -> Optional[Tuple[int, int, str]]:
            if len(seq) < 2 or any(t == 0 for t in seq):
                return None
            if seq[0] == 0:
                return None
            r = seq[1] / seq[0]
            if abs(r - round(r)) > 1e-9:
                return None
            ri = int(round(r))
            if all(seq[i + 1] == seq[i] * ri for i in range(len(seq) - 1)):
                return seq[-1] * ri, ri, "geometric"
            return None

        def _detect(seq: list[int]) -> Optional[Tuple[int, str]]:
            for fn in (_arith_next, _geom_next):
                result = fn(seq)
                if result is not None:
                    nxt, op, kind = result
                    if kind == "arithmetic":
                        return nxt, f"arithmetic d={op}"
                    return nxt, f"geometric r={op}"
            return None

        det_even = _detect(even_terms)
        det_odd = _detect(odd_terms)
        if det_even is not None and det_odd is not None:
            # Index of the predicted next term in the original `terms` series.
            next_idx = len(terms)  # 0-based; positions parity is next_idx % 2
            if next_idx % 2 == 0:
                answer = det_even[0]
                which = "even-indexed"
                detail = det_even[1]
            else:
                answer = det_odd[0]
                which = "odd-indexed"
                detail = det_odd[1]
            rule_desc = (
                f"Interleaved sub-sequences: "
                f"even-indexed ({det_even[1]}), odd-indexed ({det_odd[1]})"
            )
            steps = (
                f"Even-indexed subsequence: {even_terms} -> next = {det_even[0]} ({det_even[1]}). "
                f"Odd-indexed subsequence: {odd_terms} -> next = {det_odd[0]} ({det_odd[1]}). "
                f"The next position (index {next_idx}) is {which}, so the answer is {answer} ({detail})."
            )

    if rule_desc is None or answer is None or steps is None:
        return None

    lines = [
        "I need to find the pattern in the sequence and predict the next term.",
        "",
        f"Given terms: {', '.join(str(t) for t in terms)}",
        "",
        f"Successive differences: {', '.join(str(d) for d in diffs)}",
        "",
        f"Pattern identified: {rule_desc}.",
        "",
        f"Computing the next term:",
        f"  {steps}",
        f"Result: {answer}",
    ]
    return str(answer), "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SOLVER_TABLE: List[Tuple[PuzzleType, Callable[[str], Optional[Tuple[str, str]]]]] = [
    ("bit_manipulation", solve_bit),
    ("text_cipher", solve_cipher),
    ("algebraic", solve_algebraic),
    ("sequence", solve_sequence),
]


def solve_puzzle(prompt: str) -> Optional[Tuple[PuzzleType, str, str]]:
    """Try all solvers in order. Returns (puzzle_type, answer, cot) or None."""
    for ptype, fn in _SOLVER_TABLE:
        try:
            res = fn(prompt)
        except Exception:
            res = None
        if res is not None:
            answer, cot = res
            return ptype, answer, cot
    return None
