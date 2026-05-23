"""Synthetic reasoning puzzles (bit rules, ciphers, algebra, sequences) → JSONL-ready examples."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import string
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from scripts.utils.data_formatter import (
    build_messages,
    format_assistant_reply,
    format_assistant_reply_verified,
)


def _rand_bits(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("01") for _ in range(n))


def _int_to_bits(v: int, n: int) -> str:
    return format(v % (1 << n), f"0{n}b")


def _shr_logical(bits: str, k: int) -> str:
    k %= len(bits)
    return ("0" * k) + bits[: len(bits) - k]


def _shl_logical(bits: str, k: int) -> str:
    k %= len(bits)
    return bits[k:] + ("0" * k)


def _apply_xor_mask(bits: str, mask: str) -> str:
    return "".join("1" if b != m else "0" for b, m in zip(bits, mask))


def _rotl(bits: str, k: int) -> str:
    k %= len(bits)
    return bits[k:] + bits[:k]


def _reverse_bits(bits: str) -> str:
    return bits[::-1]


def _swap_nibbles(bits: str) -> str:
    if len(bits) != 8:
        raise ValueError("expected 8 bits")
    return bits[4:] + bits[:4]


def _flip_bit(bits: str, idx: int) -> str:
    b = list(bits)
    b[idx] = "0" if b[idx] == "1" else "1"
    return "".join(b)


def _complement(bits: str) -> str:
    return "".join("0" if c == "1" else "1" for c in bits)


RuleFn = Callable[[str], str]


@dataclass
class PuzzleExample:
    inp: str
    out: str


# ---------------------------------------------------------------------------
# Bit manipulation puzzles
# ---------------------------------------------------------------------------

def _format_bit_prompt(
    story: str,
    examples: List[PuzzleExample],
    test_inp: str,
) -> str:
    return _format_bit_prompt_n(story, examples, test_inp, 8)


def _format_bit_prompt_n(
    story: str,
    examples: List[PuzzleExample],
    test_inp: str,
    n_bits: int,
) -> str:
    lines = [
        f"In {story}, a secret bit manipulation rule transforms any {n_bits}-bit binary string.",
        "",
        "Here are examples of the transformation:",
    ]
    for ex in examples:
        lines.append(f"Input: {ex.inp}  →  Output: {ex.out}")
    lines.append("")
    lines.append(f"Now apply the same rule to this input: {test_inp}")
    lines.append(f"Respond with only the {n_bits}-bit output after you have verified the pattern.")
    return "\n".join(lines)


def generate_bit_manipulation_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    """Return (prompt, answer, cot_without_box)."""
    # 12 rule families incl. 16-bit, compositions, arithmetic shifts.
    rule_id = rng.randint(0, 11)
    n_bits = 16 if rule_id in (8, 9, 10, 11) else 8

    def _rb() -> str:
        return _rand_bits(rng, n_bits)

    if rule_id == 0:
        mask = _rand_bits(rng, n_bits)
        rule_desc = f"XOR with fixed mask {mask}"

        def rule(b: str) -> str:
            return _apply_xor_mask(b, mask)

    elif rule_id == 1:
        k = rng.randint(1, n_bits - 1)

        def rule(b: str) -> str:
            return _rotl(b, k)

        rule_desc = f"Rotate left by {k} bits"
    elif rule_id == 2:

        def rule(b: str) -> str:
            return _reverse_bits(b)

        rule_desc = f"Reverse all {n_bits} bits"
    elif rule_id == 3:

        def rule(b: str) -> str:
            return _swap_nibbles(b)

        rule_desc = "Swap upper and lower 4 bits (nibbles)"
    elif rule_id == 4:
        idx = rng.randint(0, n_bits - 1)

        def rule(b: str) -> str:
            return _flip_bit(b, idx)

        rule_desc = f"Flip bit at position {idx} (0=leftmost)"
    elif rule_id == 5:
        mod = 1 << n_bits
        m = rng.randint(1, mod - 1)

        def rule(b: str) -> str:
            v = int(b, 2)
            return _int_to_bits(v + m, n_bits)

        rule_desc = f"Interpret as unsigned int, add {m} mod {mod}"
    elif rule_id == 6:

        def rule(b: str) -> str:
            return _complement(b)

        rule_desc = "Bitwise complement (flip every bit)"
    elif rule_id == 7:
        k = rng.randint(1, n_bits - 1)
        mask = _rand_bits(rng, n_bits)

        def rule(b: str) -> str:
            return _apply_xor_mask(_rotl(b, k), mask)

        rule_desc = f"Rotate left by {k} then XOR with {mask}"
    elif rule_id == 8:
        # 16-bit: reverse then complement
        def rule(b: str) -> str:
            return _complement(_reverse_bits(b))

        rule_desc = "Reverse all 16 bits then take bitwise complement"
    elif rule_id == 9:
        # 16-bit: swap halves (upper 8 <-> lower 8)
        def rule(b: str) -> str:
            return b[8:] + b[:8]

        rule_desc = "Swap the two 8-bit halves"
    elif rule_id == 10:
        # 16-bit: logical left shift by k (fills right with zeros)
        k = rng.randint(1, 7)

        def rule(b: str) -> str:
            return _shl_logical(b, k)

        rule_desc = f"Logical left shift by {k} bits (zero-fill on right)"
    else:
        # 16-bit: add m mod 2^16 then XOR with mask
        mod = 1 << n_bits
        m = rng.randint(1, mod - 1)
        mask = _rand_bits(rng, n_bits)

        def rule(b: str) -> str:
            v = int(b, 2)
            return _apply_xor_mask(_int_to_bits(v + m, n_bits), mask)

        rule_desc = f"Add {m} mod {mod} then XOR with {mask}"

    examples: List[PuzzleExample] = []
    seen = set()
    while len(examples) < rng.randint(5, 7):
        x = _rb()
        if x in seen:
            continue
        seen.add(x)
        examples.append(PuzzleExample(x, rule(x)))
    test_inp = _rb()
    while test_inp in seen:
        test_inp = _rb()
    answer = rule(test_inp)
    story = rng.choice([
        "Alice's Wonderland", "the Enigma Gardens", "the Binary Fortress",
        "the Circuit Maze", "the Digital Archives", "the Quantum Vault",
        "the Silicon Labyrinth", "the Cipher Foundry",
    ])
    prompt = _format_bit_prompt_n(story, examples, test_inp, n_bits)
    cot = _bit_cot(rule_desc, examples, test_inp, answer)
    return prompt, answer, cot


def _bit_cot(
    rule_desc: str,
    examples: List[PuzzleExample],
    test_inp: str,
    answer: str,
) -> str:
    lines = [
        "Let me analyze each input-output pair to discover the transformation rule.",
        "",
    ]
    ex0 = examples[0]
    lines.append(f"Looking at the first example: {ex0.inp} → {ex0.out}")
    lines.append("Comparing bit by bit:")
    diffs = []
    for i, (a, b) in enumerate(zip(ex0.inp, ex0.out)):
        status = "same" if a == b else "flipped"
        diffs.append(status)
        lines.append(f"  Position {i}: {a} → {b} ({status})")

    lines.append("")
    lines.append("Let me check if this pattern holds for the other examples:")
    for ex in examples[1:]:
        lines.append(f"  {ex.inp} → {ex.out}")
    lines.append("")
    lines.append(f"After analyzing all pairs, I can see the rule is: {rule_desc}.")
    lines.append("")
    lines.append("Verification against all examples:")
    for ex in examples:
        lines.append(f"  {ex.inp} → {ex.out} ✓")
    lines.append("")
    lines.append(f"Applying the rule to the test input {test_inp}:")
    lines.append(f"Result: {answer}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cipher puzzles
# ---------------------------------------------------------------------------

def _caesar_shift(ch: str, k: int) -> str:
    if not ch.isalpha():
        return ch
    base = ord("A") if ch.isupper() else ord("a")
    o = (ord(ch) - base + k) % 26 + base
    return chr(o)


def _caesar(text: str, k: int) -> str:
    return "".join(_caesar_shift(c, k) for c in text)


def _reverse_text(text: str) -> str:
    return text[::-1]


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


_CIPHER_WORDS = [
    "HELLO", "SECRET", "PUZZLE", "NVIDIA", "REASON", "LOGIC", "CIPHER",
    "STREAM", "KERNEL", "MATRIX", "VECTOR", "PYTHON", "DELTA", "SIGMA",
    "OMEGA", "ALPHA", "BRAVO", "GAMMA", "TORCH", "MODEL", "TRAIN",
    "SCORE", "BRAIN", "THINK", "QUERY",
]


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


def generate_cipher_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    # 8 cipher families incl. atbash, affine, caesar+reverse composition.
    mode = rng.randint(0, 7)
    words = list(_CIPHER_WORDS)
    if mode == 0:
        k = rng.randint(1, 25)

        def enc(s: str) -> str:
            return _caesar(s.upper(), k)

        hint_internal = f"Caesar shift forward by {k}"
    elif mode == 1:

        def enc(s: str) -> str:
            return _reverse_text(s.upper())

        hint_internal = "Reverse the entire string"
    elif mode == 2:
        sub_map = {}
        letters = list(string.ascii_uppercase)
        rng.shuffle(letters)
        for a, b in zip(string.ascii_uppercase, letters):
            sub_map[a] = b

        def enc(s: str) -> str:
            return "".join(sub_map.get(c, c) for c in s.upper())

        hint_internal = "Simple substitution (fixed permutation)"
    elif mode == 3:
        key = rng.choice(["KEY", "CODE", "NEMO", "BYTE", "CUDA", "DEEP", "LOGIC", "RULE"])

        def enc(s: str) -> str:
            return _vigenere(s.upper(), key)

        hint_internal = f"Vigenère with keyword {key}"
    elif mode == 4:

        def enc(s: str) -> str:
            return "".join(
                chr(((ord(c.upper()) - ord("A") + 3) % 26) + ord("A"))
                if c.isalpha()
                else c
                for c in s
            )

        hint_internal = "Rotate every letter by +3 (Caesar +3)"
    elif mode == 5:

        def enc(s: str) -> str:
            return _atbash(s.upper())

        hint_internal = "Atbash cipher (A<->Z, B<->Y, ...)"
    elif mode == 6:
        # Affine: pick a coprime with 26
        a = rng.choice([3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
        b = rng.randint(1, 25)

        def enc(s: str) -> str:
            return _affine(s.upper(), a, b)

        hint_internal = f"Affine cipher y = {a}*x + {b} mod 26"
    else:
        # Composition: Caesar shift then reverse
        k = rng.randint(1, 25)

        def enc(s: str) -> str:
            return _reverse_text(_caesar(s.upper(), k))

        hint_internal = f"Caesar shift +{k} then reverse the string"

    examples: List[PuzzleExample] = []
    seen = set()
    while len(examples) < rng.randint(5, 7):
        w = rng.choice(words)
        if w in seen:
            continue
        seen.add(w)
        examples.append(PuzzleExample(w, enc(w)))
    test_word = rng.choice([x for x in words if x not in seen] or words)
    answer = enc(test_word)
    lines = [
        "In the Whispering Library, a text transformation rule maps short uppercase words.",
        "",
        "Examples:",
    ]
    for ex in examples:
        lines.append(f"Input: {ex.inp}  →  Output: {ex.out}")
    lines.append("")
    lines.append(f"Apply the same rule to: {test_word}")
    prompt = "\n".join(lines)
    cot = _cipher_cot(hint_internal, examples, test_word, answer)
    return prompt, answer, cot


def _cipher_cot(
    hint_internal: str,
    examples: List[PuzzleExample],
    test_word: str,
    answer: str,
) -> str:
    ex0 = examples[0]
    lines = [
        "Let me analyze the letter-by-letter mapping in each example.",
        "",
        f"First example: {ex0.inp} → {ex0.out}",
    ]
    for i, (a, b) in enumerate(zip(ex0.inp, ex0.out)):
        shift = (ord(b) - ord(a)) % 26
        lines.append(f"  {a} → {b} (shift +{shift})")

    lines.append("")
    lines.append("Checking if the same pattern holds for other examples:")
    for ex in examples[1:3]:
        pairs = ", ".join(f"{a}→{b}" for a, b in zip(ex.inp, ex.out))
        lines.append(f"  {ex.inp} → {ex.out}: {pairs}")

    lines.append("")
    lines.append(f"The transformation is consistent: {hint_internal}.")
    lines.append("")
    lines.append("Verification:")
    for ex in examples:
        lines.append(f"  {ex.inp} → {ex.out} ✓")
    lines.append("")
    lines.append(f"Applying to {test_word}:")
    step_by_step = ", ".join(f"{a}→{b}" for a, b in zip(test_word, answer))
    lines.append(f"  {step_by_step}")
    lines.append(f"Result: {answer}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Algebraic puzzles
# ---------------------------------------------------------------------------

def generate_algebraic_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    # 10 function families incl. cubic, piecewise, mod compositions.
    kind = rng.randint(0, 9)
    if kind == 0:
        a, b, c = rng.randint(-5, 5), rng.randint(-20, 20), rng.randint(-50, 50)

        def f(x: int) -> int:
            return a * x * x + b * x + c

        desc = f"f(x) = {a}*x^2 + {b}*x + {c}"
    elif kind == 1:
        n = rng.randint(5, 17)
        k = rng.randint(0, n - 1)

        def f(x: int) -> int:
            return (x % n) + k

        desc = f"f(x) = (x mod {n}) + {k}"
    elif kind == 2:

        def f(x: int) -> int:
            return sum(int(d) for d in str(abs(x)))

        desc = "f(x) = sum of decimal digits of |x|"
    elif kind == 3:
        mul = rng.randint(2, 9)
        add = rng.randint(1, 30)

        def f(x: int) -> int:
            return mul * x + add

        desc = f"f(x) = {mul}*x + {add}"
    elif kind == 4:
        mul = rng.randint(2, 5)

        def f(x: int) -> int:
            return abs(x) * mul

        desc = f"f(x) = |x| * {mul}"
    elif kind == 5:
        a_val = rng.randint(1, 4)
        b_val = rng.randint(1, 10)

        def f(x: int) -> int:
            return x * x * a_val + b_val

        desc = f"f(x) = {a_val}*x^2 + {b_val}"
    elif kind == 6:
        # Cubic
        a_val = rng.randint(1, 3)
        b_val = rng.randint(-5, 5)

        def f(x: int) -> int:
            return a_val * (x ** 3) + b_val * x

        desc = f"f(x) = {a_val}*x^3 + {b_val}*x"
    elif kind == 7:
        # Piecewise: positive/negative branches
        pos_m = rng.randint(2, 5)
        neg_add = rng.randint(1, 10)

        def f(x: int) -> int:
            return pos_m * x if x >= 0 else x * x + neg_add

        desc = f"f(x) = {pos_m}*x if x>=0 else x^2 + {neg_add}"
    elif kind == 8:
        # Modular linear: (a*x + b) mod n
        a_val = rng.randint(2, 7)
        b_val = rng.randint(0, 10)
        n_val = rng.randint(5, 19)

        def f(x: int) -> int:
            return ((a_val * x) + b_val) % n_val

        desc = f"f(x) = ({a_val}*x + {b_val}) mod {n_val}"
    else:
        # Digit product of abs(x), or 1 for x==0
        def f(x: int) -> int:
            ax = abs(x)
            if ax == 0:
                return 0
            p = 1
            for d in str(ax):
                p *= int(d)
            return p

        desc = "f(x) = product of decimal digits of |x| (0 for x=0)"

    xs = []
    while len(xs) < rng.randint(5, 7):
        x = rng.randint(-15, 15)
        if x in xs:
            continue
        xs.append(x)
    examples = [PuzzleExample(str(x), str(f(x))) for x in xs]
    test_x = rng.randint(-20, 20)
    while test_x in xs:
        test_x = rng.randint(-20, 20)
    answer = str(f(test_x))
    lines = [
        "Numeric puzzle: a hidden function f maps integers to integers.",
        "Examine the examples and determine f, then compute the answer.",
        "",
        "Examples (x → f(x)):",
    ]
    for ex in examples:
        lines.append(f"{ex.inp} → {ex.out}")
    lines.append("")
    lines.append(f"Compute f({test_x}).")
    prompt = "\n".join(lines)
    cot = _algebra_cot(desc, examples, test_x, answer)
    return prompt, answer, cot


def _algebra_cot(
    desc: str,
    examples: List[PuzzleExample],
    test_x: int,
    answer: str,
) -> str:
    lines = [
        "I need to find a function f that maps each input to its output.",
        "",
        "Let me examine the input-output pairs:",
    ]
    for ex in examples:
        lines.append(f"  f({ex.inp}) = {ex.out}")

    lines.append("")
    ex0, ex1 = examples[0], examples[1]
    x0, y0 = int(ex0.inp), int(ex0.out)
    x1, y1 = int(ex1.inp), int(ex1.out)
    if x1 != x0:
        slope = (y1 - y0) / (x1 - x0)
        lines.append(f"Between x={x0} and x={x1}: change in output = {y1 - y0}, change in input = {x1 - x0}")
        lines.append(f"Slope estimate: {slope}")
    lines.append("")
    lines.append(f"Testing hypothesis: {desc}")
    lines.append("")
    lines.append("Verification against all examples:")
    for ex in examples:
        lines.append(f"  f({ex.inp}) = {ex.out} ✓")
    lines.append("")
    lines.append(f"All examples check out. Now computing f({test_x}):")
    lines.append(f"f({test_x}) = {answer}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sequence puzzles
# ---------------------------------------------------------------------------

def generate_sequence_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    # 8 sequence families incl. alternating, interleaved, cubic.
    mode = rng.randint(0, 7)
    if mode == 0:
        a0 = rng.randint(0, 10)
        d = rng.randint(1, 9)
        seq = [a0 + i * d for i in range(7)]
        rule = f"Arithmetic: start {a0}, common difference {d}"
    elif mode == 1:
        a0 = rng.randint(1, 5)
        r = rng.choice([2, 3])
        seq = [a0 * (r ** i) for i in range(7)]
        rule = f"Geometric: first term {a0}, ratio {r}"
    elif mode == 2:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        seq = []
        x, y = a, b
        for _ in range(7):
            seq.append(x)
            x, y = y, x + y
        rule = f"Fibonacci-like: start {a},{b}, each term is sum of previous two"
    elif mode == 3:
        c = rng.randint(0, 2)
        seq = [i * i + c for i in range(1, 8)]
        rule = f"Quadratic: n^2 + {c}"
    elif mode == 4:
        a0 = rng.randint(1, 5)
        d1 = rng.randint(1, 3)
        seq = [a0]
        diff = d1
        for _ in range(6):
            seq.append(seq[-1] + diff)
            diff += d1
        rule = f"Second-order arithmetic: differences increase by {d1}"
    elif mode == 5:
        # Alternating +a, -b
        a0 = rng.randint(1, 10)
        pa = rng.randint(2, 7)
        pb = rng.randint(1, 5)
        seq = [a0]
        for i in range(6):
            seq.append(seq[-1] + (pa if i % 2 == 0 else -pb))
        rule = f"Alternating operations: +{pa}, -{pb}, +{pa}, -{pb}, ..."
    elif mode == 6:
        # a_n = a*n^3 + c
        a_val = rng.randint(1, 3)
        c = rng.randint(0, 5)
        seq = [a_val * (n ** 3) + c for n in range(1, 8)]
        rule = f"Cubic: a_n = {a_val}*n^3 + {c}"
    else:
        # Interleaved two sequences (even idx +d1, odd idx *r)
        a0 = rng.randint(1, 6)
        b0 = rng.randint(1, 4)
        d1 = rng.randint(1, 4)
        r = rng.choice([2, 3])
        seq = []
        for i in range(7):
            seq.append(a0 + (i // 2) * d1 if i % 2 == 0 else b0 * (r ** (i // 2)))
        rule = f"Interleaved: even indices arithmetic (+{d1}), odd indices geometric (x{r})"

    shown = seq[:-1]
    answer = str(seq[-1])
    lines = [
        "Sequence puzzle: find the next term.",
        "",
        "Given terms:",
        ", ".join(str(x) for x in shown),
        "",
        "What is the next term?",
    ]
    prompt = "\n".join(lines)
    cot = _sequence_cot(rule, shown, answer)
    return prompt, answer, cot


def _sequence_cot(rule: str, shown: List[int], answer: str) -> str:
    lines = [
        "Let me analyze the sequence to find the pattern.",
        "",
        f"Given: {', '.join(str(x) for x in shown)}",
        "",
    ]
    if len(shown) >= 2:
        diffs = [shown[i + 1] - shown[i] for i in range(len(shown) - 1)]
        lines.append("Computing successive differences:")
        diff_strs = [f"{shown[i+1]} - {shown[i]} = {diffs[i]}" for i in range(len(diffs))]
        lines.append(f"  {', '.join(diff_strs)}")
        lines.append(f"  Differences: {', '.join(str(d) for d in diffs)}")
        lines.append("")

        all_same = len(set(diffs)) == 1
        if all_same:
            lines.append(f"The differences are constant ({diffs[0]}), so this is an arithmetic sequence.")
            lines.append(f"Next term = {shown[-1]} + {diffs[0]} = {answer}")
        else:
            if len(diffs) >= 2:
                diffs2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
                lines.append(f"  Second differences: {', '.join(str(d) for d in diffs2)}")
                lines.append("")

            ratios = []
            for i in range(len(shown) - 1):
                if shown[i] != 0:
                    ratios.append(shown[i + 1] / shown[i])
            if ratios and len(set(round(r, 6) for r in ratios)) == 1:
                lines.append(f"The ratios between terms are constant ({ratios[0]:.0f}), so this is geometric.")
            else:
                lines.append(f"Analyzing the pattern further: {rule}.")

            lines.append(f"Following this pattern, the next term is {answer}.")
    else:
        lines.append(f"Pattern: {rule}.")
        lines.append(f"Next term: {answer}.")

    lines.append("")
    lines.append("Verification: the pattern is consistent with all given terms.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def puzzle_to_jsonl_record(
    prompt: str,
    answer: str,
    cot: str,
    meta: Dict[str, Any] | None = None,
    examples: List[PuzzleExample] | None = None,
) -> Dict[str, Any]:
    if examples:
        assistant = format_assistant_reply_verified(cot, answer, examples)
    else:
        assistant = format_assistant_reply(cot, answer)
    rec: Dict[str, Any] = {"messages": build_messages(prompt, assistant)}
    if meta:
        rec["meta"] = meta
    return rec


def _extract_examples_from_prompt(prompt: str) -> List[PuzzleExample]:
    """Pull 'Input: X -> Output: Y' pairs out of the generated prompt text."""
    out: List[PuzzleExample] = []
    import re as _re

    for m in _re.finditer(r"Input:\s*(\S+)\s*(?:→|->)\s*Output:\s*(\S+)", prompt):
        out.append(PuzzleExample(m.group(1), m.group(2)))
    if out:
        return out
    for m in _re.finditer(r"^\s*(-?\d+)\s*(?:→|->)\s*(-?\d+)\s*$", prompt, _re.MULTILINE):
        out.append(PuzzleExample(m.group(1), m.group(2)))
    return out


def write_synthetic_shard(
    path: Path,
    generator: Callable[[random.Random], Tuple[str, str, str]],
    puzzle_type: str,
    n: int,
    seed: int,
) -> None:
    from scripts.utils.solvers import solve_puzzle

    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    solver_used = 0
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            prompt, answer, cot = generator(rng)
            cot_source = "generator"
            # Prefer solver CoT for consistency with real-data CoTs
            solved = solve_puzzle(prompt)
            if solved is not None:
                _, solver_ans, solver_cot = solved
                if solver_ans == answer:
                    cot = solver_cot
                    cot_source = "solver"
                    solver_used += 1
            examples = _extract_examples_from_prompt(prompt)
            rec = puzzle_to_jsonl_record(
                prompt,
                answer,
                cot,
                meta={
                    "puzzle_type": puzzle_type,
                    "synthetic": True,
                    "idx": i,
                    "cot_source": cot_source,
                },
                examples=examples,
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {puzzle_type}: solver CoT used on {solver_used}/{n} ({100*solver_used/max(n,1):.0f}%)")


def write_all_synthetic(
    out_dir: Path,
    per_kind: int = 250,
    seed: int = 42,
    per_kind_overrides: Dict[str, int] | None = None,
) -> Dict[str, Path]:
    """Write four shards. Optional per_kind_overrides keys: bit_manipulation, text_cipher, algebraic, sequence."""
    overrides = per_kind_overrides or {}

    def n(kind_key: str) -> int:
        return int(overrides.get(kind_key, per_kind))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bit_manipulation": out_dir / "bit_manipulation.jsonl",
        "text_cipher": out_dir / "text_cipher.jsonl",
        "algebraic": out_dir / "algebraic.jsonl",
        "sequence": out_dir / "sequence.jsonl",
    }
    write_synthetic_shard(
        paths["bit_manipulation"],
        generate_bit_manipulation_puzzle,
        "bit_manipulation",
        n("bit_manipulation"),
        seed,
    )
    write_synthetic_shard(
        paths["text_cipher"],
        generate_cipher_puzzle,
        "text_cipher",
        n("text_cipher"),
        seed + 1,
    )
    write_synthetic_shard(
        paths["algebraic"],
        generate_algebraic_puzzle,
        "algebraic",
        n("algebraic"),
        seed + 2,
    )
    write_synthetic_shard(
        paths["sequence"],
        generate_sequence_puzzle,
        "sequence",
        n("sequence"),
        seed + 3,
    )
    return paths


def main_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Write synthetic puzzle JSONL shards")
    p.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--per-kind", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    paths = write_all_synthetic(args.out_dir, args.per_kind, args.seed)
    for k, v in paths.items():
        print(f"Wrote {k}: {v}")


if __name__ == "__main__":
    main_cli()
