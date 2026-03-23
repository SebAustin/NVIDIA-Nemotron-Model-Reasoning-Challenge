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

from scripts.utils.data_formatter import build_messages, format_assistant_reply


def _rand_bits(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("01") for _ in range(n))


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


RuleFn = Callable[[str], str]


@dataclass
class PuzzleExample:
    inp: str
    out: str


def _format_bit_prompt(
    story: str,
    rule_name: str,
    examples: List[PuzzleExample],
    test_inp: str,
) -> str:
    lines = [
        f"In {story}, a secret bit manipulation rule transforms any 8-bit binary string.",
        f"Rule hint: {rule_name}",
        "",
        "Here are examples:",
    ]
    for ex in examples:
        lines.append(f"Input: {ex.inp}  →  Output: {ex.out}")
    lines.append("")
    lines.append(f"Now apply the same rule to this input: {test_inp}")
    lines.append("Respond with only the 8-bit output after you have verified the pattern.")
    return "\n".join(lines)


def generate_bit_manipulation_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    """Return (prompt, answer, cot_without_box)."""
    rule_id = rng.randint(0, 5)
    if rule_id == 0:
        mask = _rand_bits(rng)
        rule_desc = f"XOR with fixed mask {mask}"

        def rule(b: str) -> str:
            return _apply_xor_mask(b, mask)

    elif rule_id == 1:
        k = rng.randint(1, 7)

        def rule(b: str) -> str:
            return _rotl(b, k)

        rule_desc = f"Rotate left by {k} bits"
    elif rule_id == 2:

        def rule(b: str) -> str:
            return _reverse_bits(b)

        rule_desc = "Reverse all 8 bits"
    elif rule_id == 3:

        def rule(b: str) -> str:
            return _swap_nibbles(b)

        rule_desc = "Swap upper and lower 4 bits (nibbles)"
    elif rule_id == 4:
        idx = rng.randint(0, 7)

        def rule(b: str) -> str:
            return _flip_bit(b, idx)

        rule_desc = f"Flip bit at position {idx} (0=leftmost)"
    else:
        m = rng.randint(1, 255)

        def rule(b: str) -> str:
            v = int(b, 2)
            return format((v + m) % 256, "08b")

        rule_desc = f"Interpret as unsigned int, add {m} mod 256"

    examples: List[PuzzleExample] = []
    seen = set()
    while len(examples) < rng.randint(5, 7):
        x = _rand_bits(rng)
        if x in seen:
            continue
        seen.add(x)
        examples.append(PuzzleExample(x, rule(x)))
    test_inp = _rand_bits(rng)
    while test_inp in seen:
        test_inp = _rand_bits(rng)
    answer = rule(test_inp)
    story = "Alice's Wonderland" if rng.random() < 0.5 else "the Enigma Gardens"
    prompt = _format_bit_prompt(story, rule_desc, examples, test_inp)
    cot = _bit_cot(rule_desc, examples, test_inp, answer)
    return prompt, answer, cot


def _bit_cot(
    rule_desc: str,
    examples: List[PuzzleExample],
    test_inp: str,
    answer: str,
) -> str:
    lines = [
        "I will check each example to infer a single consistent 8-bit transformation.",
        f"The stated hint is: {rule_desc}.",
        "Checking examples:",
    ]
    for ex in examples:
        lines.append(f"- {ex.inp} → {ex.out}")
    lines.append("The rule matches all shown pairs.")
    lines.append(f"Apply the same rule to {test_inp} → {answer}.")
    return "\n".join(lines)


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


def generate_cipher_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    mode = rng.randint(0, 4)
    words = [
        "HELLO",
        "SECRET",
        "PUZZLE",
        "NVIDIA",
        "REASON",
        "LOGIC",
        "CIPHER",
        "STREAM",
    ]
    if mode == 0:
        k = rng.randint(1, 25)

        def enc(s: str) -> str:
            return _caesar(s.upper(), k)

        hint = f"Caesar shift forward by {k} on A–Z"
    elif mode == 1:

        def enc(s: str) -> str:
            return _reverse_text(s.upper())

        hint = "Reverse the entire string"
    elif mode == 2:
        sub_map = {}
        letters = list(string.ascii_uppercase)
        rng.shuffle(letters)
        for a, b in zip(string.ascii_uppercase, letters):
            sub_map[a] = b

        def enc(s: str) -> str:
            return "".join(sub_map.get(c, c) for c in s.upper())

        hint = "Simple substitution on A–Z (fixed permutation)"
    elif mode == 3:
        key = rng.choice(["KEY", "CODE", "NEMO", "BYTE"])

        def enc(s: str) -> str:
            return _vigenere(s.upper(), key)

        hint = f"Vigenère with keyword {key}"
    else:

        def enc(s: str) -> str:
            return "".join(
                chr(((ord(c.upper()) - ord("A") + 3) % 26) + ord("A"))
                if c.isalpha()
                else c
                for c in s
            )

        hint = "Shift vowels? No — rotate every letter by +3 (wrap A–Z)"

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
        f"Rule hint: {hint}",
        "",
        "Examples:",
    ]
    for ex in examples:
        lines.append(f"Input: {ex.inp}  →  Output: {ex.out}")
    lines.append("")
    lines.append(f"Apply the rule to: {test_word}")
    prompt = "\n".join(lines)
    cot = _cipher_cot(hint, examples, test_word, answer)
    return prompt, answer, cot


def _cipher_cot(
    hint: str,
    examples: List[PuzzleExample],
    test_word: str,
    answer: str,
) -> str:
    lines = [
        f"Rule hint: {hint}.",
        "Examples:",
    ]
    for ex in examples:
        lines.append(f"- {ex.inp} → {ex.out}")
    lines.append(f"Applying the same mapping to {test_word} yields {answer}.")
    return "\n".join(lines)


def generate_algebraic_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    kind = rng.randint(0, 3)
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
    else:
        mul = rng.randint(2, 9)
        add = rng.randint(1, 30)

        def f(x: int) -> int:
            return mul * x + add

        desc = f"f(x) = {mul}*x + {add}"

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
        "Numeric puzzle: infer f from examples (integers).",
        f"Structure hint: {desc}",
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
    lines = [f"Hypothesis: {desc}", "Check examples:"]
    for ex in examples:
        lines.append(f"- f({ex.inp}) = {ex.out}")
    lines.append(f"Therefore f({test_x}) = {answer}.")
    return "\n".join(lines)


def generate_sequence_puzzle(
    rng: random.Random,
) -> Tuple[str, str, str]:
    mode = rng.randint(0, 3)
    if mode == 0:
        a0 = rng.randint(0, 10)
        d = rng.randint(1, 9)
        seq = [a0 + i * d for i in range(7)]
        rule = f"Arithmetic: start {a0}, add {d}"
    elif mode == 1:
        a0 = rng.randint(1, 5)
        r = rng.choice([2, 3])
        seq = [a0 * (r**i) for i in range(7)]
        rule = f"Geometric: first term {a0}, ratio {r}"
    elif mode == 2:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        seq = []
        x, y = a, b
        for _ in range(7):
            seq.append(x)
            x, y = y, x + y
        rule = f"Fibonacci-like: start {a},{b}, each term sum of previous two"
    else:
        seq = [i * i + rng.randint(0, 2) for i in range(1, 8)]
        rule = "Quadratic-ish: n^2 plus small offset (fit from terms)"

    shown = seq[:-1]
    answer = str(seq[-1])
    lines = [
        "Sequence puzzle: find the next term.",
        f"Hint: {rule}",
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
        f"Pattern note: {rule}.",
        f"Observed: {', '.join(str(x) for x in shown)}.",
        f"Next term: {answer}.",
    ]
    return "\n".join(lines)


def puzzle_to_jsonl_record(
    prompt: str,
    answer: str,
    cot: str,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    assistant = format_assistant_reply(cot, answer)
    rec: Dict[str, Any] = {"messages": build_messages(prompt, assistant)}
    if meta:
        rec["meta"] = meta
    return rec


def write_synthetic_shard(
    path: Path,
    generator: Callable[[random.Random], Tuple[str, str, str]],
    puzzle_type: str,
    n: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            prompt, answer, cot = generator(rng)
            rec = puzzle_to_jsonl_record(
                prompt,
                answer,
                cot,
                meta={"puzzle_type": puzzle_type, "synthetic": True, "idx": i},
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
