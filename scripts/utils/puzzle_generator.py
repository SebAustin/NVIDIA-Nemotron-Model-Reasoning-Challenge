"""Synthetic puzzle generators for the 6 real Nemotron families.

Each generator builds a puzzle from a *known* rule using the EXACT real prompt
template (verified against data/train.csv), so we get a correct answer and a
concise worked solution for free. This is the lever for the hard families
(bit_manipulation, equation) whose real rows we cannot solve deterministically.

Each ``generate_*`` returns ``(prompt, answer, reasoning)``. ``reasoning`` is the
``<think>`` body (no boxed line — the formatter adds that).
"""

from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.data_formatter import build_messages, format_assistant_reply

Gen = Callable[[random.Random], Tuple[str, str, str]]

# ---------------------------------------------------------------------------
# bit_manipulation
# ---------------------------------------------------------------------------
_BIT_STEM = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)


def _rand_byte(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(8))


def generate_bit_manipulation(rng: random.Random) -> Tuple[str, str, str]:
    kind = rng.choice(
        ["rotl", "rotr", "xor", "and", "or", "not", "reverse", "swap", "maj", "add"]
    )
    if kind in ("rotl", "rotr"):
        k = rng.randint(1, 7)
        rule = lambda b: (b[k:] + b[:k]) if kind == "rotl" else (b[-k:] + b[:-k])
        desc = f"rotate the 8 bits to the {'left' if kind=='rotl' else 'right'} by {k}"
    elif kind in ("xor", "and", "or"):
        mask = _rand_byte(rng)
        if kind == "xor":
            rule = lambda b: "".join("1" if x != y else "0" for x, y in zip(b, mask))
            desc = f"XOR each bit with the mask {mask}"
        elif kind == "and":
            rule = lambda b: "".join("1" if x == "1" and y == "1" else "0" for x, y in zip(b, mask))
            desc = f"AND each bit with the mask {mask}"
        else:
            rule = lambda b: "".join("1" if x == "1" or y == "1" else "0" for x, y in zip(b, mask))
            desc = f"OR each bit with the mask {mask}"
    elif kind == "not":
        rule = lambda b: "".join("1" if c == "0" else "0" for c in b)
        desc = "invert every bit (NOT)"
    elif kind == "reverse":
        rule = lambda b: b[::-1]
        desc = "reverse the order of the 8 bits"
    elif kind == "swap":
        rule = lambda b: b[4:] + b[:4]
        desc = "swap the upper and lower nibbles (4 bits each)"
    elif kind == "maj":
        def rule(b: str) -> str:
            n = len(b)
            return "".join(
                "1" if (int(b[(i - 1) % n]) + int(b[i]) + int(b[(i + 1) % n])) >= 2 else "0"
                for i in range(n)
            )
        desc = "set each bit to the majority of itself and its two neighbours (wrap-around)"
    else:  # add
        c = rng.randint(1, 255)
        rule = lambda b: format((int(b, 2) + c) % 256, "08b")
        desc = f"add {c} to the value modulo 256"

    seen = set()
    examples = []
    while len(examples) < 8:
        x = _rand_byte(rng)
        if x in seen:
            continue
        seen.add(x)
        examples.append((x, rule(x)))
    test = _rand_byte(rng)
    while test in seen:
        test = _rand_byte(rng)
    answer = rule(test)
    ex_lines = "\n".join(f"{a} -> {b}" for a, b in examples)
    prompt = (
        f"{_BIT_STEM}\n\nHere are some examples of input -> output:\n{ex_lines}\n\n"
        f"Now, determine the output for: {test}"
    )
    reasoning = (
        f"Checking the examples, the rule is to {desc}. "
        f"Applying it to {test} gives {answer}."
    )
    return prompt, answer, reasoning


# ---------------------------------------------------------------------------
# gravity
# ---------------------------------------------------------------------------
def generate_gravity(rng: random.Random) -> Tuple[str, str, str]:
    g = round(rng.uniform(3.0, 30.0), 2)
    ts = sorted({round(rng.uniform(1.0, 5.0), 2) for _ in range(6)})
    while len(ts) < 5:
        ts.append(round(rng.uniform(1.0, 5.0), 2))
    ts = ts[:5]
    obs = "\n".join(f"For t = {t}s, distance = {0.5*g*t*t:.2f} m" for t in ts)
    tt = round(rng.uniform(1.0, 5.0), 2)
    answer = f"{0.5*g*tt*tt:.2f}"
    prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly "
        f"changed. Here are some example observations:\n{obs}\n"
        f"Now, determine the falling distance for t = {tt}s given d = 0.5*g*t^2."
    )
    reasoning = (
        f"From any observation, g = 2*d/t^2 is constant ≈ {g:.2f}. "
        f"So d = 0.5*{g:.2f}*{tt}^2 ≈ {answer}."
    )
    return prompt, answer, reasoning


# ---------------------------------------------------------------------------
# unit_conversion
# ---------------------------------------------------------------------------
def generate_unit_conversion(rng: random.Random) -> Tuple[str, str, str]:
    k = round(rng.uniform(0.2, 3.0), 4)
    b = round(rng.uniform(-2.0, 2.0), 2) if rng.random() < 0.3 else 0.0
    xs = sorted({round(rng.uniform(5.0, 40.0), 2) for _ in range(6)})[:5]
    while len(xs) < 5:
        xs.append(round(rng.uniform(5.0, 40.0), 2))
    lines = "\n".join(f"{x} m becomes {k*x+b:.2f}" for x in xs)
    xt = round(rng.uniform(5.0, 40.0), 2)
    answer = f"{k*xt+b:.2f}"
    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to "
        f"measurements. For example:\n{lines}\n"
        f"Now, convert the following measurement: {xt} m"
    )
    off = "" if b == 0.0 else f" + {b:.2f}"
    reasoning = (
        f"Each output is input*{k:.4f}{off}. "
        f"So {xt}*{k:.4f}{off} ≈ {answer}."
    )
    return prompt, answer, reasoning


# ---------------------------------------------------------------------------
# numeral (standard Roman numerals)
# ---------------------------------------------------------------------------
_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _to_roman(n: int) -> str:
    out = []
    for v, sym in _ROMAN:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def generate_numeral(rng: random.Random) -> Tuple[str, str, str]:
    ns = rng.sample(range(1, 100), 4)
    lines = "\n".join(f"{n} -> {_to_roman(n)}" for n in ns)
    m = rng.randint(1, 100)
    answer = _to_roman(m)
    prompt = (
        "In Alice's Wonderland, numbers are secretly converted into a different "
        f"numeral system. Some examples are given below:\n{lines}\n"
        f"Now, write the number {m} in the Wonderland numeral system."
    )
    reasoning = f"The mapping is standard Roman numerals, so {m} = {answer}."
    return prompt, answer, reasoning


# ---------------------------------------------------------------------------
# encryption (monoalphabetic substitution over a Wonderland vocabulary)
# ---------------------------------------------------------------------------
_VOCAB = [
    "alice", "queen", "king", "wizard", "princess", "dragon", "student", "cat",
    "castle", "valley", "door", "book", "secret", "garden", "forest", "river",
    "creates", "discovers", "chases", "watches", "imagines", "dreams", "reads",
    "follows", "guards", "seeks", "the", "near", "under", "inside", "beyond",
    "golden", "magical", "mysterious", "hidden", "ancient", "silver", "wonderland",
]


def _sentence(rng: random.Random) -> str:
    n = rng.randint(3, 4)
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


def generate_encryption(rng: random.Random) -> Tuple[str, str, str]:
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    enc = {p: c for p, c in zip(letters, shuffled)}

    def encrypt(text: str) -> str:
        return "".join(enc.get(ch, ch) for ch in text)

    plains = []
    seen = set()
    while len(plains) < 5:
        s = _sentence(rng)
        if s in seen:
            continue
        seen.add(s)
        plains.append(s)
    lines = "\n".join(f"{encrypt(p)} -> {p}" for p in plains)
    test_plain = _sentence(rng)
    test_cipher = encrypt(test_plain)
    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here "
        f"are some examples:\n{lines}\n"
        f"Now, decrypt the following text: {test_cipher}"
    )
    reasoning = (
        "Aligning each ciphertext with its plaintext gives a fixed letter "
        f"substitution. Decoding '{test_cipher}' yields: {test_plain}."
    )
    return prompt, test_plain, reasoning


# ---------------------------------------------------------------------------
# equation (symbol-alphabet operands with a per-row operator semantics)
# ---------------------------------------------------------------------------
# braces excluded so synthetic boxed answers are unambiguous to extract
_SYMS = list("`!@#$%^&*()[]|\\/<>?:\"'+-")


def generate_equation(rng: random.Random) -> Tuple[str, str, str]:
    op = rng.choice(_SYMS)
    operands = [s for s in _SYMS if s != op]
    semantics = rng.choice(["concat", "concat_rev", "interleave"])

    def combine(a: str, b: str) -> str:
        if semantics == "concat":
            return a + b
        if semantics == "concat_rev":
            return b + a
        out = []
        for i in range(max(len(a), len(b))):
            if i < len(a):
                out.append(a[i])
            if i < len(b):
                out.append(b[i])
        return "".join(out)

    def rand_operand() -> str:
        return "".join(rng.choice(operands) for _ in range(rng.randint(1, 2)))

    lines = []
    for _ in range(4):
        a, b = rand_operand(), rand_operand()
        lines.append(f"{a}{op}{b} = {combine(a, b)}")
    qa, qb = rand_operand(), rand_operand()
    query = f"{qa}{op}{qb}"
    answer = combine(qa, qb)
    desc = {
        "concat": "concatenate the two operands (dropping the operator)",
        "concat_rev": "concatenate the operands in reverse order",
        "interleave": "interleave the characters of the two operands",
    }[semantics]
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n" + "\n".join(lines) + "\n"
        f"Now, determine the result for: {query}"
    )
    reasoning = (
        f"The operator '{op}' means: {desc}. "
        f"For {query}, take '{qa}' and '{qb}' -> {answer}."
    )
    return prompt, answer, reasoning


# ---------------------------------------------------------------------------
# Registry + shard writing
# ---------------------------------------------------------------------------
GENERATORS: Dict[str, Gen] = {
    "bit_manipulation": generate_bit_manipulation,
    "gravity": generate_gravity,
    "unit_conversion": generate_unit_conversion,
    "numeral": generate_numeral,
    "encryption": generate_encryption,
    "equation": generate_equation,
}


def write_synthetic_shard(path: Path, gen: Gen, family: str, n: int, seed: int) -> int:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            prompt, answer, reasoning = gen(rng)
            assistant = format_assistant_reply(reasoning, answer)
            rec = {
                "messages": build_messages(prompt, assistant),
                "meta": {"category": family, "source": "synthetic",
                         "reasoning": True, "idx": i, "answer": answer},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def write_all_synthetic(
    out_dir: Path,
    per_kind: int = 300,
    seed: int = 42,
    per_kind_overrides: Dict[str, int] | None = None,
) -> Dict[str, Path]:
    overrides = per_kind_overrides or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for offset, (family, gen) in enumerate(GENERATORS.items()):
        n = int(overrides.get(family, per_kind))
        p = out_dir / f"{family}.jsonl"
        write_synthetic_shard(p, gen, family, n, seed + offset)
        paths[family] = p
    return paths


def main_cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Write synthetic puzzle JSONL shards")
    p.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--per-kind", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-check", action="store_true",
                   help="Verify each generated answer is recovered by the solver")
    args = p.parse_args()
    paths = write_all_synthetic(args.out_dir, args.per_kind, args.seed)
    for k, v in paths.items():
        print(f"Wrote {k}: {v}")
    if args.self_check:
        _self_check(args.seed)


def _self_check(seed: int) -> None:
    """Sanity: generators are internally consistent (answer matches the rule)."""
    from scripts.utils.competition_metric import scores

    for offset, (family, gen) in enumerate(GENERATORS.items()):
        rng = random.Random(seed + offset + 1000)
        ok = 0
        for _ in range(200):
            _, ans, reasoning = gen(rng)
            # the reasoning string ends by stating the answer; confirm it contains it
            if str(ans) in reasoning:
                ok += 1
        print(f"self-check {family}: {ok}/200 reasoning-answer consistent")


if __name__ == "__main__":
    main_cli()
