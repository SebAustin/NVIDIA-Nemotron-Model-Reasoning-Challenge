"""
Generate synthetic reasoning puzzles: bit manipulation, text cipher, algebraic, sequence.
Each example: 5-7 input->output pairs + test case, prompt text, ground truth, template-based CoT.
Output format: same messages structure as training (system, user, assistant with \\boxed{}).
"""
import json
import os
import random
from typing import Any

from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT


def _ensure_8bit(s: str) -> str:
    return s.zfill(8) if len(s) <= 8 else s[-8:]


def _bit_rotate_left(bits: str, n: int) -> str:
    b = _ensure_8bit(bits)
    n = n % 8
    return b[n:] + b[:n]


def _bit_rotate_right(bits: str, n: int) -> str:
    b = _ensure_8bit(bits)
    n = n % 8
    return b[-n:] + b[:-n]


def _bit_reverse(bits: str) -> str:
    return _ensure_8bit(bits)[::-1]


def _bit_xor_mask(bits: str, mask: int) -> str:
    b = int(_ensure_8bit(bits), 2)
    return format(b ^ mask & 0xFF, "08b")


def _bit_swap_nibbles(bits: str) -> str:
    b = _ensure_8bit(bits)
    return b[4:] + b[:4]


def _bit_complement(bits: str) -> str:
    b = int(_ensure_8bit(bits), 2)
    return format(~b & 0xFF, "08b")


def generate_bit_manipulation_examples(
    rule_name: str,
    apply_rule: Any,
    num_examples: int = 6,
    seed: int | None = None,
) -> list[dict]:
    """Generate bit manipulation puzzle examples. apply_rule(bits_str) -> output_str."""
    rng = random.Random(seed)
    examples = []
    seen = set()
    for _ in range(num_examples * 3):
        if len(examples) >= num_examples:
            break
        inp = format(rng.randint(0, 255), "08b")
        if inp in seen:
            continue
        seen.add(inp)
        out = apply_rule(inp)
        examples.append({"input": inp, "output": out})
    test_inp = format(rng.randint(0, 255), "08b")
    while test_inp in seen:
        test_inp = format(rng.randint(0, 255), "08b")
    test_out = apply_rule(test_inp)
    return [
        {"examples": examples, "test_input": test_inp, "test_output": test_out, "rule_name": rule_name}
    ]


def format_bit_prompt(data: dict) -> str:
    lines = [
        "In this puzzle, a secret bit manipulation rule transforms 8-bit binary inputs into outputs.",
        "Here are some input-output pairs:",
        "",
    ]
    for ex in data["examples"]:
        lines.append(f"  Input:  {ex['input']}  ->  Output: {ex['output']}")
    lines.extend([
        "",
        "Apply the same rule to the following test input and give the output.",
        f"Test input: {data['test_input']}",
        "",
        "Provide your final answer inside \\boxed{}.",
    ])
    return "\n".join(lines)


def cot_template_bit(data: dict, rule_description: str) -> str:
    lines = [
        "Looking at the input-output pairs:",
    ]
    for ex in data["examples"]:
        lines.append(f"- {ex['input']} -> {ex['output']}")
    lines.extend([
        "",
        rule_description,
        "",
        "Verifying with the examples: they all match.",
        f"Applying to test input {data['test_input']}:",
        "",
        f"The answer is \\boxed{{{data['test_output']}}}",
    ])
    return "\n".join(lines)


def generate_bit_puzzles(count: int, seed: int = 42) -> list[dict]:
    """Generate count synthetic bit manipulation puzzles (varied rules)."""
    rng = random.Random(seed)
    rules = [
        ("rotate_left_1", lambda b: _bit_rotate_left(b, 1), "The rule is: rotate left by 1 bit."),
        ("rotate_left_2", lambda b: _bit_rotate_left(b, 2), "The rule is: rotate left by 2 bits."),
        ("rotate_right_1", lambda b: _bit_rotate_right(b, 1), "The rule is: rotate right by 1 bit."),
        ("reverse", lambda b: _bit_reverse(b), "The rule is: reverse the bit order."),
        ("swap_nibbles", lambda b: _bit_swap_nibbles(b), "The rule is: swap the two nibbles."),
        ("complement", lambda b: _bit_complement(b), "The rule is: bitwise complement (flip all bits)."),
    ]
    out = []
    for i in range(count):
        name, func, desc = rng.choice(rules)
        data_list = generate_bit_manipulation_examples(name, func, num_examples=6, seed=seed + i)
        data = data_list[0]
        prompt = format_bit_prompt(data)
        cot = cot_template_bit(data, desc)
        out.append({
            "prompt": prompt,
            "answer": data["test_output"],
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": cot},
            ],
        })
    return out


# --- Text cipher ---
def caesar_shift(text: str, shift: int) -> str:
    result = []
    for c in text:
        if "A" <= c <= "Z":
            result.append(chr((ord(c) - ord("A") + shift) % 26 + ord("A")))
        elif "a" <= c <= "z":
            result.append(chr((ord(c) - ord("a") + shift) % 26 + ord("a")))
        else:
            result.append(c)
    return "".join(result)


def generate_cipher_examples(
    encode_fn: Any,
    num_examples: int,
    seed: int,
    alphabet: str = "letters",
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    examples = []
    words = ["cat", "dog", "hello", "world", "test", "code", "alpha", "beta", "gamma", "delta"]
    for _ in range(num_examples):
        w = rng.choice(words)
        if alphabet == "letters":
            w = "".join(c for c in w if c.isalpha())
        enc = encode_fn(w)
        examples.append((w, enc))
    return examples


def format_cipher_prompt(examples: list[tuple[str, str]], test_input: str, test_output: str) -> str:
    lines = [
        "In this puzzle, a transformation rule maps each input string to an output string.",
        "Here are some input-output pairs:",
        "",
    ]
    for i, o in examples:
        lines.append(f'  Input: "{i}"  ->  Output: "{o}"')
    lines.extend([
        "",
        "Apply the same rule to the following test input and give the output.",
        f'Test input: "{test_input}"',
        "",
        "Provide your final answer inside \\boxed{}.",
    ])
    return "\n".join(lines)


def generate_text_cipher_puzzles(count: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    words = ["cat", "dog", "hello", "world", "test", "code", "alpha", "beta"]
    for i in range(count):
        shift = rng.randint(1, 25)
        examples = [(w, caesar_shift(w, shift)) for w in rng.sample(words, min(6, len(words)))]
        test_in = rng.choice(["xyz", "abc", "key", "map", "run"])
        test_out = caesar_shift(test_in, shift)
        prompt = format_cipher_prompt(examples, test_in, test_out)
        cot = (
            "Observing the pairs, each letter is shifted by a fixed amount in the alphabet. "
            f"Checking: the shift is {shift}. "
            f'Applying to "{test_in}" gives "{test_out}". '
            f"The answer is \\boxed{{{test_out}}}"
        )
        out.append({
            "prompt": prompt,
            "answer": test_out,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": cot},
            ],
        })
    return out


# --- Algebraic / numeric ---
def generate_algebraic_puzzles(count: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        # f(x) = (x mod 7) + 3
        mod_val = rng.randint(5, 15)
        add_val = rng.randint(1, 10)
        examples = []
        used = set()
        for _ in range(6):
            x = rng.randint(0, 100)
            if x in used:
                continue
            used.add(x)
            y = (x % mod_val) + add_val
            examples.append((x, y))
        test_x = rng.randint(0, 100)
        test_y = (test_x % mod_val) + add_val
        lines = [
            "A function f maps each input number to an output. Here are examples:",
            "",
        ]
        for x, y in examples:
            lines.append(f"  f({x}) = {y}")
        lines.extend([
            "",
            f"What is f({test_x})?",
            "",
            "Put your final answer inside \\boxed{}.",
        ])
        prompt = "\n".join(lines)
        cot = (
            f"The pattern is f(x) = (x mod {mod_val}) + {add_val}. "
            f"So f({test_x}) = ({test_x} mod {mod_val}) + {add_val} = {test_y}. "
            f"The answer is \\boxed{{{test_y}}}"
        )
        out.append({
            "prompt": prompt,
            "answer": str(test_y),
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": cot},
            ],
        })
    return out


# --- Sequence ---
def generate_sequence_puzzles(count: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        # Arithmetic: a, a+d, a+2d, ...
        a = rng.randint(1, 5)
        d = rng.randint(2, 7)
        seq = [a + k * d for k in range(7)]
        next_val = a + 7 * d
        lines = [
            "Consider the following sequence. What is the next term?",
            "",
            "Sequence: " + ", ".join(map(str, seq)),
            "",
            "Put your final answer inside \\boxed{}.",
        ]
        prompt = "\n".join(lines)
        cot = (
            f"The sequence is arithmetic: first term {a}, common difference {d}. "
            f"Next term = {seq[-1]} + {d} = {next_val}. "
            f"The answer is \\boxed{{{next_val}}}"
        )
        out.append({
            "prompt": prompt,
            "answer": str(next_val),
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": cot},
            ],
        })
    return out


def write_synthetic_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            # Write only the format expected by SFT: messages
            obj = {"messages": r["messages"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_all_synthetic(
    data_dir: str,
    bit_count: int = 200,
    cipher_count: int = 200,
    algebraic_count: int = 200,
    sequence_count: int = 200,
    seed: int = 42,
) -> None:
    """Generate all synthetic puzzle types and write to data/synthetic/*.jsonl."""
    synthetic_dir = os.path.join(data_dir, "synthetic")
    write_synthetic_jsonl(
        generate_bit_puzzles(bit_count, seed),
        os.path.join(synthetic_dir, "bit_manipulation.jsonl"),
    )
    write_synthetic_jsonl(
        generate_text_cipher_puzzles(cipher_count, seed),
        os.path.join(synthetic_dir, "text_cipher.jsonl"),
    )
    write_synthetic_jsonl(
        generate_algebraic_puzzles(algebraic_count, seed),
        os.path.join(synthetic_dir, "algebraic.jsonl"),
    )
    write_synthetic_jsonl(
        generate_sequence_puzzles(sequence_count, seed),
        os.path.join(synthetic_dir, "sequence.jsonl"),
    )
