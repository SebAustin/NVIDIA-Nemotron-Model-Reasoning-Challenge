"""
Phase 1: EDA and puzzle-type categorization for Nemotron Reasoning Challenge.
Run from project root: python scripts/01_eda.py
"""
import os
import re
import sys

import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "train_categorized.csv")

# Keyword rules for puzzle-type categorization (order matters: first match wins for ties)
PUZZLE_TYPE_RULES = [
    (
        "bit_manipulation",
        re.compile(
            r"\b(bit|binary|8-bit|nibble|XOR|rotate|reverse|shift|complement|mask)\b",
            re.I,
        ),
    ),
    (
        "encryption_cipher",
        re.compile(
            r"\b(cipher|encrypt|decrypt|Caesar|substitution|decode|encode|Vigenère|Vigenere)\b",
            re.I,
        ),
    ),
    (
        "algebraic_numeric",
        re.compile(
            r"\b(function|equation|f\(x\)|mod|digit|sum|polynomial|modular|arithmetic)\b",
            re.I,
        ),
    ),
    (
        "sequence",
        re.compile(
            r"\b(sequence|pattern|next term|series|Fibonacci|arithmetic progression|geometric)\b",
            re.I,
        ),
    ),
]


def categorize_puzzle_type(prompt: str) -> str:
    """Assign one puzzle type based on keyword counts; default 'other'."""
    if not isinstance(prompt, str) or not prompt.strip():
        return "other"
    text = prompt.strip().lower()
    scores = []
    for label, pattern in PUZZLE_TYPE_RULES:
        count = len(pattern.findall(text))
        scores.append((count, label))
    scores.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return scores[0][1] if scores and scores[0][0] > 0 else "other"


def infer_answer_type(answer: str) -> str:
    """Infer answer type: numeric, binary, string, etc."""
    if not isinstance(answer, str):
        return "unknown"
    s = answer.strip()
    if not s:
        return "empty"
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return "numeric"
    if re.match(r"^[01\s]+$", s) and set(s.replace(" ", "")) <= {"0", "1"}:
        return "binary"
    try:
        float(s)
        return "numeric"
    except ValueError:
        pass
    return "string"


def main() -> None:
    os.chdir(PROJECT_ROOT)
    if not os.path.isfile(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found. Download train.csv from Kaggle into data/.")
        sys.exit(1)
    if not os.path.isfile(TEST_PATH):
        print(f"Error: {TEST_PATH} not found. Download test.csv from Kaggle into data/.")
        sys.exit(1)

    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train)} rows. Test: {len(test)} rows.")
    print(f"Train columns: {list(train.columns)}")
    print(f"Test columns: {list(test.columns)}")

    # Categorize
    print("\nCategorizing puzzle types...")
    train["puzzle_type"] = train["prompt"].apply(categorize_puzzle_type)
    test["puzzle_type"] = test["prompt"].apply(categorize_puzzle_type)

    # Tokenizer
    print("Loading Nemotron tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    train["prompt_tokens"] = train["prompt"].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False)) if isinstance(x, str) else 0
    )
    test["prompt_tokens"] = test["prompt"].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False)) if isinstance(x, str) else 0
    )

    # Answer type and length (train only)
    train["answer_type"] = train["answer"].apply(infer_answer_type)
    train["answer_len_chars"] = train["answer"].astype(str).str.len()
    train["answer_len_tokens"] = train["answer"].apply(
        lambda x: len(tokenizer.encode(str(x), add_special_tokens=False))
    )

    # Distributions
    print("\n--- Puzzle type distribution (train) ---")
    print(train["puzzle_type"].value_counts())
    print("\n--- Puzzle type distribution (test) ---")
    print(test["puzzle_type"].value_counts())

    print("\n--- Prompt length (tokens) ---")
    for df, name in [(train, "train"), (test, "test")]:
        t = df["prompt_tokens"]
        print(f"{name}: min={t.min()}, max={t.max()}, median={t.median():.0f}, mean={t.mean():.0f}")
        over = (t >= 8192).sum()
        near = ((t >= 7000) & (t < 8192)).sum()
        print(f"  Over 8192: {over}, in [7000, 8192): {near}")

    print("\n--- Answer type (train) ---")
    print(train["answer_type"].value_counts())
    print("\n--- Answer length (train) ---")
    print(f"Chars: min={train['answer_len_chars'].min()}, max={train['answer_len_chars'].max()}")
    print(f"Tokens: min={train['answer_len_tokens'].min()}, max={train['answer_len_tokens'].max()}")

    # Examples per type
    print("\n--- Example prompts and answers (2 per type) ---")
    for ptype in train["puzzle_type"].unique():
        subset = train[train["puzzle_type"] == ptype].head(2)
        print(f"\n[{ptype}]")
        for _, row in subset.iterrows():
            prompt_preview = (row["prompt"] or "")[:300] + "..." if len(str(row["prompt"] or "")) > 300 else (row["prompt"] or "")
            print(f"  Answer: {row['answer']}")
            print(f"  Prompt: {prompt_preview}\n")

    # Pattern summary
    print("\n--- Pattern summary ---")
    print(
        "Each prompt contains several input→output examples demonstrating a hidden rule, "
        "then asks the model to apply that rule to a new input. The model must place the "
        "final answer inside \\boxed{}."
    )

    # Length vs max_model_len
    system_tokens = 100
    cot_tokens = 2000
    train["est_total_tokens"] = train["prompt_tokens"] + system_tokens + cot_tokens
    over_est = (train["est_total_tokens"] > 8192).sum()
    print(f"\n--- Estimated total (prompt + {system_tokens} system + {cot_tokens} CoT) ---")
    print(f"Examples with est. total > 8192: {over_est} / {len(train)}")

    # Save
    train.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
