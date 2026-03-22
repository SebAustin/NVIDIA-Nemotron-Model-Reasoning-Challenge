"""
Phase 2: Build SFT dataset — CoT generation on train, synthetic puzzles, merge and filter.
Run from project root: python scripts/02_prepare_data.py
"""
import argparse
import json
import os
import re
import sys

# Ensure project root is on path when run from any cwd (e.g. Kaggle, or scripts/ as cwd)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from transformers import AutoTokenizer

from scripts.utils.answer_extractor import extract_boxed_answer
from scripts.utils.cot_generator import build_training_example, generate_cot_with_retries
from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT
from scripts.utils.model_utils import local_load_kwargs, resolve_model_path
from scripts.utils.puzzle_generator import generate_all_synthetic

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_CATEGORIZED_CSV = os.path.join(DATA_DIR, "train_categorized.csv")
TRAIN_SFT_JSONL = os.path.join(DATA_DIR, "train_sft.jsonl")
MAX_TOTAL_TOKENS = 7000


def load_train_source() -> pd.DataFrame:
    """Load train.csv or train_categorized.csv."""
    if os.path.isfile(TRAIN_CATEGORIZED_CSV):
        return pd.read_csv(TRAIN_CATEGORIZED_CSV)
    if os.path.isfile(TRAIN_CSV):
        return pd.read_csv(TRAIN_CSV)
    raise FileNotFoundError(
        f"Neither {TRAIN_CATEGORIZED_CSV} nor {TRAIN_CSV} found. Run 01_eda.py or add train.csv."
    )


def run_cot_generation(
    train_df: pd.DataFrame,
    max_examples: int | None = None,
    skip_api: bool = False,
) -> list[dict]:
    """Generate CoT for each training row; keep only correct answers. Optionally skip API (synthetic-only)."""
    if skip_api:
        return []
    out = []
    n = len(train_df) if max_examples is None else min(max_examples, len(train_df))
    for idx in range(n):
        row = train_df.iloc[idx]
        prompt = row["prompt"]
        answer = str(row["answer"]).strip()
        content, correct = generate_cot_with_retries(prompt, answer)
        if correct and content:
            ex = build_training_example(prompt, answer, content, DEFAULT_SYSTEM_PROMPT)
            out.append(ex)
        if (idx + 1) % 10 == 0:
            print(f"  CoT: {idx + 1}/{n} done, kept {len(out)}")
    return out


def load_synthetic_jsonl(path: str) -> list[dict]:
    """Load a JSONL of {messages: [...]}."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def filter_and_merge(
    records: list[dict],
    tokenizer: AutoTokenizer,
    max_tokens: int = MAX_TOTAL_TOKENS,
) -> list[dict]:
    """Drop duplicates (by normalized user prompt), drop if total tokens > max_tokens, ensure CoT consistent with boxed."""
    seen_prompts = set()
    filtered = []
    for r in records:
        messages = r.get("messages") or r
        if not messages or len(messages) < 3:
            continue
        user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
        norm = re.sub(r"\s+", " ", user_content.strip())
        if norm in seen_prompts:
            continue
        seen_prompts.add(norm)
        assistant_content = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        extracted = extract_boxed_answer(assistant_content)
        if not extracted:
            continue
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        num_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        if num_tokens > max_tokens:
            continue
        filtered.append({"messages": messages})
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT data: CoT + synthetic, then merge.")
    parser.add_argument("--skip-cot", action="store_true", help="Skip API CoT generation (synthetic only)")
    parser.add_argument("--max-cot", type=int, default=None, help="Max training examples to get CoT for")
    parser.add_argument("--synthetic-only", action="store_true", help="Only generate synthetic, no train CoT")
    parser.add_argument("--bit", type=int, default=200, help="Synthetic bit puzzles")
    parser.add_argument("--cipher", type=int, default=200, help="Synthetic cipher puzzles")
    parser.add_argument("--algebraic", type=int, default=200, help="Synthetic algebraic puzzles")
    parser.add_argument("--sequence", type=int, default=200, help="Synthetic sequence puzzles")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    model_path_raw = os.environ.get("NEMOTRON_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    model_path = resolve_model_path(model_path_raw)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **local_load_kwargs(model_path_raw))
    all_records = []

    if not args.synthetic_only:
        train_df = load_train_source()
        print(f"Loaded {len(train_df)} training rows.")
        cot_records = run_cot_generation(
            train_df, max_examples=args.max_cot, skip_api=args.skip_cot
        )
        print(f"CoT records kept: {len(cot_records)}")
        for r in cot_records:
            all_records.append({"messages": r["messages"]})

    print("Generating synthetic puzzles...")
    generate_all_synthetic(
        DATA_DIR,
        bit_count=args.bit,
        cipher_count=args.cipher,
        algebraic_count=args.algebraic,
        sequence_count=args.sequence,
    )
    for name in ["bit_manipulation", "text_cipher", "algebraic", "sequence"]:
        path = os.path.join(DATA_DIR, "synthetic", f"{name}.jsonl")
        syn = load_synthetic_jsonl(path)
        for obj in syn:
            all_records.append(obj)
        print(f"  Loaded {len(syn)} from {name}")

    print("Filtering and merging...")
    merged = filter_and_merge(all_records, tokenizer, max_tokens=MAX_TOTAL_TOKENS)
    print(f"After filter: {len(merged)} examples.")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAIN_SFT_JSONL, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {TRAIN_SFT_JSONL}")


if __name__ == "__main__":
    main()
