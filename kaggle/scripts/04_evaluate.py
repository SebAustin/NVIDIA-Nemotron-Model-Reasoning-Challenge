#!/usr/bin/env python3
"""Phase 4: Evaluate base+LoRA with vLLM using competition-style decoding."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


_CATEGORY_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = []


def _compile_category_patterns() -> None:
    """Build the prompt -> category classifier once.

    Categories match the 9 actual test families confirmed against the
    Progress-Prize winning reference (tonghuikang/nemotron). Order matters: the
    first match wins, so the more specific patterns (cryptarithm, equation
    numeric) are listed before broader ones.
    """
    import re as _re

    if _CATEGORY_PATTERNS:
        return
    specs: list[tuple[str, str]] = [
        # cryptarithm: digit-letter substitution puzzles like SEND + MORE = MONEY
        (
            "cryptarithm",
            r"\b(cryptarithm|cryptarithmetic|alphametic|letters\s+represent\s+digits"
            r"|each\s+letter\s+represents\s+(a\s+)?digit)\b",
        ),
        # equation_numeric: solve-for-x / find-the-number numeric equations
        (
            "equation_numeric",
            r"\b(solve\s+for|find\s+the\s+value\s+of|what\s+is\s+the\s+value\s+of)"
            r"|\b(equation|simultaneous|system\s+of\s+equations)\b",
        ),
        # gravity: falling object / projectile / physics gravity problems
        (
            "gravity",
            r"\b(gravity|free\s*fall|falling|dropped|projectile|gravitational"
            r"|9\.81|9\.8\s*m/s|m/s\^?2|m\s*s\^?-?2)\b",
        ),
        # unit_conversion: convert X units to Y units
        (
            "unit_conversion",
            r"\b(convert|conversion)\b.*\b(to|into)\b"
            r"|\b(km|kilometer|meter|mile|inch|foot|feet|yard|gram|kilogram|kg|pound|lb"
            r"|ounce|liter|gallon|fahrenheit|celsius|kelvin|second|minute|hour)\b.*\b(to|into|in)\b",
        ),
        # numeral: base/radix conversion (binary -> hex -> roman etc.)
        (
            "numeral",
            r"\b(roman\s+numeral|base[-\s]?\d+|hexadecimal|octal"
            r"|convert\s+\d+\s+from\s+base|in\s+base\s+\d+)\b",
        ),
        # cipher: text encryption (Caesar, Vigenere, substitution, etc.)
        (
            "cipher",
            r"\b(cipher|encrypt|decrypt|caesar|vigen|substitution\s+cipher"
            r"|atbash|affine\s+cipher|encoded\s+message)\b",
        ),
        # bit_manipulation: binary bit operations
        (
            "bit_manipulation",
            r"\b(bit\s+manipulation|bit\s+shift|bit-?wise|xor|rotation|complement"
            r"|8-?bit\s+binary|16-?bit\s+binary)\b"
            r"|\b[01]{8,}\b",
        ),
    ]
    for name, pattern in specs:
        _CATEGORY_PATTERNS.append((name, _re.compile(pattern, _re.IGNORECASE)))


def _classify_local(prompt: str) -> str:
    """Map a puzzle prompt to one of the 9 test categories (or 'other')."""
    if not isinstance(prompt, str):
        return "other"
    _compile_category_patterns()
    for name, pattern in _CATEGORY_PATTERNS:
        if pattern.search(prompt):
            return name
    return "other"


def build_prompt(tokenizer, user_prompt: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM eval with LoRA")
    parser.add_argument("--adapter-path", type=Path, default=Path("lora_adapter"))
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_ID,
        help=(
            "Base-model path or HF id. On Kaggle, pass the locally-attached "
            "Kaggle Model directory (e.g. /kaggle/input/nemotron-3-nano-30b-a3b-bf16/...) "
            "so vLLM does not hit the network."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-categorized",
        type=Path,
        default=None,
        help="Default: <data-dir>/train_categorized.csv",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=0,
        help=(
            "If > 0, sample N rows from each classified puzzle family (Phase 0.3 "
            "diagnostic). Overrides --max-samples / --val-fraction. Writes a "
            "per-family breakdown CSV to <report-dir>/eval_by_family.csv."
        ),
    )
    parser.add_argument(
        "--per-family-csv",
        type=Path,
        default=None,
        help="Override path for the per-family breakdown CSV.",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--lora-name", type=str, default="reasoning_adapter")
    parser.add_argument("--lora-id", type=int, default=1)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as e:
        raise SystemExit(
            "vLLM is not installed. Use requirements-vllm.txt in an inference env."
        ) from e

    train_path = args.train_categorized or (args.data_dir / "train_categorized.csv")
    if not train_path.is_file():
        train_path = args.data_dir / "train.csv"
    if not train_path.is_file():
        raise SystemExit(f"No training CSV at {train_path}")
    df = pd.read_csv(train_path)
    if "prompt" not in df.columns or "answer" not in df.columns:
        raise SystemExit("CSV must include columns 'prompt' and 'answer'")
    if "puzzle_type" not in df.columns:
        df["puzzle_type"] = df["prompt"].map(_classify_local)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.samples_per_family > 0:
        # Phase 0.3 diagnostic mode: equal-size strata per family.
        parts: list[pd.DataFrame] = []
        for fam in sorted(df["puzzle_type"].unique()):
            sub = df[df["puzzle_type"] == fam].head(args.samples_per_family)
            parts.append(sub)
        val_df = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    else:
        n_val = max(1, int(len(df) * args.val_fraction))
        val_df = df.head(n_val)
        if args.max_samples > 0:
            val_df = val_df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        build_prompt(tokenizer, str(r.prompt), DEFAULT_SYSTEM_PROMPT)
        for r in val_df.itertuples(index=False)
    ]

    llm = LLM(
        model=args.model_path,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=64,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
    lora_req = LoRARequest(args.lora_name, args.lora_id, str(args.adapter_path.resolve()))

    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    correct = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    errors: list[dict] = []

    for row, out in tqdm(
        zip(val_df.itertuples(index=False), outputs),
        total=len(val_df),
        desc="Eval",
    ):
        text = out.outputs[0].text
        pred = extract_boxed_answer(text)
        gold = str(getattr(row, "answer")).strip()
        ok = pred is not None and answers_match(gold, pred)
        if ok:
            correct += 1
        ptype = getattr(row, "puzzle_type", "other")
        by_type[str(ptype)].append(ok)
        if not ok:
            errors.append(
                {
                    "id": str(getattr(row, "id", "")),
                    "puzzle_type": str(ptype),
                    "gold": gold,
                    "pred": pred,
                    "completion_preview": text[:2000],
                }
            )

    total = len(val_df)
    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    for ptype, flags in sorted(by_type.items()):
        c = sum(1 for x in flags if x)
        print(f"  {ptype}: {c}/{len(flags)} = {c/len(flags):.4f}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    err_path = args.report_dir / "eval_errors.jsonl"
    with err_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote error cases: {err_path}")

    csv_path = args.per_family_csv or (args.report_dir / "eval_by_family.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["puzzle_type", "correct", "total", "accuracy"])
        for ptype in sorted(by_type):
            flags = by_type[ptype]
            c = sum(1 for x in flags if x)
            n = len(flags)
            writer.writerow([ptype, c, n, f"{c/n:.4f}" if n else "0.0000"])
        writer.writerow(
            ["__overall__", correct, total, f"{acc:.4f}" if total else "0.0000"],
        )
    print(f"Wrote per-family breakdown: {csv_path}")


if __name__ == "__main__":
    main()
