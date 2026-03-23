#!/usr/bin/env python3
"""Phase 4: Evaluate base+LoRA with vLLM using competition-style decoding."""

from __future__ import annotations

import argparse
import json
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

MODEL_ID = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"


def _classify_local(prompt: str) -> str:
    import re

    if not isinstance(prompt, str):
        return "other"
    t = prompt.lower()
    if re.search(r"\b(bit|binary|8-?bit|nibble|xor|rotate|complement)\b", t):
        return "bit_manipulation"
    if re.search(
        r"\b(cipher|encrypt|decrypt|caesar|vigen|substitution|substitute)\b",
        t,
    ):
        return "encryption"
    if re.search(
        r"\b(equation|polynomial|modulo|algebra|f\(|function|digit sum)\b",
        t,
    ):
        return "algebraic"
    if re.search(r"\b(sequence|term|next in|arithmetic|geometric|fibonacci)\b", t):
        return "sequence"
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
    n_val = max(1, int(len(df) * args.val_fraction))
    val_df = df.head(n_val)
    if args.max_samples > 0:
        val_df = val_df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompts = [
        build_prompt(tokenizer, str(r.prompt), DEFAULT_SYSTEM_PROMPT)
        for r in val_df.itertuples(index=False)
    ]

    llm = LLM(
        model=MODEL_ID,
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


if __name__ == "__main__":
    main()
