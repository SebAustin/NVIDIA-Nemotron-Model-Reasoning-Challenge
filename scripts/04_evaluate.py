#!/usr/bin/env python3
"""Phase 4: Local eval mirroring the competition metric exactly.

Holds out 10% of train per category, loads base + LoRA adapter in vLLM with the
competition decoding (greedy: temperature=0.0, top_p=1.0, max_tokens=7680,
max_model_len=8192, max_lora_rank=32), builds prompts with the EXACT eval template
(no system prompt, boxed-answer suffix, enable_thinking=True), extracts answers
with the same chain the metric uses (boxed -> heuristic -> last numeric), scores
exact-string OR relative-tol 1e-2, and reports overall + per-category accuracy plus
a CSV of every miss (id, category, gold, pred, raw_output).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.utils.answer_extractor import answers_match, extract_final_answer
from scripts.utils.data_formatter import build_user_content
from scripts.utils.solvers import FAMILIES, classify_family

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def stratified_holdout(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    parts = []
    for cat, sub in df.groupby("category"):
        sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = max(1, int(len(sub) * frac))
        parts.append(sub.head(n))
    return pd.concat(parts).reset_index(drop=True)


def build_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": build_user_content(user_prompt)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM eval with LoRA (mirrors metric)")
    ap.add_argument("--adapter-path", type=Path, default=Path("lora_adapter"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--train-categorized", type=Path, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all holdout")
    ap.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    ap.add_argument("--base-only", action="store_true",
                    help="Evaluate the base model with NO adapter (baseline).")
    args = ap.parse_args()

    src = args.train_categorized or (args.data_dir / "train_categorized.csv")
    if not src.is_file():
        src = args.data_dir / "train.csv"
    if not src.is_file():
        raise SystemExit(f"No training CSV at {src}")
    df = pd.read_csv(src)
    for col in ("prompt", "answer"):
        if col not in df.columns:
            raise SystemExit(f"CSV must include column '{col}'")
    if "category" not in df.columns:
        df["category"] = df["prompt"].map(classify_family)

    val = stratified_holdout(df, args.holdout_frac, args.seed)
    if args.max_samples > 0:
        val = val.head(args.max_samples)
    print(f"Holdout: {len(val)} rows across {val['category'].nunique()} categories")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as e:
        raise SystemExit(
            "vLLM/transformers not installed. Use an inference env "
            "(requirements-vllm.txt)."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, str(r.prompt)) for r in val.itertuples(index=False)]

    llm = LLM(
        model=MODEL_ID,
        enable_lora=not args.base_only,
        max_lora_rank=32,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
    gen_kwargs = {}
    if not args.base_only:
        gen_kwargs["lora_request"] = LoRARequest(
            "reasoning_adapter", 1, str(args.adapter_path.resolve())
        )
    outputs = llm.generate(prompts, sampling, **gen_kwargs)

    correct = 0
    by_cat: dict = defaultdict(list)
    misses = []
    for row, out in zip(val.itertuples(index=False), outputs):
        text = out.outputs[0].text
        pred = extract_final_answer(text)
        gold = str(getattr(row, "answer")).strip()
        ok = pred is not None and answers_match(gold, str(pred))
        correct += int(ok)
        cat = str(getattr(row, "category", "other"))
        by_cat[cat].append(ok)
        if not ok:
            misses.append({
                "id": str(getattr(row, "id", "")),
                "category": cat,
                "gold": gold,
                "pred": pred,
                "raw_output": text[:4000],
            })

    total = len(val)
    print(f"\n=== Overall accuracy: {correct}/{total} = {correct/total:.4f} ===")
    print("Per-category:")
    for cat in FAMILIES + ["other"]:
        flags = by_cat.get(cat)
        if not flags:
            continue
        c = sum(flags)
        print(f"  {cat:18s} {c:4d}/{len(flags):<4d} = {c/len(flags):.4f}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    tag = "base" if args.base_only else "adapter"
    miss_path = args.report_dir / f"eval_misses_{tag}.csv"
    with miss_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "gold", "pred", "raw_output"])
        w.writeheader()
        w.writerows(misses)
    print(f"\nWrote {len(misses)} misses -> {miss_path}")


if __name__ == "__main__":
    main()
