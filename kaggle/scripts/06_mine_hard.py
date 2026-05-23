#!/usr/bin/env python3
"""Phase 6: Hard-case mining (rejection-sampling fine-tuning).

After an initial adapter is trained, this script:
1. Runs the adapter against held-out or train-shuffled prompts.
2. Collects rows where the predicted boxed answer is wrong (or missing).
3. Calls a teacher LLM (OpenAI or Anthropic) to produce a VERIFIED CoT for each
   failing row (retries with temperature bump until extracted answer matches gold).
4. Appends the new verified rows to `--append-to` (default data/train_sft.jsonl).

Re-run 03_train_lora.py with the augmented dataset for a second SFT epoch round.
Repeat until accuracy plateaus. Standard RFT recipe. +2-6pp typical.

Requires a teacher API key in OPENAI_API_KEY or ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import PeftModel

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.cot_generator import generate_cot_with_verification
from scripts.utils.data_formatter import (
    DEFAULT_SYSTEM_PROMPT,
    build_messages,
    format_assistant_reply_verified,
)
from scripts.utils.puzzle_generator import _extract_examples_from_prompt

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejection-sampling fine-tune augmentation")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=MODEL_ID)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--append-to", type=Path, default=Path("data/train_sft.jsonl"),
        help="Existing SFT JSONL; new hard-case rows are appended.",
    )
    parser.add_argument("--max-samples", type=int, default=500, help="How many prompts to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--cot-backend", choices=("anthropic", "openai"), default="openai")
    parser.add_argument("--cot-model", type=str, default="gpt-4o")
    parser.add_argument("--cot-max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: Dict[str, Any] = dict(trust_remote_code=True, device_map="auto")
    if args.quantize:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kw["torch_dtype"] = torch.bfloat16

    print(f"Loading base {args.base_model} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kw)
    model = PeftModel.from_pretrained(base, str(args.adapter_path))
    model.eval()

    hard_rows: List[Dict[str, Any]] = []
    n_correct = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mining"):
        prompt_text = str(row["prompt"])
        gold = str(row["answer"]).strip()
        chat = build_chat_prompt(tokenizer, prompt_text)
        ids = tokenizer(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed_answer(gen)
        if pred is not None and answers_match(gold, pred):
            n_correct += 1
            continue
        hard_rows.append({"prompt": prompt_text, "gold": gold, "adapter_pred": pred, "id": row.get("id", "")})

    total = len(df)
    acc = n_correct / total if total else 0.0
    print(f"\nAdapter accuracy on sampled rows: {acc:.3f} ({n_correct}/{total})")
    print(f"Hard cases to verify with teacher: {len(hard_rows)}")

    env_key = "OPENAI_API_KEY" if args.cot_backend == "openai" else "ANTHROPIC_API_KEY"
    if hard_rows and not os.environ.get(env_key):
        raise SystemExit(
            f"Missing {env_key} — cannot call teacher for CoT. Re-run with the key set."
        )

    appended = 0
    with args.append_to.open("a", encoding="utf-8") as f:
        for r in tqdm(hard_rows, desc="Teacher CoT"):
            try:
                res = generate_cot_with_verification(
                    r["prompt"], r["gold"],
                    backend=args.cot_backend, model=args.cot_model,
                    max_tokens=args.cot_max_tokens,
                )
            except Exception as e:
                print(f"teacher fail id={r['id']}: {e!r}")
                continue
            if res.extracted is None or not answers_match(r["gold"], res.extracted):
                continue
            cot_only = res.raw_text
            if "\\boxed" in cot_only:
                cot_only = cot_only.split("\\boxed")[0].rstrip()
            examples = _extract_examples_from_prompt(r["prompt"])
            assistant = format_assistant_reply_verified(cot_only, r["gold"], examples)
            rec = {
                "messages": build_messages(r["prompt"], assistant),
                "meta": {"puzzle_type": "hard_mined", "mined": True, "id": r["id"]},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            appended += 1

    print(f"Appended {appended} hard-case rows to {args.append_to}")
    print("Re-run 03_train_lora.py on the enlarged dataset to close the gap.")


if __name__ == "__main__":
    main()
