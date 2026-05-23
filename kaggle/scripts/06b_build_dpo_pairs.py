#!/usr/bin/env python3
"""Build DPO (chosen, rejected) pairs from adapter predictions + teacher CoT.

For every prompt in --train-csv:
  1. adapter generates a greedy answer; extract boxed pred.
  2. teacher (OpenAI/Anthropic) generates a verified CoT whose boxed answer == gold.
  3. If adapter is wrong:
       rejected = adapter full generation
       chosen   = teacher reply (reasoning + boxed gold)
     Write row {"prompt", "chosen", "rejected", "id"} to --output.

Use the produced file with 07_dpo.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    format_assistant_reply_verified,
)
from scripts.utils.puzzle_generator import _extract_examples_from_prompt

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO pairs from adapter failures.")
    ap.add_argument("--adapter-path", type=Path, required=True)
    ap.add_argument("--base-model", type=str, default=MODEL_ID)
    ap.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/dpo_pairs.jsonl"))
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--cot-backend", choices=("anthropic", "openai"), default="openai")
    ap.add_argument("--cot-model", type=str, default="gpt-4o")
    ap.add_argument("--cot-max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env_key = "OPENAI_API_KEY" if args.cot_backend == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(env_key):
        raise SystemExit(f"Missing {env_key}.")

    df = pd.read_csv(args.train_csv).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_path))
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="DPO pairs"):
            prompt_text = str(row["prompt"])
            gold = str(row["answer"]).strip()
            msgs = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_boxed_answer(gen)
            if pred is not None and answers_match(gold, pred):
                continue  # adapter already right -> no preference signal

            try:
                res = generate_cot_with_verification(
                    prompt_text, gold,
                    backend=args.cot_backend, model=args.cot_model,
                    max_tokens=args.cot_max_tokens,
                )
            except Exception as e:
                print(f"teacher fail id={row.get('id','')}: {e!r}")
                continue
            if res.extracted is None or not answers_match(gold, res.extracted):
                continue

            cot_only = res.raw_text
            if "\\boxed" in cot_only:
                cot_only = cot_only.split("\\boxed")[0].rstrip()
            examples = _extract_examples_from_prompt(prompt_text)
            chosen = format_assistant_reply_verified(cot_only, gold, examples)
            rec = {
                "id": str(row.get("id", "")),
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": gen,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} DPO pairs to {args.output}")


if __name__ == "__main__":
    main()
