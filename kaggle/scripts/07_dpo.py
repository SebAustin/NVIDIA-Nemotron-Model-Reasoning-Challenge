#!/usr/bin/env python3
"""Phase 7: DPO refinement on top of the SFT adapter.

Input JSONL must have one row per preference pair with keys:
  - "prompt": the user prompt (string, after chat-template if you want; or raw user turn)
  - "chosen": teacher-verified assistant reply that ends with \\boxed{gold}
  - "rejected": adapter-generated wrong assistant reply for the same prompt

The script runs TRL DPOTrainer with beta=0.1 defaults; LoRA adapter is attached
to the quantized base so we learn a small preference adjustment on top of SFT.
Produces a refined adapter at --output-dir.

Typical gain on reasoning tasks: +1-3pp over SFT-only, IF the pairs are clean.
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

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _build_chat(tokenizer, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    p = argparse.ArgumentParser(description="DPO refinement")
    p.add_argument("--pairs", type=Path, required=True, help="JSONL of prompt/chosen/rejected rows.")
    p.add_argument("--sft-adapter", type=Path, required=True, help="Trained SFT LoRA adapter to start from.")
    p.add_argument("--base-model", type=str, default=MODEL_ID)
    p.add_argument("--output-dir", type=Path, default=Path("lora_adapter_dpo"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("dpo_output"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6, help="DPO needs a low LR (typical 1e-6..5e-6).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.pairs.is_file():
        raise SystemExit(f"Missing pairs file: {args.pairs}")

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
    base = prepare_model_for_kbit_training(base)
    # Attach SFT adapter as the policy; DPO will further tune it.
    model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
    model.print_trainable_parameters()

    # DPOTrainer will create its own reference model copy internally (frozen).
    ds = load_dataset("json", data_files=str(args.pairs), split="train")

    def _format(row):
        # DPO rows need a fully formed chat prompt; 'chosen'/'rejected' are assistant replies.
        user = row.get("prompt") or row.get("user") or ""
        return {
            "prompt": _build_chat(tokenizer, user),
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
        }

    ds = ds.map(_format, remove_columns=[c for c in ds.column_names if c not in {"prompt", "chosen", "rejected"}])

    cfg = DPOConfig(
        output_dir=str(args.checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,       # DPOTrainer creates frozen ref from current model copy
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved DPO-refined adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
