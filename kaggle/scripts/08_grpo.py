#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) — the primary score booster.

Training strategy: 0.59 → 0.88+ by using ground-truth answer correctness
as a reward signal instead of imitating teacher CoTs.

Why GRPO beats SFT here:
  - Train.csv has verified ground-truth answers → free reward signal
  - Model explores diverse reasoning paths, keeps what works
  - No teacher API cost, no distribution mismatch from imitation learning
  - Directly optimizes the evaluation metric (answer correctness)

Two reward functions:
  1. Correctness:  +1.0 correct  |  +0.1 has \\boxed{} but wrong  |  -0.5 no \\boxed{}
  2. Format:       +0.2 ends with }  |  +0.1 has \\boxed{}  |  -0.2 neither

Usage on A100 (40 GB, ~3-4 hr):
    python scripts/08_grpo.py \\
        --train-csv data/train.csv \\
        --sft-adapter lora_adapter_sft \\
        --base-model <local-model-path> \\
        --output-dir lora_adapter_grpo \\
        --epochs 2 \\
        --limit 500

Usage on A100 (80 GB, fewer constraints):
    python scripts/08_grpo.py \\
        --train-csv data/train.csv \\
        --sft-adapter lora_adapter_sft \\
        --base-model <path> \\
        --output-dir lora_adapter_grpo \\
        --epochs 3
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pandas as pd
from datasets import Dataset

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

KAGGLE_LORA_TARGET_REGEX = (
    r".*\.(in_proj|out_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_correctness(
    completions: List[str],
    answer: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """Primary reward: correct boxed answer = +1.0, wrong boxed = +0.1, no boxed = -0.5."""
    golds: List[str] = answer if answer is not None else kwargs.get("answer", [])
    rewards: List[float] = []
    for completion, gold in zip(completions, golds):
        pred = extract_boxed_answer(completion)
        if pred is not None and answers_match(str(gold).strip(), pred):
            rewards.append(1.0)
        elif pred is not None:
            rewards.append(0.1)  # format half-credit
        else:
            rewards.append(-0.5)
    return rewards


def reward_format(completions: List[str], **kwargs) -> List[float]:
    """Format compliance: reward responses that end with \\boxed{...}."""
    rewards: List[float] = []
    for c in completions:
        stripped = c.rstrip()
        if "\\boxed{" in stripped and stripped.endswith("}"):
            rewards.append(0.2)
        elif "\\boxed{" in stripped:
            rewards.append(0.05)
        else:
            rewards.append(-0.2)
    return rewards


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cuda_vram_gb(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)


def _build_grpo_prompt(tokenizer, user_prompt: str) -> str:
    """Format prompt with chat template + add_generation_prompt for GRPO rollout generation."""
    msgs = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _sort_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Curriculum: start with shorter/simpler prompts (easier to get reward signal early)."""
    df = df.copy()
    df["_plen"] = df["prompt"].str.len()
    df = df.sort_values("_plen").drop(columns=["_plen"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str, vram_gb: float):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    use_quant = vram_gb < 65
    if use_quant:
        # Skip Mamba-specific modules from 4-bit (Triton kernels need raw weights)
        skip = ["lm_head", "in_proj", "out_proj", "conv1d", "x_proj", "dt_proj"]
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        print(f"[08_grpo] Loaded in 4-bit NF4 (VRAM={vram_gb:.0f}GB)", flush=True)
    else:
        loaded = False
        for attn in ("flash_attention_2", None):
            try:
                kw: dict = dict(
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                if attn:
                    kw["attn_implementation"] = attn
                model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
                loaded = True
                print(f"[08_grpo] Loaded bf16 + attn={attn} (VRAM={vram_gb:.0f}GB)", flush=True)
                break
            except Exception as e:
                if attn is None:
                    raise
                print(f"[08_grpo] {attn} unavailable ({e!r}), retrying without.", flush=True)
        if not loaded:
            raise RuntimeError("Model load failed")

    return model


def _attach_lora(
    model,
    sft_adapter: Optional[Path],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if sft_adapter and sft_adapter.is_dir():
        print(f"[08_grpo] Warm-starting from SFT adapter: {sft_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(sft_adapter), is_trainable=True)
    else:
        lc_kw: dict = dict(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=KAGGLE_LORA_TARGET_REGEX,
        )
        sig = inspect.signature(LoraConfig.__init__).parameters
        if "use_rslora" in sig:
            lc_kw["use_rslora"] = True
        model = get_peft_model(model, LoraConfig(**lc_kw))
        print("[08_grpo] Fresh LoRA attached (no SFT adapter specified)", flush=True)

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# GRPOConfig helper (handles TRL version differences)
# ---------------------------------------------------------------------------

def _build_grpo_config(args, num_gen: int):
    from trl import GRPOConfig

    sig = inspect.signature(GRPOConfig.__init__).parameters

    kw: dict = dict(
        output_dir=str(args.checkpoint_dir),
        num_train_epochs=max(1, int(args.epochs)),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=0.5,
    )

    # num_generations: TRL >=0.12 name; some older versions use num_return_sequences
    for param_name in ("num_generations", "num_return_sequences"):
        if param_name in sig:
            kw[param_name] = num_gen
            break

    # max tokens for rollouts
    for param_name in ("max_new_tokens", "max_completion_length"):
        if param_name in sig:
            kw[param_name] = args.max_new_tokens
            break

    # generation temperature
    if "temperature" in sig:
        kw["temperature"] = args.temperature

    # max_prompt_length (prevents OOM on very long prompts)
    if "max_prompt_length" in sig:
        kw["max_prompt_length"] = args.max_prompt_len

    return GRPOConfig(**kw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="GRPO training for Nemotron reasoning (score booster)")
    p.add_argument("--train-csv", type=Path, default=Path("data/train.csv"),
                   help="Competition train.csv with 'prompt' and 'answer' columns.")
    p.add_argument("--sft-adapter", type=Path, default=None,
                   help="Path to SFT LoRA adapter for warm-start (recommended). "
                        "If None, fresh LoRA is attached to base model.")
    p.add_argument("--base-model", type=str, default=MODEL_ID,
                   help="HF model ID or local path to base model weights.")
    p.add_argument("--output-dir", type=Path, default=Path("lora_adapter_grpo"),
                   help="Where to save the final GRPO adapter.")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("grpo_output"),
                   help="Trainer checkpoint directory (intermediate checkpoints).")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank (max 32 per competition rules).")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=2.0,
                   help="GRPO training epochs (2-3 is typical; 1 already gives big lift).")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="GRPO learning rate (lower than SFT; 5e-5 is a safe starting point).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum).")
    p.add_argument("--num-generations", type=int, default=0,
                   help="Rollouts per prompt during GRPO (0=auto: 4 on 40GB, 8 on 80GB). "
                        "More rollouts → better reward signal but slower training.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens per rollout completion (512 is fast; 1024 if accuracy plateaus).")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for rollouts. Must be >0 for GRPO to work. "
                        "0.7-0.9 is typical; higher = more diversity = better early training.")
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="Max training rows (0=all). Start with 500 to test; "
                        "all rows if session time allows.")
    p.add_argument("--curriculum", action="store_true", default=True,
                   help="Sort problems by prompt length (easier first = faster reward signal early).")
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    args = p.parse_args()

    vram = _cuda_vram_gb()
    print(f"[08_grpo] GPU VRAM: {vram:.1f} GB", flush=True)
    print(f"[08_grpo] torch={torch.__version__}", flush=True)

    num_gen = args.num_generations
    if num_gen <= 0:
        num_gen = 8 if vram >= 65 else 4 if vram >= 30 else 2
    print(f"[08_grpo] num_generations={num_gen} | max_new_tokens={args.max_new_tokens}", flush=True)

    # Load competition data
    if not args.train_csv.is_file():
        raise SystemExit(f"Missing {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    if args.curriculum:
        df = _sort_by_difficulty(df)
        print("[08_grpo] Curriculum: sorted by prompt length (short first)", flush=True)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"[08_grpo] Training on {len(df)} problems", flush=True)

    # Tokenizer (load before model to format prompts early)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build GRPO dataset: 'prompt' (formatted) + 'answer' (for reward fn)
    records = []
    for _, row in df.iterrows():
        formatted_prompt = _build_grpo_prompt(tokenizer, str(row["prompt"]))
        records.append({
            "prompt": formatted_prompt,
            "answer": str(row["answer"]).strip(),
        })
    dataset = Dataset.from_list(records)
    print(f"[08_grpo] Dataset: {len(dataset)} rows", flush=True)

    # Load base model
    model = _load_model(args.base_model, vram)

    # Attach LoRA (fresh or from SFT warm-start)
    model = _attach_lora(model, args.sft_adapter, args.lora_r, args.lora_alpha, args.lora_dropout)

    # Apply Nemotron MoE dtype patch if available
    try:
        # The patch is defined in 03_train_lora.py — try to reuse it
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_train_lora_mod",
            str(_ROOT / "scripts" / "03_train_lora.py"),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "_patch_moe_dtype_mismatch"):
                mod._patch_moe_dtype_mismatch()
                print("[08_grpo] Applied Nemotron MoE dtype patch", flush=True)
    except Exception:
        pass  # Not critical — training will raise if the bug is triggered

    # Build GRPO config
    config = _build_grpo_config(args, num_gen)

    # Build GRPOTrainer — handle reward_funcs vs reward_fn API differences across TRL versions
    from trl import GRPOTrainer
    trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters

    trainer_kw: dict = dict(
        model=model,
        config=config,
        train_dataset=dataset,
    )

    if "reward_funcs" in trainer_sig:
        # TRL >=0.12 (current standard)
        trainer_kw["reward_funcs"] = [reward_correctness, reward_format]
    elif "reward_fn" in trainer_sig:
        # Older TRL API — single function; merge both rewards
        def _combined_reward(completions: List[str], **kwargs) -> List[float]:
            c = reward_correctness(completions, **kwargs)
            f = reward_format(completions, **kwargs)
            return [a + b for a, b in zip(c, f)]
        trainer_kw["reward_fn"] = _combined_reward

    if "processing_class" in trainer_sig:
        trainer_kw["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kw["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kw)

    print(
        f"[08_grpo] Starting GRPO — {len(df)} prompts × {args.epochs:.0f} epochs × "
        f"{num_gen} rollouts × {args.max_new_tokens} max tokens/rollout.",
        flush=True,
    )
    trainer.train()

    # Save adapter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Fix adapter_config base model path (required for Kaggle evaluation)
    cfg_path = args.output_dir / "adapter_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["base_model_name_or_path"] = MODEL_ID
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[08_grpo] Saved GRPO adapter → {args.output_dir}", flush=True)
    print(
        "[08_grpo] Next: run 05_package_submission.py --adapter-dir lora_adapter_grpo "
        "--output submission_grpo.zip  then submit to Kaggle.",
        flush=True,
    )


if __name__ == "__main__":
    main()
