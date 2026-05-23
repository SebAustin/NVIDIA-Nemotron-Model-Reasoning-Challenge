#!/usr/bin/env python3
"""Phase 3: QLoRA SFT with Unsloth (preferred) or Hugging Face PEFT fallback."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer


def _sft_config_kwargs(max_seq_length: int) -> dict:
    """TRL renames max_seq_length → max_length on SFTConfig in recent versions."""
    sig = inspect.signature(SFTConfig.__init__).parameters
    out: dict = {}
    if "max_seq_length" in sig:
        out["max_seq_length"] = max_seq_length
    elif "max_length" in sig:
        out["max_length"] = max_seq_length
    if "dataset_text_field" in sig:
        out["dataset_text_field"] = None
    if "packing" in sig:
        out["packing"] = False
    return out


def _kwargs_for_callable(func, kwargs: dict) -> dict:
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}

MODEL_ID = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
# Bump when changing load/offload behavior (check logs on Kaggle to confirm sync).
_TRAIN_LORA_BUILD = "2025-03-23+ram-cap-sigkill"

# Default HF-style targets (good for many Llama-like MoE checkpoints)
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Kaggle competition baseline (matches public notebooks using Nemotron-3 Nano PyTorch modules)
KAGGLE_LORA_TARGET_REGEX = r".*\.(in_proj|out_proj|up_proj|down_proj)$"


def _find_kaggle_competition_nemotron_dir() -> str | None:
    """Weights mounted as a competition Model input under /kaggle/input/models/..."""
    preferred = Path("/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16")
    for root in (preferred, Path("/kaggle/input/models")):
        if not root.is_dir():
            continue
        configs = [c for c in root.rglob("config.json") if c.is_file()]
        if not configs:
            continue
        if root == preferred:
            configs.sort(key=lambda p: len(p.parts))
            return str(configs[0].parent.resolve())
        for cfg in sorted(configs, key=lambda p: len(str(p))):
            low = str(cfg).lower()
            if "nemotron" in low and "nano" in low:
                return str(cfg.parent.resolve())
    return None


def resolve_model_weights_path(cli_path: str) -> str:
    """
    Prefer a local directory with config.json; then env NEMOTRON_MODEL_PATH;
    then Kaggle competition mount. Otherwise keep cli_path (Hub id or path).
    """
    p = Path(cli_path)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    envp = os.environ.get("NEMOTRON_MODEL_PATH", "").strip()
    if envp:
        ep = Path(envp)
        if ep.is_dir() and (ep / "config.json").is_file():
            return str(ep.resolve())
    found = _find_kaggle_competition_nemotron_dir()
    if found:
        return found
    return cli_path


def _ensure_base_model_in_adapter(save_dir: Path, base_id: str) -> None:
    cfg_path = save_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = base_id
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_model_unsloth(max_seq_length: int):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer, "unsloth"


def _lora_target_modules(mode: str):
    if mode == "kaggle_nemotron":
        return KAGGLE_LORA_TARGET_REGEX
    if mode == "hf_linear":
        return LORA_TARGET_MODULES
    raise ValueError(f"Unknown lora target mode: {mode}")


def _from_pretrained_dtype_kw(model_cls) -> dict:
    """Use `dtype` when supported (transformers deprecation of torch_dtype)."""
    params = inspect.signature(model_cls.from_pretrained).parameters
    if "dtype" in params:
        return {"dtype": torch.bfloat16}
    if "torch_dtype" in params:
        return {"torch_dtype": torch.bfloat16}
    return {}


def _cuda_total_bytes(device_index: int = 0) -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(device_index).total_memory)
    except Exception:
        return None


def _normalize_max_memory(mm: dict) -> dict:
    """
    Accelerate only treats *integer* keys as GPU ids (see get_max_memory: gpu_devices =
    keys where isinstance(k, int)). JSON uses string keys, so '{"0":"6GiB"}' must become
    {0: bytes, "cpu": bytes} or the GPU cap is ignored and the full ~15GiB 4-bit model
    lands on a 15GiB card → OOM.
    """
    out: dict = {}
    for k, v in mm.items():
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = v
        else:
            out[k] = v
    return out


def _auto_cpu_max_memory_cap() -> str:
    """
    Accelerate packs offloaded weights into *host RAM* up to the "cpu" cap. Using
    cpu=200GiB on a ~13–30GiB machine makes the loader commit too much RAM → Linux
    OOM killer sends SIGKILL (often near end of shard load).
    """
    env = os.environ.get("NEMOTRON_MAX_MEMORY_CPU", "").strip()
    if env:
        return env if "GiB" in env or "GB" in env.upper() else f"{env}GiB"
    try:
        import psutil

        total_gib = psutil.virtual_memory().total / (1024**3)
        # Stay under real RAM: kernel + notebook + loader buffers need headroom.
        cap = int(total_gib * 0.38)
        cap = max(6, min(cap, 18))
        return f"{cap}GiB"
    except Exception:
        return "10GiB"


def _auto_max_memory_for_quantized_load() -> dict[int | str, str] | None:
    """
    30B in 4-bit is ~15 GiB of weights alone; a 14–16 GiB GPU cannot hold the full
    model + load buffers. Cap GPU 0 (integer key!), bounded host RAM, disk spill.
    """
    total = _cuda_total_bytes(0)
    if total is None or total >= 22 * 1024**3:
        return None
    gib = total / (1024**3)
    # Tight GPU cap + CPU + disk: shard load can spike; MoE size estimates can be off.
    gpu_gib = max(2, min(6, int(gib * 0.20)))
    return {
        0: f"{gpu_gib}GiB",
        "cpu": _auto_cpu_max_memory_cap(),
        "disk": "150GiB",
    }


def load_model_peft(
    model_path: str,
    *,
    quantize: bool,
    lora_alpha: int,
    lora_target_mode: str,
    offload_folder: Path,
    max_memory: dict | None = None,
    no_auto_max_memory: bool = False,
):
    import gc

    import transformers
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tv = getattr(transformers, "__version__", "0")
    try:
        t_major = int(tv.split(".", 1)[0])
    except ValueError:
        t_major = 0
    if t_major >= 5 and quantize:
        print(
            "[03_train_lora] WARNING: transformers>=5 + 4-bit often OOMs on ~16GB GPUs "
            "(HF issue: quantized weights materialize on GPU before quant). "
            "Pin: pip install 'transformers>=4.45,<5'. Current:",
            tv,
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload_folder = Path(offload_folder).expanduser().resolve()
    offload_folder.mkdir(parents=True, exist_ok=True)
    offload_str = str(offload_folder)
    print(f"[03_train_lora] offload_folder={offload_str}", flush=True)

    effective_mm = _normalize_max_memory(max_memory) if max_memory else None
    if effective_mm is None and quantize and not no_auto_max_memory:
        effective_mm = _auto_max_memory_for_quantized_load()
        if effective_mm is not None:
            print(
                f"[03_train_lora] auto max_memory (small GPU + 4-bit): {effective_mm}",
                flush=True,
            )

    _sig = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    base_kw: dict = dict(
        device_map="auto",
        trust_remote_code=True,
        offload_folder=offload_str,
    )
    # QLoRA: avoid bf16 dtype on the full skeleton — let BitsAndBytesConfig own compute dtype.
    if not quantize:
        base_kw.update(_from_pretrained_dtype_kw(AutoModelForCausalLM))
    if "low_cpu_mem_usage" in _sig:
        base_kw["low_cpu_mem_usage"] = True
    if effective_mm is not None and "max_memory" in _sig:
        base_kw["max_memory"] = effective_mm
    if quantize:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_kw["quantization_config"] = bnb

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FlashAttention-2 adds VRAM pressure; on tight GPUs prefer SDPA first.
    low_vram_load = effective_mm is not None
    attn_order: list[str | None] = (
        ["sdpa", None] if low_vram_load else ["flash_attention_2", "sdpa", None]
    )
    model = None
    last_err: Exception | None = None
    for attn in attn_order:
        try:
            if attn:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation=attn,
                    **base_kw,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **base_kw)
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        if last_err is not None:
            raise last_err
        raise RuntimeError("from_pretrained failed with no model and no error")

    if quantize:
        model = prepare_model_for_kbit_training(model)

    targets = _lora_target_modules(lora_target_mode)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, "peft"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for Nemotron")
    parser.add_argument("--data-path", type=Path, default=Path("data/train_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("lora_adapter"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("lora_output"))
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_ID,
        help="HF model id or local path (e.g. kagglehub.model_download(...)).",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Load base in bf16 without 4-bit quantization (typical Kaggle kagglehub weights).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha; competition notebooks often use 16 with kaggle_nemotron targets.",
    )
    parser.add_argument(
        "--lora-target-mode",
        choices=("hf_linear", "kaggle_nemotron"),
        default="hf_linear",
        help="kaggle_nemotron: regex in_proj/out_proj/up/down_proj (Kaggle starter pattern).",
    )
    parser.add_argument(
        "--adapter-base-name",
        type=str,
        default=MODEL_ID,
        help="Written to adapter_config.json base_model_name_or_path for submission tooling.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=None,
        help="Disk directory for accelerate when MoE weights spill off GPU (default: <checkpoint-dir>/hf_offload).",
    )
    parser.add_argument(
        "--max-memory-json",
        type=str,
        default=None,
        help='Optional accelerate max_memory JSON (numeric keys may be strings; script coerces to int). '
        'Example: \'{"0":"5GiB","cpu":"240GiB"}\'.',
    )
    parser.add_argument(
        "--no-auto-max-memory",
        action="store_true",
        help="Do not infer max_memory for <22GB GPUs when using 4-bit (can OOM loading 30B on T4).",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-peft", action="store_true", help="Skip Unsloth")
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 on Kaggle/CUDA to avoid fork deadlocks; was 2).",
    )
    args = parser.parse_args()

    use_wandb = os.environ.get("USE_WANDB", "").lower() in ("1", "true", "yes")
    report_to = "wandb" if use_wandb else "none"

    if not args.data_path.is_file():
        raise SystemExit(f"Missing dataset {args.data_path}. Run 02_prepare_data.py first.")

    print(f"[03_train_lora] build={_TRAIN_LORA_BUILD}", flush=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    offload_folder = args.offload_folder or (args.checkpoint_dir / "hf_offload")
    env_off = os.environ.get("HF_OFFLOAD_FOLDER")
    if env_off:
        offload_folder = Path(env_off)

    max_memory = None
    if args.max_memory_json:
        max_memory = json.loads(args.max_memory_json)
        if not isinstance(max_memory, dict):
            raise SystemExit("--max-memory-json must be a JSON object")

    ds = load_dataset("json", data_files=str(args.data_path), split="train")
    ds = ds.train_test_split(test_size=args.test_size, seed=args.seed)

    model_path = resolve_model_weights_path(args.model_path)
    if model_path != args.model_path:
        print(
            f"[03_train_lora] --model-path resolved: {args.model_path!r} -> {model_path!r}",
            flush=True,
        )

    model = None
    tokenizer = None
    backend = ""
    use_unsloth = (
        not args.force_peft
        and not args.no_quant
        and model_path == MODEL_ID
    )
    if use_unsloth:
        try:
            model, tokenizer, backend = load_model_unsloth(args.max_seq_length)
            print("Loaded model via Unsloth")
        except Exception as e:
            print(f"Unsloth load failed ({e!r}); falling back to PEFT.")
    if model is None:
        try:
            model, tokenizer, backend = load_model_peft(
                model_path,
                quantize=not args.no_quant,
                lora_alpha=args.lora_alpha,
                lora_target_mode=args.lora_target_mode,
                offload_folder=offload_folder,
                max_memory=max_memory,
                no_auto_max_memory=args.no_auto_max_memory,
            )
            q = "PEFT + 4-bit" if not args.no_quant else "PEFT + bf16 (no quant)"
            print(f"Loaded model via {q}")
        except Exception as e:
            err = str(e).lower()
            if "mamba" in err:
                raise SystemExit(
                    "PEFT load failed: Nemotron-3 includes Mamba layers and needs `mamba-ssm` "
                    "(and usually `causal-conv1d`). On Kaggle, add to your pip cell:\n"
                    "  pip install mamba-ssm causal-conv1d\n"
                    f"Original error: {e!r}"
                ) from e
            if "offload_folder" in err or "offloaded to the disk" in err:
                raise SystemExit(
                    "PEFT load failed: MoE + device_map often needs a disk offload dir. "
                    "This script defaults to <checkpoint-dir>/hf_offload; set explicitly with:\n"
                    "  --offload-folder /kaggle/working/project/lora_output/hf_offload\n"
                    "Or drop --no-quant to use 4-bit QLoRA and reduce GPU/disk offload.\n"
                    f"Original error: {e!r}"
                ) from e
            if "not a valid model identifier" in err or "is not a local folder" in err:
                raise SystemExit(
                    "PEFT load failed: Hugging Face could not load this --model-path.\n"
                    "On Kaggle, add the competition **Model** input (mounts under /kaggle/input/models/...) "
                    "or set env NEMOTRON_MODEL_PATH to a folder containing config.json.\n"
                    "This script auto-searches /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/.\n"
                    f"Original error: {e!r}"
                ) from e
            if "out of memory" in err:
                raise SystemExit(
                    "PEFT load failed: CUDA OOM while loading weights.\n"
                    "1) Pin transformers 4.x (v5 can materialize FP weights on GPU before 4-bit quant — "
                    "max_memory may not help):  pip install -U 'transformers>=4.45,<5'\n"
                    "2) Integer GPU keys in max_memory (script auto-sets this). Tighten further, e.g.\n"
                    '     --max-memory-json \'{"0":"2GiB","cpu":"200GiB","disk":"120GiB"}\'\n'
                    "3) Avoid --no-quant on T4.\n"
                    f"Original error: {e!r}"
                ) from e
            raise SystemExit(
                f"PEFT load failed ({e!r}).\n"
                "If you still see 'provide offload_folder' or 'Try --force-peft', your Kaggle copy of "
                "scripts/03_train_lora.py is OUTDATED — re-upload the dataset and re-run Bootstrap.\n"
                "Otherwise try: explicit --offload-folder, --max-memory-json, or remove --no-quant."
            ) from e

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    common_kwargs = dict(
        output_dir=str(args.checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=torch.cuda.is_available(),
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=report_to,
        dataloader_num_workers=args.dataloader_workers,
    )
    if hasattr(TrainingArguments, "eval_strategy"):
        common_kwargs["eval_strategy"] = "epoch"
        common_kwargs["load_best_model_at_end"] = True
        common_kwargs["metric_for_best_model"] = "eval_loss"
    else:
        common_kwargs["evaluation_strategy"] = "epoch"
        common_kwargs["load_best_model_at_end"] = True
        common_kwargs["metric_for_best_model"] = "eval_loss"

    cfg_kw = {**common_kwargs, **_sft_config_kwargs(args.max_seq_length)}
    cfg_kw = _kwargs_for_callable(SFTConfig.__init__, cfg_kw)
    try:
        training_args = SFTConfig(**cfg_kw)
    except TypeError:
        cfg_kw.pop("packing", None)
        training_args = SFTConfig(**cfg_kw)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        formatting_func=formatting_func,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(
        "Starting trainer.train() — the progress bar may stay at 0% for a long time on the "
        "**first** step (MoE + disk offload + first backward). If it never moves after ~30–60 min, "
        "set --dataloader-workers 0 and ensure GPU RAM is not thrashing.",
        flush=True,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    _ensure_base_model_in_adapter(args.output_dir, args.adapter_base_name)
    print(f"Saved LoRA adapter to {args.output_dir} (backend={backend})")


if __name__ == "__main__":
    main()
