"""
Phase 3: LoRA SFT (and optional GRPO) for Nemotron-3-Nano-30B-A3B-BF16.
Run from project root: python scripts/03_train_lora.py [--grpo]
"""
import argparse
import os
import sys

# Ensure project root is on path when run from any cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Must run before `import torch` / first CUDA alloc (Kaggle OOM hints often cite the wrong var name).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_SFT_JSONL = os.path.join(DATA_DIR, "train_sft.jsonl")
LORA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lora_output")
LORA_ADAPTER_DIR = os.path.join(PROJECT_ROOT, "lora_adapter")
from scripts.utils.model_utils import local_load_kwargs, resolve_model_path

_MODEL_PATH_RAW = os.environ.get("NEMOTRON_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
MODEL_NAME = resolve_model_path(_MODEL_PATH_RAW)


def _local_load_kwargs() -> dict:
    return local_load_kwargs(_MODEL_PATH_RAW)


def _hf_offload_dir() -> str:
    d = os.path.join(PROJECT_ROOT, "hf_offload")
    os.makedirs(d, exist_ok=True)
    return d


def _device_map_kwargs_for_quantized_load() -> dict:
    """
    Cap VRAM per GPU so a 30B 4-bit MoE splits across GPUs + CPU/RAM instead of filling GPU 0 during load.

    Parallel tensor materialization can briefly spike one GPU; defaults are conservative (~7–9 GiB/GPU on 2×T4).
    Env:
      SFT_MAX_MEMORY_GB_PER_GPU — same cap for every GPU (overrides asymmetric defaults)
      SFT_MAX_MEMORY_GB_GPU0 / SFT_MAX_MEMORY_GB_GPU1 — per-device GiB (when SFT_MAX_MEMORY_GB_PER_GPU unset)
      SFT_MAX_MEMORY_CPU — CPU RAM budget for offload (default 160GiB)
    """
    if not torch.cuda.is_available():
        return {}
    n = torch.cuda.device_count()
    cpu_cap = os.environ.get("SFT_MAX_MEMORY_CPU", "160GiB")
    if cpu_cap.isdigit():
        cpu_cap = f"{cpu_cap}GiB"

    uniform = os.environ.get("SFT_MAX_MEMORY_GB_PER_GPU")
    max_memory: dict = {}
    if uniform is not None:
        gb = int(uniform)
        max_memory = {i: f"{gb}GiB" for i in range(n)}
    elif n >= 2:
        g0 = int(os.environ.get("SFT_MAX_MEMORY_GB_GPU0", "7"))
        g1 = int(os.environ.get("SFT_MAX_MEMORY_GB_GPU1", "9"))
        max_memory = {0: f"{g0}GiB", 1: f"{g1}GiB"}
        for i in range(2, n):
            max_memory[i] = f"{g1}GiB"
    else:
        g0 = int(os.environ.get("SFT_MAX_MEMORY_GB_GPU0", "5"))
        max_memory = {0: f"{g0}GiB"}

    max_memory["cpu"] = cpu_cap
    gpu_caps = {k: max_memory[k] for k in max_memory if k != "cpu"}
    print(f"Quantized load: {n} GPU(s), max_memory (GPUs) = {gpu_caps}, CPU = {cpu_cap}")
    return {
        "device_map": "auto",
        "max_memory": max_memory,
        "low_cpu_mem_usage": True,
        "offload_folder": _hf_offload_dir(),
    }


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, **_local_load_kwargs())


def load_model_unsloth():
    from unsloth import FastLanguageModel
    kw = _local_load_kwargs()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
        **kw,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def _get_model_config():
    """Load config with tie_word_embeddings=False to silence tied-weights warning."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, **_local_load_kwargs())
    config.tie_word_embeddings = False
    return config


def load_model_peft():
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if torch.cuda.device_count() < 2:
        print(
            "Warning: Only one GPU is visible. 30B MoE in 4-bit needs ~15GB+ VRAM for weights; use **2× T4** on Kaggle "
            "(Accelerator → 2× GPU) so the model splits across GPUs. Single T4 will CPU-offload and often OOM or train very slowly."
        )
    map_kw = _device_map_kwargs_for_quantized_load()
    # Cap fraction of each GPU PyTorch may use (reduces OOM during parallel weight materialization).
    frac = float(os.environ.get("SFT_CUDA_MEMORY_FRACTION", "0.82"))
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_per_process_memory_fraction(frac, i)
        except Exception:
            pass
    torch.cuda.empty_cache()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=_get_model_config(),
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **_local_load_kwargs(),
            **map_kw,
        )
    except torch.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA OOM while loading the 4-bit model. Try: (1) Kaggle **2× GPU T4** not 1×; "
            "(2) lower caps, e.g. SFT_MAX_MEMORY_GB_GPU0=6 SFT_MAX_MEMORY_GB_GPU1=8 or SFT_MAX_MEMORY_GB_PER_GPU=6; "
            "(3) SFT_CUDA_MEMORY_FRACTION=0.78; (4) more CPU RAM for offload: SFT_MAX_MEMORY_CPU=200. "
            f"Original error: {e}"
        ) from e
    except ImportError as e:
        err = str(e)
        if "selective_scan_cuda" in err or "c10_cuda" in err or "mamba_ssm" in err:
            raise RuntimeError(
                "mamba_ssm CUDA extensions do not match your installed PyTorch (ABI mismatch). "
                "Typical on Kaggle after `pip install` upgrades torch. Fix: run\n"
                "  pip install mamba-ssm causal-conv1d --no-cache-dir --force-reinstall\n"
                "and use requirements-kaggle-peft.txt (no torch line) so pip does not upgrade torch.\n"
                f"Original: {e}"
            ) from e
        raise
    except ValueError as e:
        if "dispatched on the CPU or the disk" in str(e):
            cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            raise RuntimeError(
                f"Model could not fit on GPU. {e}\n"
                f"Your GPU compute capability is {cap[0]}.{cap[1]}. "
                "On Kaggle, Tesla P100 (6.0) is not supported by the default PyTorch. "
                "Re-run the notebook to get a different GPU (e.g. T4, V100, A100), or use a competition that provides one."
            ) from e
        raise
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, **_local_load_kwargs())
    tokenizer.pad_token = tokenizer.eos_token

    lora_r = int(os.environ.get("LORA_R", "32"))
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for gradient checkpointing with LoRA
    model.print_trainable_parameters()
    return model, tokenizer


def run_sft(use_unsloth: bool = True) -> None:
    os.chdir(PROJECT_ROOT)
    if not os.path.isfile(TRAIN_SFT_JSONL):
        print(f"Error: {TRAIN_SFT_JSONL} not found. Run scripts/02_prepare_data.py first.")
        sys.exit(1)

    print("Loading model and tokenizer...")
    if use_unsloth:
        try:
            model, tokenizer = load_model_unsloth()
        except Exception as e:
            print(f"Unsloth failed: {e}. Falling back to PEFT.")
            model, tokenizer = load_model_peft()
    else:
        model, tokenizer = load_model_peft()

    dataset = load_dataset("json", data_files=TRAIN_SFT_JSONL, split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # Default 2048 for 2×T4 (16GB each); set SFT_MAX_SEQ_LENGTH=4096 or 8192 if you have more VRAM.
    max_seq = int(os.environ.get("SFT_MAX_SEQ_LENGTH", "2048"))
    sft_config = SFTConfig(
        output_dir=LORA_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=4,
        max_seq_length=max_seq,
        dataset_text_field=None,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )
    trainer.train()
    os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    print(f"Saved adapter to {LORA_ADAPTER_DIR}")


def make_reward_fn_for_grpo():
    """Return a reward callable (completions, prompts, ground_truths) -> list of floats."""
    from scripts.utils.answer_extractor import answers_match, extract_boxed_answer

    def reward_fn(completions, prompts, ground_truths):
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            extracted = extract_boxed_answer(completion)
            if extracted is None:
                rewards.append(-1.0)
            elif answers_match(extracted, gt):
                rewards.append(1.0)
            else:
                try:
                    pred_val = float(extracted)
                    gt_val = float(gt)
                    if abs(pred_val - gt_val) <= 1e-6 * abs(gt_val):
                        rewards.append(1.0)
                    else:
                        rewards.append(-0.5)
                except (ValueError, TypeError):
                    rewards.append(-0.5)
        return rewards

    return reward_fn


def run_grpo() -> None:
    """
    Optional Stage 2: GRPO on top of SFT adapter.
    Requires lora_adapter/ from a previous SFT run. Builds a prompt+answer dataset from
    train_sft.jsonl and runs GRPOTrainer with boxed-answer reward.
    """
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from scripts.utils.answer_extractor import extract_boxed_answer

    os.chdir(PROJECT_ROOT)
    if not os.path.isdir(LORA_ADAPTER_DIR):
        print("lora_adapter/ not found. Run SFT first (without --grpo).")
        sys.exit(1)
    if not os.path.isfile(TRAIN_SFT_JSONL):
        print(f"{TRAIN_SFT_JSONL} not found.")
        sys.exit(1)

    # Load base model then load SFT adapter
    print("Loading base model and SFT adapter for GRPO...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=_get_model_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **_local_load_kwargs(),
    )
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset: need "prompt" and "answer" for reward. Build from train_sft.jsonl.
    ds = load_dataset("json", data_files=TRAIN_SFT_JSONL, split="train")

    def to_prompt_and_answer(example):
        messages = example["messages"]
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        asst = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        extracted = extract_boxed_answer(asst)
        answer = extracted.strip() if extracted else ""
        return {"prompt": user, "answer": answer}

    ds = ds.map(to_prompt_and_answer, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["answer"]) > 0)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds["train"]

    # Reward function: TRL may pass batch dict; we expect ground_truths in batch
    reward_fn = make_reward_fn_for_grpo()

    def _reward(completions, prompts, batch):
        gts = batch.get("answer", [])
        if len(gts) != len(completions):
            gts = [""] * len(completions)
        return reward_fn(completions, prompts, gts)

    grpo_config = GRPOConfig(
        output_dir=os.path.join(PROJECT_ROOT, "lora_grpo_output"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        num_train_epochs=1,
        bf16=True,
        num_generations=4,
        max_completion_length=4096,
        max_prompt_length=4096,
        seed=42,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=_reward,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    print(f"GRPO adapter saved to {LORA_ADAPTER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo", action="store_true", help="Run GRPO stage after SFT (placeholder)")
    parser.add_argument("--peft-only", action="store_true", help="Use HuggingFace PEFT instead of Unsloth")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "No NVIDIA GPU detected. LoRA training requires an NVIDIA GPU (e.g. ~24GB VRAM for 4-bit).\n"
            "Run Phase 3 on Kaggle (enable GPU), Google Colab with GPU, or a cloud VM with NVIDIA GPU.\n"
            "Use run_all.py --skip-train to run only EDA, data prep, and packaging locally."
        )
        sys.exit(1)
    # Kaggle sometimes assigns Tesla P100 (sm_60); default PyTorch there does not support sm_60.
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7:
            print(
                f"Warning: GPU compute capability is {cap[0]}.{cap[1]} (e.g. Tesla P100).\n"
                "Kaggle's PyTorch often does not support sm_60. If loading fails with 'dispatched on the CPU',\n"
                "re-run the notebook to get a different GPU (e.g. T4, V100, A100) or use a competition with a supported GPU."
            )
    except Exception:
        pass

    if args.grpo:
        run_grpo()
    else:
        run_sft(use_unsloth=not args.peft_only)


if __name__ == "__main__":
    main()
