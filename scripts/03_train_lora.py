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

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_SFT_JSONL = os.path.join(DATA_DIR, "train_sft.jsonl")
LORA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lora_output")
LORA_ADAPTER_DIR = os.path.join(PROJECT_ROOT, "lora_adapter")
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def load_model_unsloth():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
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
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=_get_model_config(),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
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

    # Use 4096 for training on 2×T4 (16GB each) to avoid OOM; inference can still use 8192.
    max_seq = int(os.environ.get("SFT_MAX_SEQ_LENGTH", "4096"))
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
