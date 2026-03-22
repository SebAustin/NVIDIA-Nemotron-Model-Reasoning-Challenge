"""
Phase 4: Evaluate LoRA adapter with vLLM — inference, answer extraction, accuracy by puzzle type.
Run from project root: python scripts/04_evaluate.py
Requires vllm (install with pip install -r requirements-vllm.txt on Linux+CUDA).
"""
import os
import sys

# Ensure project root is on path when run from any cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print(
        "Skipping evaluation: vLLM is not installed. vLLM only supports Linux with an NVIDIA GPU;\n"
        "it does not install on macOS. To evaluate, run this script on Kaggle (with GPU) or a\n"
        "Linux+NVIDIA machine after: pip install -r requirements-vllm.txt\n"
        "Continuing without evaluation (exit 0)."
    )
    sys.exit(0)

from scripts.utils.model_utils import local_load_kwargs, resolve_model_path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LORA_ADAPTER_DIR = os.path.join(PROJECT_ROOT, "lora_adapter")
_MODEL_PATH_RAW = os.environ.get("NEMOTRON_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
MODEL_NAME = resolve_model_path(_MODEL_PATH_RAW)


def load_eval_data():
    """Load validation data: train_categorized 10% or train.csv slice with answers."""
    import pandas as pd
    train_cat = os.path.join(DATA_DIR, "train_categorized.csv")
    train_csv = os.path.join(DATA_DIR, "train.csv")
    if os.path.isfile(train_cat):
        df = pd.read_csv(train_cat)
    elif os.path.isfile(train_csv):
        df = pd.read_csv(train_csv)
    else:
        return None, None, None
    # Hold out 10% for eval
    df = df.sample(frac=0.1, random_state=42)
    if "answer" not in df.columns:
        return None, None, None
    return df["prompt"].tolist(), df["answer"].tolist(), df.get("puzzle_type")


def build_prompts_for_inference(prompts: list[str], system_prompt: str) -> list[str]:
    """Build full prompt (system + user) for each item using chat template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, **local_load_kwargs(_MODEL_PATH_RAW))
    out = []
    for user_text in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out.append(full)
    return out


def main() -> None:
    os.chdir(PROJECT_ROOT)
    if not os.path.isdir(LORA_ADAPTER_DIR):
        print(f"Error: {LORA_ADAPTER_DIR} not found. Run 03_train_lora.py first.")
        sys.exit(1)

    from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
    from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT

    prompts, ground_truths, puzzle_types = load_eval_data()
    if prompts is None or not prompts:
        print("No evaluation data (train.csv or train_categorized.csv with answer column).")
        sys.exit(1)

    print("Building inference prompts...")
    full_prompts = build_prompts_for_inference(prompts, DEFAULT_SYSTEM_PROMPT)

    print("Loading vLLM with LoRA...")
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=64,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=7680,
    )
    lora_request = LoRARequest("reasoning_adapter", 1, LORA_ADAPTER_DIR)

    print("Running inference...")
    outputs = llm.generate(
        full_prompts,
        sampling_params,
        lora_request=lora_request,
    )
    completions = [o.outputs[0].text for o in outputs]

    # Extract answers and score
    extracted = [extract_boxed_answer(c) for c in completions]
    correct = [answers_match(ex, gt) for ex, gt in zip(extracted, ground_truths)]
    accuracy = sum(correct) / len(correct) if correct else 0.0
    print(f"\nOverall accuracy: {accuracy:.2%} ({sum(correct)}/{len(correct)})")

    by_type = {}
    if puzzle_types is not None and len(puzzle_types) == len(correct):
        from collections import defaultdict
        by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        for pt, c in zip(puzzle_types, correct):
            by_type[pt]["total"] += 1
            if c:
                by_type[pt]["correct"] += 1
        print("\nAccuracy by puzzle type:")
        for pt in sorted(by_type.keys()):
            t = by_type[pt]
            acc = t["correct"] / t["total"] if t["total"] else 0
            print(f"  {pt}: {acc:.2%} ({t['correct']}/{t['total']})")

    # Failure examples
    failures = [(i, extracted[i], ground_truths[i], completions[i][:500]) for i in range(len(correct)) if not correct[i]]
    print(f"\nFailure examples (first 5):")
    for i, (idx, ex, gt, preview) in enumerate(failures[:5]):
        print(f"  [{i+1}] extracted={ex!r}  ground_truth={gt!r}")
        print(f"      completion preview: {preview!r}...")

    # Optional: write report
    report_path = os.path.join(PROJECT_ROOT, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall accuracy: {accuracy:.2%}\n")
        if by_type:
            for pt in sorted(by_type.keys()):
                t = by_type[pt]
                acc = t["correct"] / t["total"] if t["total"] else 0
                f.write(f"  {pt}: {acc:.2%}\n")
        f.write("\nFailure sample:\n")
        for idx, ex, gt, _ in failures[:10]:
            f.write(f"  extracted={ex!r}  gt={gt!r}\n")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
