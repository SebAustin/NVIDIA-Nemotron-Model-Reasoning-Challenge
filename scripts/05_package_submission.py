"""
Phase 5: Package LoRA adapter into submission.zip with validation.
Run from project root: python scripts/05_package_submission.py
"""
import json
import os
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "lora_adapter")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission.zip")

# Tokenizer files that may be omitted for LoRA-only submission (check competition rules)
SKIP_FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"}


def main() -> None:
    os.chdir(PROJECT_ROOT)

    if not os.path.isdir(ADAPTER_DIR):
        print(
            f"{ADAPTER_DIR} not found. Phase 3 (training) was skipped or has not run.\n"
            "To get a submission: run Phase 3 on a machine with an NVIDIA GPU (e.g. Kaggle notebook\n"
            "with GPU enabled). Then run this script there to create submission.zip, or copy lora_adapter/\n"
            "back here and run: python scripts/05_package_submission.py"
        )
        return

    config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
    if not os.path.isfile(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path) as f:
        config = json.load(f)
    r = config.get("r")
    if r is not None and r > 32:
        print(f"Error: LoRA rank {r} exceeds max 32.")
        return
    base = config.get("base_model_name_or_path", "")
    print(f"LoRA rank: {r}")
    print(f"Base model: {base or 'NOT SET'}")

    safetensors = [f for f in os.listdir(ADAPTER_DIR) if f.endswith(".safetensors")]
    bin_files = [f for f in os.listdir(ADAPTER_DIR) if f.endswith(".bin")]
    if not safetensors and not bin_files:
        print("Error: No adapter weights (.safetensors or .bin) found.")
        return

    with zipfile.ZipFile(SUBMISSION_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ADAPTER_DIR):
            for file in files:
                if file in SKIP_FILES:
                    continue
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, ADAPTER_DIR)
                zf.write(filepath, arcname)
                print(f"  Added: {arcname}")

    with zipfile.ZipFile(SUBMISSION_PATH, "r") as zf:
        print(f"\nSubmission contents:")
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

    print(f"\nsubmission.zip created ({os.path.getsize(SUBMISSION_PATH):,} bytes)")


if __name__ == "__main__":
    main()
