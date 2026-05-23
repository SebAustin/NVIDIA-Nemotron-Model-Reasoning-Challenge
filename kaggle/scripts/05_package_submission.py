#!/usr/bin/env python3
"""Phase 5: Validate LoRA adapter and build submission.zip."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Files that look like tokenizer artifacts. Whether to include them is decided
# at call time. Kaggle's harness loads the LoRA adapter against the base model's
# tokenizer (a LoRA does not modify embeddings), so shipping these is normally
# decorative. Two exceptions where they DO matter:
#   1. We saved a custom `chat_template.jinja` that differs from the base model's.
#   2. We added special tokens during training (we don't).
# Phase 0.2 audit chose the safer default: INCLUDE them (`--exclude-tokenizer`
# to opt out). This matches what `kaggle/kaggle_submission.ipynb` Phase 2 has
# always done, so the two paths now agree.
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package LoRA submission zip")
    parser.add_argument("--adapter-dir", type=Path, default=Path("lora_adapter"))
    parser.add_argument("--output", type=Path, default=Path("submission.zip"))
    parser.add_argument(
        "--exclude-tokenizer",
        action="store_true",
        help="Strip tokenizer files from the zip (use only for the A/B test).",
    )
    # Kept for backwards compat; flips meaning vs `--exclude-tokenizer`.
    parser.add_argument(
        "--include-tokenizer",
        action="store_true",
        help="Deprecated; tokenizer files are included by default.",
    )
    args = parser.parse_args()

    adapter = args.adapter_dir
    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter}")

    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        config = json.load(f)
    r = config.get("r")
    if r is None:
        raise SystemExit("adapter_config.json missing 'r'")
    if r > 32:
        raise SystemExit(f"LoRA rank {r} exceeds max of 32")
    print(f"LoRA rank: {r}")
    base = config.get("base_model_name_or_path", "NOT SET")
    print(f"Base model: {base}")

    safetensors = [f for f in os.listdir(adapter) if f.endswith(".safetensors")]
    bin_files = [f for f in os.listdir(adapter) if f.endswith(".bin")]
    if not safetensors and not bin_files:
        raise SystemExit("No adapter weights found (.safetensors or .bin)")

    skip: set[str] = set(TOKENIZER_NAMES) if args.exclude_tokenizer else set()
    if args.include_tokenizer and args.exclude_tokenizer:
        raise SystemExit("Pass only one of --include-tokenizer / --exclude-tokenizer")

    out_zip = args.output
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(adapter):
            for name in files:
                if name in skip:
                    continue
                filepath = Path(root) / name
                arcname = filepath.relative_to(adapter)
                zf.write(filepath, str(arcname))
                print(f"  Added: {arcname}")

    with zipfile.ZipFile(out_zip, "r") as zf:
        print("\nSubmission contents:")
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

    size = out_zip.stat().st_size
    print(f"\nCreated {out_zip} ({size:,} bytes)")


if __name__ == "__main__":
    main()
