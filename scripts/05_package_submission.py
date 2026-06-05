#!/usr/bin/env python3
"""Phase 5: Validate the LoRA adapter and build submission.zip.

The eval harness unzips the submission and loads the adapter directly, so the
adapter files (adapter_config.json + adapter_model.safetensors) must sit at the
ZIP ROOT — matching the official demo's ``zip -m submission.zip *``. We verify
adapter_config.json is present with r<=32, write the zip with files at the root,
and print the final file listing plus r / target_modules for a sanity check.
"""

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

# tokenizer files are not part of a LoRA adapter; exclude by default
SKIP_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Package LoRA submission zip")
    ap.add_argument("--adapter-dir", type=Path, default=Path("lora_adapter"))
    ap.add_argument("--output", type=Path, default=Path("submission.zip"))
    ap.add_argument("--include-tokenizer", action="store_true")
    args = ap.parse_args()

    adapter = args.adapter_dir
    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter}")

    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path} — not a valid PEFT adapter.")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    r = config.get("r")
    if r is None:
        raise SystemExit("adapter_config.json missing 'r'.")
    if r > 32:
        raise SystemExit(f"LoRA rank r={r} exceeds the competition max of 32.")

    weights = [f for f in os.listdir(adapter)
               if f.endswith((".safetensors", ".bin"))]
    if not weights:
        raise SystemExit("No adapter weights (.safetensors/.bin) found.")

    skip = set() if args.include_tokenizer else set(SKIP_NAMES)
    if args.output.exists():
        args.output.unlink()
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(adapter):
            for name in files:
                if name in skip:
                    continue
                fp = Path(root) / name
                zf.write(fp, str(fp.relative_to(adapter)))  # arcname at ZIP ROOT

    # ----- final sanity check -----
    with zipfile.ZipFile(args.output) as zf:
        names = zf.namelist()
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")
    print("Submission contents (zip root):")
    for n in names:
        print(f"  - {n}")
    has_cfg = "adapter_config.json" in names
    has_w = any(n.endswith((".safetensors", ".bin")) for n in names)
    print("\nSanity check:")
    print(f"  adapter_config.json at root : {has_cfg}")
    print(f"  adapter weights present     : {has_w}")
    print(f"  r                           : {r}  (<= 32 ✓)")
    print(f"  lora_alpha                  : {config.get('lora_alpha')}")
    print(f"  target_modules              : {config.get('target_modules')}")
    print(f"  base_model                  : {config.get('base_model_name_or_path')}")
    if not (has_cfg and has_w):
        raise SystemExit("Submission is missing config or weights at the zip root.")
    print("\nOK: submission.zip is ready.")


if __name__ == "__main__":
    main()
