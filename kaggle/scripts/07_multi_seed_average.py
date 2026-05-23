#!/usr/bin/env python3
"""Phase 7: Multi-seed LoRA training + weight averaging.

Per Kaggle Grandmasters Playbook tip #7: training multiple models with
different random seeds and averaging them is a cheap, reliable way to
squeeze out an extra +0.02-0.05 in accuracy. Kaggle only accepts ONE LoRA
adapter, so we average the LoRA `A`/`B` matrices into a single submission
adapter (Wortsman et al., 'Model Soups').

Usage:
    python scripts/07_multi_seed_average.py \\
        --num-seeds 3 \\
        --base-seed 42 \\
        --model-path /path/to/Nemotron \\
        --output-dir lora_adapter \\
        --intermediate-dir multi_seed_runs

The script:
  1. Calls 03_train_lora.py with seeds [base_seed, base_seed+1, ...].
  2. Each run writes its adapter to multi_seed_runs/seed_{i}/lora_adapter/.
  3. Averages the resulting adapters into --output-dir.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.lora_merge import average_adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed LoRA train + average")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/train_sft.jsonl"),
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("multi_seed_runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lora_adapter"),
        help="Directory for the averaged adapter (this is what gets submitted).",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-target-mode", type=str, default="kaggle_nemotron")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--no-bf16-full",
        action="store_true",
        help="Quantize during training (default: full bf16, matches SFT v2 recipe).",
    )
    parser.add_argument(
        "--completion-only",
        action="store_true",
        default=True,
        help="Mask system+user tokens (default on, matches SFT v2 recipe).",
    )
    parser.add_argument(
        "--no-completion-only",
        dest="completion_only",
        action="store_false",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; just average existing per-seed adapters in --intermediate-dir.",
    )
    args = parser.parse_args()

    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    seed_adapter_dirs: List[Path] = []

    for i in range(args.num_seeds):
        seed = args.base_seed + i
        run_dir = args.intermediate_dir / f"seed_{seed}"
        adapter_dir = run_dir / "lora_adapter"
        ckpt_dir = run_dir / "lora_output"
        seed_adapter_dirs.append(adapter_dir)

        if args.skip_train:
            if not (adapter_dir / "adapter_config.json").is_file():
                raise SystemExit(
                    f"--skip-train set but no adapter at {adapter_dir}. "
                    "Train first or drop --skip-train."
                )
            print(f"[seed={seed}] reusing existing adapter at {adapter_dir}")
            continue

        cmd = [
            sys.executable, "scripts/03_train_lora.py",
            "--data-path", str(args.data_path),
            "--output-dir", str(adapter_dir),
            "--checkpoint-dir", str(ckpt_dir),
            "--model-path", args.model_path,
            "--lora-target-mode", args.lora_target_mode,
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--max-seq-length", str(args.max_seq_length),
            "--seed", str(seed),
            "--force-peft",
            "--no-nemotron-kaggle-patches",
            "--dataloader-workers", "0",
        ]
        if not args.no_bf16_full:
            cmd.append("--no-quant")
        if args.completion_only:
            cmd.append("--completion-only")
        print(f"\n=== Seed {i+1}/{args.num_seeds} (seed={seed}) ===")
        print(" ".join(cmd))
        log_path = run_dir / "train_log.txt"
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as logf:
            rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(
                f"Seed {seed} training failed (rc={rc}). See {log_path} for details."
            )
        print(f"  done (log: {log_path})")

    print(f"\nAveraging {len(seed_adapter_dirs)} adapters into {args.output_dir}")
    summary = average_adapters(seed_adapter_dirs, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"Submission-ready adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
