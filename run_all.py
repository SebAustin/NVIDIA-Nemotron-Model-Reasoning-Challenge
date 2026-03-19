#!/usr/bin/env python3
"""
Run the full Nemotron Reasoning Challenge pipeline in order:
  01_eda.py -> 02_prepare_data.py -> 03_train_lora.py -> 04_evaluate.py -> 05_package_submission.py

Usage (from project root, with venv activated):
  python run_all.py [options]

Options:
  --skip-eda          Skip Phase 1 (requires data/train.csv, data/test.csv already present)
  --skip-prepare      Skip Phase 2 (requires data/train_categorized.csv or data/train.csv)
  --synthetic-only   In Phase 2, only generate synthetic data (no CoT API calls)
  --skip-train        Skip Phase 3 (LoRA training; needs GPU)
  --skip-eval         Skip Phase 4 (vLLM evaluation; needs GPU)
  --skip-package      Skip Phase 5 (submission zip)
  --max-cot N         In Phase 2, limit CoT generation to N training examples (default: all)
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS = [
    "01_eda.py",
    "02_prepare_data.py",
    "03_train_lora.py",
    "04_evaluate.py",
    "05_package_submission.py",
]


def run_script(name: str, extra_args: list[str]) -> bool:
    path = os.path.join(PROJECT_ROOT, "scripts", name)
    if not os.path.isfile(path):
        print(f"Skip {name}: not found at {path}")
        return True
    cmd = [sys.executable, path] + extra_args
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Exiting: {name} failed with code {result.returncode}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Nemotron pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip Phase 1 EDA")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip Phase 2 data prep")
    parser.add_argument("--synthetic-only", action="store_true", help="Phase 2: only synthetic data")
    parser.add_argument("--skip-train", action="store_true", help="Skip Phase 3 LoRA training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip Phase 4 evaluation")
    parser.add_argument("--skip-package", action="store_true", help="Skip Phase 5 packaging")
    parser.add_argument("--max-cot", type=int, default=None, help="Phase 2: max CoT examples")
    args = parser.parse_args()

    steps = [
        (not args.skip_eda, "01_eda.py", []),
        (not args.skip_prepare, "02_prepare_data.py", (
            ["--synthetic-only"] if args.synthetic_only else []
            + (["--max-cot", str(args.max_cot)] if args.max_cot is not None else [])
        )),
        (not args.skip_train, "03_train_lora.py", []),
        (not args.skip_eval, "04_evaluate.py", []),
        (not args.skip_package, "05_package_submission.py", []),
    ]

    for do_run, name, extra in steps:
        if not do_run:
            print(f"\nSkipping {name} (disabled)")
            continue
        if not run_script(name, extra):
            return 1

    print("\n" + "="*60)
    print("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
