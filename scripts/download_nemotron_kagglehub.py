#!/usr/bin/env python3
"""
Download Nemotron 30B via kagglehub (Kaggle Model Hub).
Run on Kaggle or where kagglehub is installed and authenticated.

Usage:
  python scripts/download_nemotron_kagglehub.py

Output: prints the path to the model files. Set NEMOTRON_MODEL_PATH to that path
before running the training pipeline.
"""
import sys

try:
    import kagglehub
except ImportError:
    print(
        "kagglehub not installed. Run: pip install kagglehub\n"
        "This script is intended for Kaggle (kagglehub is preinstalled there).",
        file=sys.stderr,
    )
    sys.exit(1)

path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print("Path to model files:", path)
print("\nSet NEMOTRON_MODEL_PATH to this path before training, e.g.:")
print(f"  export NEMOTRON_MODEL_PATH={path}")
