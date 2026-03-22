#!/usr/bin/env python3
"""
Zip the Nemotron model directory (for upload as a Kaggle Dataset or backup).
Uses NEMOTRON_MODEL_PATH env var, or pass path as first argument.

Usage:
  export NEMOTRON_MODEL_PATH=/path/to/model
  python scripts/zip_nemotron_model.py

  # Or:
  python scripts/zip_nemotron_model.py /path/to/model
"""
import argparse
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip the Nemotron model directory")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.environ.get("NEMOTRON_MODEL_PATH"),
        help="Path to model directory (default: NEMOTRON_MODEL_PATH env)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output zip path (default: nemotron-model.zip in project root)",
    )
    args = parser.parse_args()

    src = args.path
    if not src or not os.path.isdir(src):
        print(
            "Error: Provide a valid model path via NEMOTRON_MODEL_PATH or as argument.\n"
            "Example: python scripts/zip_nemotron_model.py /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            file=sys.stderr,
        )
        sys.exit(1)

    out = args.output or os.path.join(PROJECT_ROOT, "nemotron-model.zip")
    base_name = out.replace(".zip", "").rstrip("/")
    print(f"Zipping {src} -> {base_name}.zip")
    shutil.make_archive(base_name, "zip", os.path.dirname(src), os.path.basename(src))
    zip_path = base_name + ".zip"
    if os.path.exists(zip_path):
        print(f"Created {zip_path} ({os.path.getsize(zip_path) / (1024**3):.1f} GB)")
    else:
        print("Zip may have been created elsewhere", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
