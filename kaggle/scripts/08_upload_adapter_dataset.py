#!/usr/bin/env python3
"""Phase 4b: Upload the trained LoRA adapter to Kaggle as a versioned Dataset.

This is the deployment step for Code Competition format submissions:
1. The adapter is bundled into a staging directory with a `dataset-metadata.json`.
2. The Kaggle CLI (`kaggle datasets create` or `kaggle datasets version`) is
   invoked to publish or version the dataset.
3. The dataset becomes visible at https://www.kaggle.com/datasets/<dataset-id>
   so a Kaggle inference notebook can attach it via "Add Data".

Prerequisites:
- Kaggle CLI installed (`pip install kaggle`).
- Credentials at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY in env).

Typical usage on Colab after training completes:
    python scripts/08_upload_adapter_dataset.py \\
        --adapter-dir lora_adapter \\
        --dataset-id sebmontreal/nemotron-lora-adapter \\
        --title "Nemotron LoRA Adapter (Reasoning Challenge)" \\
        --first-time

Subsequent runs (after retraining):
    python scripts/08_upload_adapter_dataset.py \\
        --adapter-dir lora_adapter \\
        --dataset-id sebmontreal/nemotron-lora-adapter \\
        --version-notes "v3: pseudo-labels + 2 epochs"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_kaggle_creds() -> None:
    """Honor either ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if cred_path.is_file():
        try:
            os.chmod(cred_path, 0o600)
        except PermissionError:
            pass
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    raise SystemExit(
        "No Kaggle credentials found. Either:\n"
        "  1) Put kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or\n"
        "  2) Set KAGGLE_USERNAME and KAGGLE_KEY env vars before running this."
    )


def _stage_adapter(
    adapter_dir: Path,
    staging_dir: Path,
    dataset_id: str,
    title: str,
) -> None:
    """Copy adapter files into staging dir and write dataset-metadata.json."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Copy every file from the adapter dir (small: <500 MB typically).
    copied = 0
    for src in adapter_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, staging_dir / src.name)
            copied += 1
    if copied == 0:
        raise SystemExit(f"No files copied from {adapter_dir} -- is the adapter trained?")

    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Staged {copied} file(s) into {staging_dir} for dataset {dataset_id}")


def _run_kaggle(args_list: list[str]) -> None:
    """Run a kaggle CLI command, streaming output."""
    print("$ kaggle " + " ".join(args_list))
    proc = subprocess.run(["kaggle"] + args_list, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"Kaggle CLI exit code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to Kaggle Dataset")
    parser.add_argument("--adapter-dir", type=Path, default=Path("lora_adapter"))
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Kaggle dataset slug, e.g. 'username/nemotron-lora-adapter'.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Human-readable title for the dataset (defaults to the slug name).",
    )
    parser.add_argument(
        "--first-time",
        action="store_true",
        help="Use 'datasets create' (first publish) instead of 'datasets version'.",
    )
    parser.add_argument(
        "--version-notes",
        type=str,
        default="updated adapter",
        help="Note shown on the dataset page for this version.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path(".kaggle_upload_staging"),
        help="Working directory used to assemble the upload (will be wiped).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Mark a freshly-created dataset public (otherwise it's private).",
    )
    args = parser.parse_args()

    if "/" not in args.dataset_id or args.dataset_id.count("/") != 1:
        raise SystemExit("--dataset-id must be of the form 'owner/slug'")
    if not args.adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory not found: {args.adapter_dir}")

    _ensure_kaggle_creds()
    title = args.title or args.dataset_id.split("/", 1)[1].replace("-", " ").title()
    _stage_adapter(args.adapter_dir, args.staging_dir, args.dataset_id, title)

    if args.first_time:
        cli = ["datasets", "create", "-p", str(args.staging_dir)]
        if args.public:
            cli.append("--public")
        _run_kaggle(cli)
        url = f"https://www.kaggle.com/datasets/{args.dataset_id}"
        print(f"\nDataset created. View at: {url}")
    else:
        _run_kaggle(
            [
                "datasets",
                "version",
                "-p",
                str(args.staging_dir),
                "-m",
                args.version_notes,
            ]
        )
        url = f"https://www.kaggle.com/datasets/{args.dataset_id}"
        print(f"\nNew version published. View at: {url}")

    print(
        "\nNext step: in your Kaggle inference notebook, click 'Add Data' and "
        f"attach this dataset by ID: {args.dataset_id}"
    )


if __name__ == "__main__":
    main()
