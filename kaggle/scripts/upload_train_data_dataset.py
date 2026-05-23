#!/usr/bin/env python3
"""Phase 1.7: Upload the offline-built training data to Kaggle as a Dataset.

After running `kaggle/scripts/build_offline_data.sh` on your laptop, run this
to publish `data/train_sft.jsonl` + `data/pseudo_test.jsonl` + `data/synthetic/`
as a versioned Kaggle Dataset. The training notebook attaches it via "Add Data"
so it can train offline.

Pattern mirrors `kaggle/scripts/08_upload_adapter_dataset.py` and
`kaggle/scripts/11_upload_scripts_dataset.py`.

Usage
-----
First-time create (use any username/slug you control):
    python kaggle/scripts/upload_train_data_dataset.py \
        --slug your-username/nemotron-train-data \
        --create

Subsequent versions:
    python kaggle/scripts/upload_train_data_dataset.py \
        --slug your-username/nemotron-train-data \
        --message "v2: teacher CoTs + curriculum"

Prerequisites
-------------
- `pip install kaggle` and credentials at `~/.kaggle/kaggle.json` (chmod 600)
  or `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars.
- `kaggle/data/train_sft.jsonl` exists (run `build_offline_data.sh` first).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_KAGGLE_DIR = Path(__file__).resolve().parent.parent  # .../kaggle/
DATA_DIR = REPO_KAGGLE_DIR / "data"


def _ensure_kaggle_creds() -> None:
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
        "No Kaggle credentials found. Either put kaggle.json at "
        "~/.kaggle/kaggle.json (chmod 600), or export KAGGLE_USERNAME + "
        "KAGGLE_KEY before running this."
    )


def _stage_data(staging_dir: Path, slug: str, title: str) -> int:
    """Copy training data + synthetic shards into staging_dir."""
    train_sft = DATA_DIR / "train_sft.jsonl"
    if not train_sft.is_file():
        raise SystemExit(
            f"Missing {train_sft}. Run kaggle/scripts/build_offline_data.sh first."
        )

    files_copied = 0
    shutil.copy2(train_sft, staging_dir / train_sft.name)
    files_copied += 1

    pseudo = DATA_DIR / "pseudo_test.jsonl"
    if pseudo.is_file():
        shutil.copy2(pseudo, staging_dir / pseudo.name)
        files_copied += 1

    synth_src = DATA_DIR / "synthetic"
    if synth_src.is_dir():
        synth_dst = staging_dir / "synthetic"
        synth_dst.mkdir(parents=True, exist_ok=True)
        for p in synth_src.iterdir():
            if p.is_file() and p.suffix == ".jsonl":
                shutil.copy2(p, synth_dst / p.name)
                files_copied += 1

    reports_src = DATA_DIR / "reports"
    if reports_src.is_dir():
        reports_dst = staging_dir / "reports"
        reports_dst.mkdir(parents=True, exist_ok=True)
        for p in reports_src.iterdir():
            if p.is_file():
                shutil.copy2(p, reports_dst / p.name)
                files_copied += 1

    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": title,
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
    )

    return files_copied


def _run_kaggle(args: list[str]) -> None:
    print("$ kaggle " + " ".join(args), flush=True)
    subprocess.run(["kaggle", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload offline-built SFT training data as a Kaggle Dataset.",
    )
    parser.add_argument(
        "--slug",
        type=str,
        required=True,
        help="Kaggle dataset slug, e.g. your-username/nemotron-train-data",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Nemotron Reasoning Challenge - training data",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="First-time create (instead of versioning an existing dataset).",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="updated training data",
        help="Version notes (only used when not --create).",
    )
    args = parser.parse_args()

    _ensure_kaggle_creds()

    with tempfile.TemporaryDirectory(prefix="nemotron-train-data-") as tmp:
        staging_dir = Path(tmp)
        n = _stage_data(staging_dir, args.slug, args.title)
        print(f"Staged {n} files in {staging_dir}", flush=True)

        if args.create:
            _run_kaggle(
                [
                    "datasets",
                    "create",
                    "-p",
                    str(staging_dir),
                    "-r",
                    "zip",
                ]
            )
        else:
            _run_kaggle(
                [
                    "datasets",
                    "version",
                    "-p",
                    str(staging_dir),
                    "-r",
                    "zip",
                    "-m",
                    args.message,
                ]
            )

    print(
        "\nDone. Attach the dataset in your Kaggle training notebook via\n"
        f"  Add Data -> search '{args.slug.split('/')[-1]}' -> Add.\n"
        "Inside the notebook it appears at /kaggle/input/<slug>/."
    )


if __name__ == "__main__":
    main()
