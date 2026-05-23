#!/usr/bin/env python3
"""Upload (or version) the kaggle/scripts/ folder as a Kaggle Dataset.

This makes the helper modules (09_verify_kaggle_inputs.py, kaggle_nemotron_paths.py,
utils/, etc.) available at /kaggle/input/<slug>/scripts/ inside the Kaggle
inference notebook (kaggle_submission.ipynb), so it can resolve paths and run.

Usage examples
--------------
First-time create (uses the slug from dataset-metadata.json next to this script):

    python scripts/11_upload_scripts_dataset.py --create

Subsequent updates (creates a new version of the existing dataset):

    python scripts/11_upload_scripts_dataset.py --message "fix verifier paths"

Override the slug (e.g. when running under a different Kaggle username):

    python scripts/11_upload_scripts_dataset.py \
        --slug your-username/nemotron-scripts --message "v2"

Notes
-----
- Requires the `kaggle` CLI installed (`pip install kaggle`) and credentials
  available either at ~/.kaggle/kaggle.json or via KAGGLE_USERNAME/KAGGLE_KEY
  env vars.
- Only the scripts/ folder is uploaded; notebooks, data/, and large artifacts
  are intentionally excluded.
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
SCRIPTS_DIR = REPO_KAGGLE_DIR / "scripts"
DEFAULT_METADATA = REPO_KAGGLE_DIR / "dataset-metadata.json"


def _load_default_slug() -> str | None:
    if not DEFAULT_METADATA.is_file():
        return None
    try:
        return json.loads(DEFAULT_METADATA.read_text())["id"]
    except Exception:
        return None


def _stage_scripts(staging_dir: Path, slug: str) -> None:
    """Copy scripts/ into staging_dir and write dataset-metadata.json."""
    dest_scripts = staging_dir / "scripts"
    if dest_scripts.exists():
        shutil.rmtree(dest_scripts)
    shutil.copytree(
        SCRIPTS_DIR,
        dest_scripts,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": slug.split("/")[-1],
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
    )


def _run_kaggle(args: list[str]) -> None:
    print("$ kaggle " + " ".join(args), flush=True)
    res = subprocess.run(["kaggle", *args])
    if res.returncode != 0:
        raise SystemExit(f"kaggle CLI failed with exit code {res.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        default=_load_default_slug(),
        help="Dataset slug, e.g. 'username/nemotron-scripts'. Defaults to the "
        "id in kaggle/dataset-metadata.json if present.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="First-time create. Without this flag a new VERSION is pushed.",
    )
    parser.add_argument(
        "--message",
        "-m",
        default="update scripts",
        help="Version message (only used when versioning).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public on first create (default: private).",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Don't delete the staging directory after upload (for debugging).",
    )
    args = parser.parse_args()

    if not args.slug:
        parser.error(
            "no --slug given and no id found in kaggle/dataset-metadata.json"
        )
    if "/" not in args.slug:
        parser.error("--slug must look like 'username/dataset-name'")
    if not SCRIPTS_DIR.is_dir():
        parser.error(f"scripts dir not found at {SCRIPTS_DIR}")

    staging = Path(tempfile.mkdtemp(prefix="kaggle-scripts-"))
    try:
        _stage_scripts(staging, args.slug)
        print(f"Staged dataset payload at: {staging}")
        print(f"  -> scripts/ files: "
              f"{sum(1 for _ in (staging / 'scripts').rglob('*') if _.is_file())}")

        if args.create:
            cmd = ["datasets", "create", "-p", str(staging), "-r", "zip"]
            if args.public:
                cmd.append("--public")
            _run_kaggle(cmd)
        else:
            _run_kaggle(
                [
                    "datasets",
                    "version",
                    "-p",
                    str(staging),
                    "-m",
                    args.message,
                    "-r",
                    "zip",
                ]
            )
        print(f"\nDone. Dataset slug: {args.slug}")
        print("In your Kaggle inference notebook, click 'Add Data' and attach "
              f"this dataset, then re-run kaggle_submission.ipynb.")
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
