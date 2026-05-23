#!/usr/bin/env python3
"""Phase 4d: Upload the Nemotron base model to Kaggle as a Dataset.

Use this ONLY if the competition does not pre-mount the Nemotron Nano model
under `/kaggle/input/`. Most competitions about a specific model do mount it
automatically, so check first by running:

    python scripts/09_verify_kaggle_inputs.py

If the [base model] section says NOT detected, then you need to publish your
own copy as a Kaggle Dataset and attach it to the inference notebook.

What this script does:
  1. Locates the model files (HF cache snapshot or local path).
  2. Stages them with a `dataset-metadata.json`.
  3. Calls `kaggle datasets create` (or `version`).

Caveats:
  - Nemotron-3-Nano-30B-A3B-BF16 is ~60 GB on disk. Uploading from Colab
    will take ~20-40 min and uses outbound bandwidth. Run from a node
    that has the model already cached locally.
  - Kaggle Dataset size cap is generous (>100 GB), but per-file caps apply.
    The HF safetensors shards for this model are individually <5 GB so we
    are fine.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REQUIRED_FILES = ("config.json",)
_TOKENIZER_HINTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
)


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
        "No Kaggle credentials. Set KAGGLE_USERNAME + KAGGLE_KEY env vars "
        "or place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)."
    )


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")
    for name in _REQUIRED_FILES:
        if not (model_dir / name).is_file():
            raise SystemExit(
                f"Missing {name} in {model_dir}. This does not look like an "
                "HF model directory (config.json is required)."
            )
    if not any((model_dir / hint).is_file() for hint in _TOKENIZER_HINTS):
        print(
            f"Warning: no tokenizer file found in {model_dir} "
            f"(looking for any of {_TOKENIZER_HINTS}). Inference may fail."
        )


def _stage_model(
    model_dir: Path,
    staging_dir: Path,
    dataset_id: str,
    title: str,
) -> int:
    """Symlink or copy the model files into a flat staging directory.

    Symlinking is preferred (saves disk + time) but Kaggle CLI follows symlinks
    so the upload still gets the actual bytes. We fall back to copying when
    symlink isn't possible (e.g. Windows or cross-device).
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    files = sorted(p for p in model_dir.iterdir() if p.is_file())
    use_symlink = True
    for src in files:
        dst = staging_dir / src.name
        try:
            if use_symlink:
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
        except OSError:
            use_symlink = False
            shutil.copy2(src, dst)

    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    method = "symlinked" if use_symlink else "copied"
    print(f"Staged {len(files)} file(s) ({method}) into {staging_dir}")
    return len(files)


def _run_kaggle(args_list: list[str]) -> None:
    print("$ kaggle " + " ".join(args_list))
    rc = subprocess.run(["kaggle"] + args_list).returncode
    if rc != 0:
        raise SystemExit(f"Kaggle CLI exit code {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Nemotron base model to Kaggle")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local HF model directory (containing config.json + safetensors shards).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Kaggle dataset slug, e.g. 'username/nemotron-3-nano-30b-bf16'.",
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--first-time",
        action="store_true",
        help="Use 'datasets create' instead of 'datasets version'.",
    )
    parser.add_argument(
        "--version-notes",
        type=str,
        default="initial upload",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path(".kaggle_base_model_staging"),
    )
    args = parser.parse_args()

    if "/" not in args.dataset_id or args.dataset_id.count("/") != 1:
        raise SystemExit("--dataset-id must look like 'owner/slug'")
    _ensure_kaggle_creds()
    _validate_model_dir(args.model_dir)

    title = args.title or args.dataset_id.split("/", 1)[1].replace("-", " ").title()
    _stage_model(args.model_dir, args.staging_dir, args.dataset_id, title)

    if args.first_time:
        _run_kaggle(["datasets", "create", "-p", str(args.staging_dir), "--dir-mode", "zip"])
    else:
        _run_kaggle(
            [
                "datasets",
                "version",
                "-p",
                str(args.staging_dir),
                "-m",
                args.version_notes,
                "--dir-mode",
                "zip",
            ]
        )

    print(
        f"\nDone. Attach this dataset to your inference notebook by ID: "
        f"{args.dataset_id}"
    )
    print(f"Page: https://www.kaggle.com/datasets/{args.dataset_id}")


if __name__ == "__main__":
    main()
