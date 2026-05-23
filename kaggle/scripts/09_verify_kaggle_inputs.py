#!/usr/bin/env python3
"""Phase 4c: Verify the Kaggle notebook environment has the inputs we expect.

Kaggle Code Competitions run with `/kaggle/input/` populated by:
  - The competition's own attached data (test.csv, sample_submission.csv).
  - The base model dataset (sometimes auto-mounted by the competition).
  - Any user-attached datasets, e.g. our LoRA adapter.

This script walks `/kaggle/input/`, identifies which of those are present,
prints concrete paths a downstream notebook should use, and exits non-zero
if a critical input is missing.

It is intentionally side-effect-free (read-only walk) so it's safe to run
inside the offline Kaggle submission environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_THIS = Path(__file__).resolve()
# Make this script's own directory importable so we can pull in its sibling
# 'kaggle_nemotron_paths.py' regardless of how the enclosing dataset is
# nested under /kaggle/input (Kaggle mounts can vary: /kaggle/input/<slug>/...
# or /kaggle/input/datasets/<owner>/<slug>/...).
_THIS_DIR = _THIS.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
# Also expose the parent of this dir so 'from scripts.* import ...' keeps
# working when the upload is wrapped in a 'scripts/' folder (the original
# layout assumed by the rest of the pipeline).
_PARENT = _THIS.parents[1] if len(_THIS.parents) >= 2 else _THIS_DIR
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from scripts.kaggle_nemotron_paths import find_kaggle_competition_nemotron_dir
except ModuleNotFoundError:
    from kaggle_nemotron_paths import find_kaggle_competition_nemotron_dir  # type: ignore[no-redef]


def _list_kaggle_input_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _looks_like_lora_adapter(d: Path) -> bool:
    cfg = d / "adapter_config.json"
    if cfg.is_file():
        return True
    # Sometimes attached datasets nest the adapter one level deeper.
    for sub in d.rglob("adapter_config.json"):
        if sub.is_file():
            return True
    return False


def _find_lora_adapter_dir(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    for top in _list_kaggle_input_dirs(root):
        if (top / "adapter_config.json").is_file():
            return top
        for sub in top.rglob("adapter_config.json"):
            return sub.parent
    return None


def _find_competition_csv(root: Path, name: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for top in _list_kaggle_input_dirs(root):
        # Most competitions place CSVs at the top level of their dataset folder.
        cand = top / name
        if cand.is_file():
            return cand
    # Fall back to a recursive search (slow but reliable).
    try:
        for hit in root.rglob(name):
            if hit.is_file():
                return hit
    except OSError:
        pass
    return None


def _read_adapter_summary(adapter_dir: Path) -> Dict[str, object]:
    summary: Dict[str, object] = {"path": str(adapter_dir)}
    cfg_path = adapter_dir / "adapter_config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
        summary["r"] = cfg.get("r")
        summary["lora_alpha"] = cfg.get("lora_alpha")
        summary["base_model"] = cfg.get("base_model_name_or_path")
        summary["target_modules"] = cfg.get("target_modules")
    except (OSError, json.JSONDecodeError) as e:
        summary["error"] = repr(e)
    weights = list(adapter_dir.glob("adapter_model.*"))
    summary["weights_files"] = [w.name for w in weights]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Kaggle dataset mounts")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--require-test-csv", action="store_true", default=True)
    parser.add_argument(
        "--no-require-test-csv",
        dest="require_test_csv",
        action="store_false",
        help="Skip the test.csv requirement (useful when running locally to debug).",
    )
    parser.add_argument("--require-base-model", action="store_true", default=True)
    parser.add_argument(
        "--no-require-base-model",
        dest="require_base_model",
        action="store_false",
    )
    parser.add_argument("--require-adapter", action="store_true", default=True)
    parser.add_argument(
        "--no-require-adapter",
        dest="require_adapter",
        action="store_false",
    )
    args = parser.parse_args()

    root: Path = args.input_root
    print(f"Scanning {root} ...")
    if not root.is_dir():
        print(f"  (note) {root} does not exist - probably running outside Kaggle.")

    print("\nTop-level mounts:")
    for d in _list_kaggle_input_dirs(root):
        print(f"  - {d.name}")

    base_model_dir = find_kaggle_competition_nemotron_dir()
    print("\n[base model]")
    if base_model_dir:
        print(f"  Found Nemotron base model at: {base_model_dir}")
    else:
        print("  Nemotron base model NOT detected under /kaggle/input.")
        print("  Action: attach the Nemotron Nano dataset via 'Add Data' in the")
        print("          notebook editor, OR upload the model with")
        print("          scripts/10_upload_base_model.py if your environment lacks it.")

    adapter_dir = _find_lora_adapter_dir(root)
    print("\n[LoRA adapter]")
    if adapter_dir is not None:
        info = _read_adapter_summary(adapter_dir)
        print(f"  Found adapter at: {adapter_dir}")
        print(f"  rank: {info.get('r')}  alpha: {info.get('lora_alpha')}")
        print(f"  base_model: {info.get('base_model')}")
        print(f"  weights: {info.get('weights_files')}")
    else:
        print("  No LoRA adapter found in /kaggle/input.")
        print("  Action: upload via scripts/08_upload_adapter_dataset.py from Colab,")
        print("          then 'Add Data' to attach the resulting dataset.")

    test_csv = _find_competition_csv(root, "test.csv")
    sample_csv = _find_competition_csv(root, "sample_submission.csv")
    print("\n[competition data]")
    print(f"  test.csv:              {test_csv if test_csv else 'NOT FOUND'}")
    print(f"  sample_submission.csv: {sample_csv if sample_csv else 'NOT FOUND'}")

    missing: List[str] = []
    if args.require_base_model and not base_model_dir:
        missing.append("base_model")
    if args.require_adapter and adapter_dir is None:
        missing.append("lora_adapter")
    if args.require_test_csv and test_csv is None:
        missing.append("test.csv")

    if missing:
        print(f"\nFAILED: required inputs missing: {', '.join(missing)}")
        sys.exit(1)

    summary = {
        "base_model_dir": base_model_dir,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "test_csv": str(test_csv) if test_csv else None,
        "sample_submission_csv": str(sample_csv) if sample_csv else None,
    }
    print("\nOK. Resolved inputs:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
