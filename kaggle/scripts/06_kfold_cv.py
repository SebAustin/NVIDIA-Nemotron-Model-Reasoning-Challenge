#!/usr/bin/env python3
"""Phase 6: K-fold cross-validation harness for the LoRA SFT pipeline.

Per Kaggle Grandmasters Playbook tip #0: 'If you can't trust your validation
score, you're flying blind.' This script:

1. Splits real `train.csv` into K folds (StratifiedKFold over puzzle_type).
2. For each fold:
   a. Builds a per-fold `train_sft.jsonl` (synth + solver/api/template + pseudo
      from the K-1 training folds only).
   b. Calls `scripts/03_train_lora.py` to train a fold-specific LoRA adapter.
   c. Calls `scripts/04_evaluate.py` over the held-out fold and parses accuracy
      out of stdout.
3. Aggregates the K accuracies into a CV summary.

This is expensive (K full LoRA training runs). A typical Colab Pro session can
run K=3 in 6-8h. Use --folds 3 (default) or --folds 5 if you have budget.

Outputs:
- `kfold_runs/fold_{i}/lora_adapter/` per fold
- `kfold_runs/fold_{i}/eval_log.txt` per fold
- `kfold_runs/cv_summary.json` aggregated metrics
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from scripts.utils.data_formatter import build_messages, format_assistant_reply
from scripts.utils.solvers import solve_puzzle


def _stratified_kfold_indices(
    df: pd.DataFrame,
    k: int,
    seed: int,
    strat_col: str = "puzzle_type",
) -> List[np.ndarray]:
    """Return K disjoint index arrays stratified by `strat_col`.

    No sklearn dependency: shuffle each stratum and round-robin into folds.
    """
    rng = np.random.RandomState(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    if strat_col not in df.columns:
        df = df.assign(**{strat_col: "other"})
    for _, group in df.groupby(strat_col):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        for i, gi in enumerate(idx):
            folds[i % k].append(int(gi))
    return [np.array(sorted(f)) for f in folds]


def _build_fold_jsonl(
    train_df: pd.DataFrame,
    train_sft_full: List[Dict[str, Any]],
    fold_test_ids: set[str],
    out_path: Path,
) -> int:
    """Write a per-fold train_sft.jsonl that excludes records derived from the
    held-out fold's prompt IDs (so we don't leak validation labels)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in train_sft_full:
            meta = rec.get("meta") or {}
            rec_id = str(meta.get("id", ""))
            if rec_id and rec_id in fold_test_ids:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def _parse_eval_accuracy(stdout_text: str) -> float | None:
    """Pull 'Accuracy: 0.7421 (...)' off the eval script's stdout."""
    m = re.search(r"Accuracy:\s*([0-9.]+)", stdout_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold CV runner")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument(
        "--train-sft",
        type=Path,
        default=Path("data/train_sft.jsonl"),
        help="Master SFT jsonl produced by 02_prepare_data.py.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("kfold_runs"))
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local path or HF id of the base model.",
    )
    parser.add_argument("--lora-target-mode", type=str, default="kaggle_nemotron")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train all folds without running 04_evaluate.py (e.g. when vLLM is unavailable).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="0 = use entire held-out fold for eval; >0 = subsample.",
    )
    args = parser.parse_args()

    train_csv = args.train_csv or (args.data_dir / "train_categorized.csv")
    if not train_csv.is_file():
        train_csv = args.data_dir / "train.csv"
    if not train_csv.is_file():
        raise SystemExit(f"No training CSV at {train_csv}")
    if not args.train_sft.is_file():
        raise SystemExit(
            f"Missing master SFT jsonl at {args.train_sft}. Run 02_prepare_data.py first."
        )

    df = pd.read_csv(train_csv).reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    if "puzzle_type" not in df.columns:
        df["puzzle_type"] = "other"

    # Load the master SFT jsonl once.
    sft_records: List[Dict[str, Any]] = []
    with args.train_sft.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sft_records.append(json.loads(line))

    fold_indices = _stratified_kfold_indices(df, args.folds, args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    fold_results: List[Dict[str, Any]] = []
    for fi in range(args.folds):
        fold_dir = args.work_dir / f"fold_{fi}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        val_idx = fold_indices[fi]
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Held-out prompt IDs (strings) so we can scrub them from the SFT jsonl.
        val_ids = {str(x) for x in val_df["id"].astype(str).tolist()}

        fold_sft = fold_dir / "train_sft.jsonl"
        kept = _build_fold_jsonl(df, sft_records, val_ids, fold_sft)

        # Per-fold val CSV: 04_evaluate.py treats it as `train_categorized.csv`
        # and uses --val-fraction 1.0 to evaluate every row.
        val_csv = fold_dir / "val.csv"
        val_df.to_csv(val_csv, index=False)
        adapter_dir = fold_dir / "lora_adapter"
        ckpt_dir = fold_dir / "lora_output"

        print(f"\n=== Fold {fi+1}/{args.folds} ===")
        print(f"  train_sft: {fold_sft} ({kept} rows)")
        print(f"  val rows:  {len(val_df)}")
        print(f"  adapter:   {adapter_dir}")

        train_cmd = [
            sys.executable,
            "scripts/03_train_lora.py",
            "--data-path", str(fold_sft),
            "--output-dir", str(adapter_dir),
            "--checkpoint-dir", str(ckpt_dir),
            "--model-path", args.model_path,
            "--lora-target-mode", args.lora_target_mode,
            "--lora-alpha", str(args.lora_alpha),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--max-seq-length", str(args.max_seq_length),
            "--seed", str(args.seed + fi),
            "--force-peft",
            "--no-nemotron-kaggle-patches",
            "--dataloader-workers", "0",
        ]
        print("  TRAIN:", " ".join(train_cmd))
        train_log = fold_dir / "train_log.txt"
        with train_log.open("w") as logf:
            rc = subprocess.run(train_cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
        print(f"  train rc={rc} (log: {train_log})")
        if rc != 0:
            fold_results.append({"fold": fi, "status": "train_failed", "accuracy": None})
            continue

        if args.skip_eval:
            fold_results.append({"fold": fi, "status": "trained_only", "accuracy": None})
            continue

        eval_cmd = [
            sys.executable,
            "scripts/04_evaluate.py",
            "--adapter-path", str(adapter_dir),
            "--data-dir", str(args.data_dir),
            "--train-categorized", str(val_csv),
            "--val-fraction", "1.0",
            "--seed", str(args.seed),
            "--report-dir", str(fold_dir),
        ]
        if args.max_eval_samples > 0:
            eval_cmd += ["--max-samples", str(args.max_eval_samples)]
        print("  EVAL:", " ".join(eval_cmd))
        eval_log = fold_dir / "eval_log.txt"
        eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True)
        eval_log.write_text(eval_proc.stdout + "\n--- stderr ---\n" + eval_proc.stderr)
        acc = _parse_eval_accuracy(eval_proc.stdout)
        print(f"  eval rc={eval_proc.returncode} accuracy={acc}")
        fold_results.append(
            {
                "fold": fi,
                "status": "ok" if eval_proc.returncode == 0 else "eval_failed",
                "accuracy": acc,
                "n_val": int(len(val_df)),
            }
        )

    accs = [r["accuracy"] for r in fold_results if r.get("accuracy") is not None]
    summary: Dict[str, Any] = {
        "folds": fold_results,
        "n_folds_with_acc": len(accs),
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "std_accuracy": float(np.std(accs)) if accs else None,
        "min_accuracy": float(np.min(accs)) if accs else None,
        "max_accuracy": float(np.max(accs)) if accs else None,
    }
    summary_path = args.work_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== CV summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
