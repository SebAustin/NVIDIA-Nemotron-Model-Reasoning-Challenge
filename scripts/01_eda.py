#!/usr/bin/env python3
"""Phase 1: EDA + family categorization for the Nemotron Reasoning Challenge.

Clusters prompts into the 6 real puzzle families (verified on train.csv), writes
``data/train_categorized.csv`` with a ``category`` column, prints a per-category
count table and 3 (prompt, answer) examples per category, and writes a markdown
report. Token-length stats are included only if ``transformers`` + the tokenizer
are available (``--no-tokenizer`` to skip; default auto-skips on failure).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from scripts.utils.solvers import FAMILIES, classify_family

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def answer_kind(answer) -> str:
    if answer is None or (isinstance(answer, float) and np.isnan(answer)):
        return "missing"
    s = str(answer).strip()
    if re.fullmatch(r"[01]{8}", s):
        return "binary_8bit"
    try:
        float(s)
        return "numeric"
    except ValueError:
        pass
    if re.fullmatch(r"[IVXLCDM]+", s):
        return "roman"
    if len(s) <= 32:
        return "short_text"
    return "long_text"


def _maybe_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[eda] tokenizer unavailable ({e!r}); skipping token-length stats.")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA for train/test CSVs")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-categorized", type=Path, default=None)
    ap.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    ap.add_argument("--tokenizer-model", type=str, default=MODEL_ID)
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    data_dir = args.data_dir
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    out_cat = args.output_categorized or (data_dir / "train_categorized.csv")

    train_path = data_dir / "train.csv"
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")
    train = pd.read_csv(train_path)
    for col in ("prompt", "answer"):
        if col not in train.columns:
            raise SystemExit(f"train.csv must contain column '{col}'")

    train["category"] = train["prompt"].map(classify_family)
    train.to_csv(out_cat, index=False)
    print(f"Wrote categorized train: {out_cat}  ({len(train)} rows)")

    cat_counts = Counter(train["category"])
    print("\n=== Per-category counts ===")
    for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:5d}  ({v/len(train):5.1%})")

    answer_kinds = train.groupby("category")["answer"].apply(
        lambda s: dict(Counter(s.map(answer_kind)))
    )

    print("\n=== 3 examples per category ===")
    for fam in FAMILIES:
        sub = train[train["category"] == fam].head(3)
        print(f"\n### {fam}")
        for _, row in sub.iterrows():
            preview = str(row["prompt"]).replace("\n", " ")[:160]
            print(f"  - answer={row['answer']!r}  prompt: {preview}…")

    # token-length stats (optional)
    tok = None if args.no_tokenizer else _maybe_tokenizer(args.tokenizer_model)
    tok_stats = ""
    if tok is not None:
        lens = [len(tok(str(p), add_special_tokens=False)["input_ids"])
                for p in train["prompt"]]
        a = np.array(lens)
        tok_stats = (f"- prompt tokens: min={a.min()} max={a.max()} "
                     f"mean={a.mean():.1f} median={np.median(a):.0f}\n")
        print("\n=== Prompt token length ===\n" + tok_stats)

    # markdown report
    lines = ["# Nemotron Reasoning — EDA report\n\n",
             "Few-shot rule-induction puzzles; the hidden rule must be inferred "
             "from input→output examples.\n\n",
             "## Per-category counts\n\n"]
    for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}**: {v} ({v/len(train):.1%})\n")
    lines.append("\n## Answer kinds by category\n\n")
    for fam in FAMILIES:
        lines.append(f"- {fam}: {answer_kinds.get(fam, {})}\n")
    if tok_stats:
        lines.append("\n## Prompt token length\n\n" + tok_stats)
    lines.append("\n## Examples\n")
    for fam in FAMILIES:
        sub = train[train["category"] == fam].head(3)
        lines.append(f"\n### {fam}\n")
        for _, row in sub.iterrows():
            lines.append(f"- answer={row['answer']!r}\n\n  ```\n  "
                         + str(row["prompt"])[:400].replace("\n", "\n  ")
                         + "\n  ```\n")
    (report_dir / "eda_report.md").write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote report: {report_dir / 'eda_report.md'}")


if __name__ == "__main__":
    main()
