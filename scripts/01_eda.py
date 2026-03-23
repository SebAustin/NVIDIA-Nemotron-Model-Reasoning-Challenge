#!/usr/bin/env python3
"""Phase 1: EDA for Nemotron Reasoning Challenge train/test CSVs."""

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
from transformers import AutoTokenizer

MODEL_ID = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"


def classify_puzzle_type(prompt: str) -> str:
    if not isinstance(prompt, str):
        return "other"
    t = prompt.lower()
    if re.search(r"\b(bit|binary|8-?bit|nibble|xor|rotate|complement)\b", t):
        return "bit_manipulation"
    if re.search(
        r"\b(cipher|encrypt|decrypt|caesar|vigen|substitution|substitute)\b",
        t,
    ):
        return "encryption"
    if re.search(
        r"\b(equation|polynomial|modulo|algebra|f\(|function|digit sum)\b",
        t,
    ):
        return "algebraic"
    if re.search(r"\b(sequence|term|next in|arithmetic|geometric|fibonacci)\b", t):
        return "sequence"
    return "other"


def answer_kind(answer: str) -> str:
    if answer is None or (isinstance(answer, float) and np.isnan(answer)):
        return "missing"
    s = str(answer).strip()
    if re.fullmatch(r"[01]{8}", s):
        return "binary_8bit"
    if re.fullmatch(r"[01]+", s):
        return "binary_string"
    try:
        float(s)
        return "numeric"
    except ValueError:
        pass
    if len(s) <= 32:
        return "short_text"
    return "long_text"


def token_lengths(tokenizer, texts: list[str]) -> list[int]:
    out: list[int] = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        out.append(len(ids))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for train/test CSVs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-categorized",
        type=Path,
        default=None,
        help="Default: <data-dir>/train_categorized.csv",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--tokenizer-model", type=str, default=MODEL_ID)
    parser.add_argument(
        "--assistant-budgets",
        type=int,
        nargs="*",
        default=[2048, 4096],
        help="Token budgets to add to prompt length for length warnings",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    out_cat = args.output_categorized or (data_dir / "train_categorized.csv")

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")
    train = pd.read_csv(train_path)
    if "prompt" not in train.columns:
        raise SystemExit("train.csv must contain column 'prompt'")
    if "answer" not in train.columns:
        raise SystemExit("train.csv must contain column 'answer'")

    test = None
    if test_path.is_file():
        test = pd.read_csv(test_path)
        if "prompt" not in test.columns:
            raise SystemExit("test.csv must contain column 'prompt'")

    train["puzzle_type"] = train["prompt"].map(classify_puzzle_type)
    train.to_csv(out_cat, index=False)
    print(f"Wrote categorized train: {out_cat}")

    type_counts = Counter(train["puzzle_type"])
    answer_kinds = train["answer"].map(answer_kind)
    kind_counts = Counter(answer_kinds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
    )
    train_prompt_lens = token_lengths(tokenizer, train["prompt"].tolist())
    train["prompt_tokens"] = train_prompt_lens

    lines: list[str] = []
    lines.append("# Nemotron Reasoning — EDA report\n")
    lines.append("## Puzzle pattern\n")
    lines.append(
        "Each prompt typically lists several input→output examples that illustrate a hidden "
        "rule, then asks for the output on a new input (few-shot rule induction).\n"
    )
    lines.append("## Puzzle type distribution (train)\n")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Answer kind distribution (train)\n")
    for k, v in sorted(kind_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}\n")

    pt = np.array(train_prompt_lens)
    lines.append("\n## Prompt token length (train, tokenizer)\n")
    lines.append(f"- min: {pt.min()}, max: {pt.max()}, mean: {pt.mean():.1f}, median: {np.median(pt):.1f}\n")

    alens = train["answer"].astype(str).str.len()
    lines.append("\n## Answer character length (train)\n")
    lines.append(
        f"- min: {alens.min()}, max: {alens.max()}, mean: {alens.mean():.1f}\n"
    )

    system_stub = (
        "You are an expert logical reasoning assistant. [...] \\boxed{}."
    )
    sys_tok = len(
        tokenizer(system_stub, add_special_tokens=False)["input_ids"]
    )
    lines.append("\n## Length vs max_model_len=8192 (rough)\n")
    lines.append(
        f"- Approx system stub tokens: {sys_tok} (replace with real template for production).\n"
    )
    for budget in args.assistant_budgets:
        total = np.array(train_prompt_lens) + sys_tok + budget
        over = (total > 8192).sum()
        lines.append(
            f"- prompt + system + assistant_budget={budget}: rows over 8192 tokens: {over} / {len(train)}\n"
        )

    lines.append("\n## Sample prompts by type (2–3 each)\n")
    for ptype in sorted(type_counts.keys()):
        sub = train[train["puzzle_type"] == ptype].head(3)
        lines.append(f"\n### {ptype}\n")
        for _, row in sub.iterrows():
            pid = row.get("id", "")
            lines.append(f"- id={pid}\n")
            pr = str(row["prompt"])
            preview = pr[:800] + ("…" if len(pr) > 800 else "")
            lines.append(f"  prompt preview:\n```\n{preview}\n```\n")
            lines.append(f"  answer: {row.get('answer', '')!r}\n")

    if test is not None:
        test_lens = token_lengths(tokenizer, test["prompt"].tolist())
        tt = np.array(test_lens)
        lines.append("\n## Test prompt token lengths\n")
        lines.append(f"- min: {tt.min()}, max: {tt.max()}, mean: {tt.mean():.1f}\n")

    report_path = report_dir / "eda_report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
