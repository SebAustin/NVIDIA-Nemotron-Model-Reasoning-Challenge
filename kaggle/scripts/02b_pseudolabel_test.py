#!/usr/bin/env python3
"""Phase 2b: Solver-based pseudo-labeling on the public test.csv.

The Kaggle competition exposes the full test prompts (only labels are hidden).
For test prompts the programmatic solvers can crack, we already produce a
verified gold answer (verified against EVERY example pair in the prompt).
Those answers are added to the SFT training set.

This is "pseudo-labeling" in the Kaggle Grandmasters sense, but with a
deterministic teacher: solvers only emit answers when they verify their
hypothesis matches all in-prompt examples, so false positives are rare and
the labels are essentially gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from tqdm import tqdm

from scripts.utils.data_formatter import build_messages, format_assistant_reply
from scripts.utils.solvers import solve_puzzle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solver-distilled pseudo-labels for test.csv (extra training signal).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/test.csv"),
        help="Path to public test.csv (must contain a 'prompt' column).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pseudo_test.jsonl"),
        help="Output JSONL of solver-verified pseudo-labels.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/reports"),
        help="Where to write the coverage report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all rows; >0 = process only first N rows (debug).",
    )
    args = parser.parse_args()

    if not args.test_csv.is_file():
        raise SystemExit(f"Missing {args.test_csv}")

    df = pd.read_csv(args.test_csv)
    if "prompt" not in df.columns:
        raise SystemExit("test.csv must contain a 'prompt' column")
    if args.limit > 0:
        df = df.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    by_type: Counter = Counter()
    solved_by_type: Counter = Counter()
    records: List[Dict[str, Any]] = []
    failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pseudo-label"):
        prompt = str(row["prompt"])
        try:
            res = solve_puzzle(prompt)
        except Exception:
            res = None
            failures += 1
        if res is None:
            by_type["unsolved"] += 1
            continue
        ptype, answer, cot = res
        by_type[ptype] += 1
        solved_by_type[ptype] += 1
        assistant = format_assistant_reply(cot, answer)
        rec: Dict[str, Any] = {
            "messages": build_messages(prompt, assistant),
            "meta": {
                "puzzle_type": ptype,
                "source": "test_csv",
                "id": str(row.get("id", "")),
                "cot_source": "solver_test",
                "answer": answer,
            },
        }
        records.append(rec)

    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(df)
    solved = len(records)
    pct = 100 * solved / max(total, 1)

    lines: List[str] = []
    lines.append("# Solver pseudo-label coverage (public test.csv)\n\n")
    lines.append(f"- Test rows scanned: **{total}**\n")
    lines.append(f"- Solver-verified pseudo-labels: **{solved} ({pct:.1f}%)**\n")
    lines.append(f"- Solver exceptions: {failures}\n\n")
    lines.append("## By puzzle type\n\n")
    lines.append("| puzzle_type | solved |\n|---|---|\n")
    for k in sorted(by_type.keys()):
        lines.append(f"| {k} | {by_type[k]} |\n")
    report_path = args.report_dir / "pseudo_label_coverage.md"
    report_path.write_text("".join(lines), encoding="utf-8")

    print()
    print(f"Pseudo-labels written: {args.output} ({solved}/{total} = {pct:.1f}%)")
    for k in sorted(solved_by_type.keys()):
        print(f"  {k}: {solved_by_type[k]}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
