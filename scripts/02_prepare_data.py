#!/usr/bin/env python3
"""Phase 2: Build data/train_sft.jsonl (no external API).

Pipeline:
  * Real train rows -> family from category. For solvable families we attach a
    solver-VERIFIED reasoning trace (rule stated, applied, answer). Hard-family
    rows (bit_manipulation, equation) and any solver miss become direct-answer
    examples anchored on the gold answer.
  * Synthetic puzzles (puzzle_generator) add reasoning traces with KNOWN rules
    for all 6 families; hard families are upweighted.
  * Enforce a ~75% reasoning / 25% direct-answer mix (preserve base reasoning),
    dedup, optional token-length filter, shuffle, write JSONL of {messages,meta}.

Each example is verified consistent with the gold answer before inclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.utils.cot_generator import build_reasoning
from scripts.utils.competition_metric import scores
from scripts.utils.data_formatter import build_messages, format_assistant_reply
from scripts.utils.puzzle_generator import write_all_synthetic
from scripts.utils.solvers import (
    FAMILIES,
    build_encryption_vocab,
    classify_family,
    solve,
)

# hard families to upweight (their real rows aren't deterministically solvable)
HARD = {"bit_manipulation", "equation"}


def _fingerprint(messages: List[Dict[str, str]]) -> str:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return hashlib.sha1(re.sub(r"\s+", " ", user.strip().lower()).encode()).hexdigest()


def _example(prompt: str, answer: str, reasoning: Optional[str], category: str,
             source: str, rid: str = "") -> Dict[str, Any]:
    assistant = format_assistant_reply(reasoning or "", answer)
    return {
        "messages": build_messages(prompt, assistant),
        "meta": {"category": category, "source": source, "id": rid,
                 "reasoning": bool(reasoning), "answer": answer},
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _char_len(messages: List[Dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages)


def build_real_examples(df: pd.DataFrame, vocab: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stats = Counter()
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"]).strip()
        fam = str(row.get("category") or classify_family(prompt))
        reasoning = None
        if fam not in HARD:
            pred = solve(prompt, fam, vocab=vocab)
            if pred is not None and scores(answer, str(pred)):
                reasoning = build_reasoning(prompt, fam, answer)
        out.append(_example(prompt, answer, reasoning, fam, "train_csv",
                            str(row.get("id", ""))))
        stats[(fam, "reasoning" if reasoning else "direct")] += 1
    for fam in FAMILIES:
        r = stats[(fam, "reasoning")]
        d = stats[(fam, "direct")]
        print(f"  real {fam:18s} reasoning={r:5d} direct={d:5d}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SFT JSONL")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--train-categorized", type=Path, default=None)
    ap.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--output", type=Path, default=Path("data/train_sft.jsonl"))
    ap.add_argument("--synthetic-per-kind", type=int, default=300)
    ap.add_argument("--hard-multiplier", type=float, default=3.0,
                    help="Synthetic upweight factor for hard families.")
    ap.add_argument("--direct-ratio", type=float, default=0.25,
                    help="Target fraction of direct-answer (no-reasoning) examples.")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="Drop examples whose rendered messages exceed this length.")
    ap.add_argument("--limit-train", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = args.data_dir
    train_cat = args.train_categorized or (data_dir / "train_categorized.csv")
    src = train_cat if train_cat.is_file() else (data_dir / "train.csv")
    df = pd.read_csv(src)
    if "category" not in df.columns:
        df["category"] = df["prompt"].map(classify_family)
    if args.limit_train > 0:
        df = df.head(args.limit_train)
    print(f"Loaded {len(df)} real rows from {src}")

    vocab = build_encryption_vocab(
        df[df.category == "encryption"]["prompt"].tolist()
    )

    print("Building real examples...")
    real = build_real_examples(df, vocab)

    print("Generating synthetic examples...")
    overrides = {f: int(args.synthetic_per_kind * args.hard_multiplier) for f in HARD}
    paths = write_all_synthetic(args.synthetic_dir, per_kind=args.synthetic_per_kind,
                                seed=args.seed, per_kind_overrides=overrides)
    synthetic: List[Dict[str, Any]] = []
    for p in paths.values():
        synthetic.extend(_load_jsonl(p))
    print(f"  synthetic total={len(synthetic)}")

    # dedup + length filter
    merged = real + synthetic
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    for r in merged:
        fp = _fingerprint(r["messages"])
        if fp in seen:
            continue
        if _char_len(r["messages"]) > args.max_chars:
            continue
        seen.add(fp)
        kept.append(r)

    # enforce reasoning/direct ratio by downsampling the over-represented side
    reasoning = [r for r in kept if r["meta"]["reasoning"]]
    direct = [r for r in kept if not r["meta"]["reasoning"]]
    target_direct_frac = args.direct_ratio
    total = len(reasoning) + len(direct)
    max_direct = int(target_direct_frac * total)
    if len(direct) > max_direct:
        rng.shuffle(direct)
        direct = direct[:max_direct]
    # if too few direct, convert a slice of reasoning rows to direct-answer
    elif len(direct) < max_direct:
        need = max_direct - len(direct)
        rng.shuffle(reasoning)
        convert, reasoning = reasoning[:need], reasoning[need:]
        for r in convert:
            ans = r["meta"]["answer"]
            r["messages"][-1]["content"] = format_assistant_reply("", ans)
            r["meta"]["reasoning"] = False
            direct.append(r)

    final = reasoning + direct
    rng.shuffle(final)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat = Counter(r["meta"]["category"] for r in final)
    print(f"\nWrote {args.output} ({len(final)} rows)")
    print(f"  reasoning={len(reasoning)}  direct={len(direct)}  "
          f"({len(direct)/max(1,len(final)):.1%} direct)")
    print("  by category:", dict(by_cat))


if __name__ == "__main__":
    main()
