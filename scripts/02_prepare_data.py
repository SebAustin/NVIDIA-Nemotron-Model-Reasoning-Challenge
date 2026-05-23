#!/usr/bin/env python3
"""Phase 2: Build train_sft.jsonl from train CSV (optional API CoT) + synthetic shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.cot_generator import generate_cot_with_verification
from scripts.utils.data_formatter import build_messages, format_assistant_reply
from scripts.utils.puzzle_generator import write_all_synthetic

MODEL_ID = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"


def normalize_prompt_key(user_text: str) -> str:
    t = user_text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def fingerprint_record(messages: List[Dict[str, str]]) -> str:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    h = hashlib.sha1(normalize_prompt_key(user).encode("utf-8")).hexdigest()
    return h


def count_tokens_chat(tokenizer, messages: List[Dict[str, str]]) -> int:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def infer_puzzle_type_from_meta(rec: Dict[str, Any]) -> str:
    meta = rec.get("meta") or {}
    if isinstance(meta, dict) and meta.get("puzzle_type"):
        return str(meta["puzzle_type"])
    user = next(
        (m["content"] for m in rec["messages"] if m["role"] == "user"),
        "",
    )
    t = user.lower()
    if "bit manipulation" in t or "binary string" in t:
        return "bit_manipulation"
    if "cipher" in t or "library" in t:
        return "text_cipher"
    if "sequence" in t or "next term" in t:
        return "sequence"
    if "f(" in t or "numeric puzzle" in t:
        return "algebraic"
    return "other"


def balance_records(
    records: List[Dict[str, Any]],
    max_per_type: int,
) -> List[Dict[str, Any]]:
    rng_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        ptype = infer_puzzle_type_from_meta(r)
        rng_groups[ptype].append(r)
    out: List[Dict[str, Any]] = []
    for ptype, items in rng_groups.items():
        items = items[:max_per_type]
        out.extend(items)
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row_from_csv(
    row: pd.Series,
    cot_text: str,
    gold: str,
    puzzle_type: str,
) -> Dict[str, Any]:
    prompt = str(row["prompt"])
    assistant = format_assistant_reply(cot_text, str(gold).strip())
    rec: Dict[str, Any] = {
        "messages": build_messages(prompt, assistant),
        "meta": {
            "puzzle_type": puzzle_type,
            "source": "train_csv",
            "id": str(row.get("id", "")),
        },
    }
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-categorized",
        type=Path,
        default=None,
        help="Default: <data-dir>/train_categorized.csv",
    )
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("data/train_sft.jsonl"))
    parser.add_argument("--tokenizer-model", type=str, default=MODEL_ID)
    parser.add_argument("--max-tokens-per-example", type=int, default=7000)
    parser.add_argument("--max-per-type", type=int, default=2000)
    parser.add_argument("--synthetic-per-kind", type=int, default=250)
    parser.add_argument("--skip-cot", action="store_true", help="Skip API CoT on real train")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Alias for --skip-cot (compat with older Kaggle notebooks).",
    )
    parser.add_argument(
        "--max-cot",
        type=int,
        default=None,
        help="Alias for --limit-train (max train rows when using API CoT).",
    )
    parser.add_argument(
        "--bit",
        type=int,
        default=None,
        help="Synthetic count for bit_manipulation (default: --synthetic-per-kind).",
    )
    parser.add_argument(
        "--cipher",
        type=int,
        default=None,
        help="Synthetic count for text_cipher.",
    )
    parser.add_argument(
        "--algebraic",
        type=int,
        default=None,
        help="Synthetic count for algebraic.",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=None,
        help="Synthetic count for sequence.",
    )
    parser.add_argument(
        "--cot-backend",
        choices=["anthropic", "openai"],
        default="anthropic",
    )
    parser.add_argument(
        "--cot-model",
        type=str,
        default="claude-sonnet-4-20250514",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=4096)
    parser.add_argument("--limit-train", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()

    if args.synthetic_only:
        args.skip_cot = True
    if args.max_cot is not None:
        args.limit_train = args.max_cot

    if not args.skip_cot:
        env_key = (
            "ANTHROPIC_API_KEY"
            if args.cot_backend == "anthropic"
            else "OPENAI_API_KEY"
        )
        if not os.environ.get(env_key):
            raise SystemExit(
                f"Missing {env_key}. Set it or use --skip-cot for synthetic-only data."
            )

    data_dir = args.data_dir
    train_cat = args.train_categorized or (data_dir / "train_categorized.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
    )

    overrides: Dict[str, int] = {}
    if args.bit is not None:
        overrides["bit_manipulation"] = args.bit
    if args.cipher is not None:
        overrides["text_cipher"] = args.cipher
    if args.algebraic is not None:
        overrides["algebraic"] = args.algebraic
    if args.sequence is not None:
        overrides["sequence"] = args.sequence

    paths = write_all_synthetic(
        args.synthetic_dir,
        per_kind=args.synthetic_per_kind,
        seed=42,
        per_kind_overrides=overrides or None,
    )
    synthetic_records: List[Dict[str, Any]] = []
    for p in paths.values():
        if p.is_file():
            synthetic_records.extend(load_jsonl(p))
    print(f"Synthetic examples: {len(synthetic_records)}")

    api_records: List[Dict[str, Any]] = []
    if not args.skip_cot:
        if not train_cat.is_file():
            raise SystemExit(
                f"Missing {train_cat}. Run 01_eda.py first or pass --skip-cot."
            )
        df = pd.read_csv(train_cat)
        if args.limit_train > 0:
            df = df.head(args.limit_train)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CoT (API)"):
            gold = str(row["answer"]).strip()
            puzzle_type = str(row.get("puzzle_type", "other"))
            try:
                res = generate_cot_with_verification(
                    str(row["prompt"]),
                    gold,
                    backend=args.cot_backend,
                    model=args.cot_model,
                    max_tokens=args.cot_max_tokens,
                )
            except Exception as e:
                print(f"CoT row failed (id={row.get('id')}): {e!r}")
                continue
            if res.extracted is None or not answers_match(gold, res.extracted):
                continue
            cot_only = res.raw_text
            if "\\boxed" in cot_only:
                cot_only = cot_only.split("\\boxed")[0].rstrip()
            rec = row_from_csv(row, cot_only, gold, puzzle_type)
            api_records.append(rec)
        print(f"API-verified examples: {len(api_records)}")
    else:
        print("Skipping API CoT (--skip-cot).")

    merged = synthetic_records + api_records
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for r in merged:
        msgs = r.get("messages")
        if not isinstance(msgs, list):
            continue
        fp = fingerprint_record(msgs)
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(r)

    filtered: List[Dict[str, Any]] = []
    for r in tqdm(deduped, desc="Token filter"):
        ntok = count_tokens_chat(tokenizer, r["messages"])
        if ntok <= args.max_tokens_per_example:
            filtered.append(r)

    balanced = balance_records(filtered, args.max_per_type)
    counts = Counter(infer_puzzle_type_from_meta(x) for x in balanced)
    print("Final counts by type:", dict(counts))
    save_jsonl(args.output, balanced)
    print(f"Wrote {args.output} ({len(balanced)} rows)")


if __name__ == "__main__":
    main()
