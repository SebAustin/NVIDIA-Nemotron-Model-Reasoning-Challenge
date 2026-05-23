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
from typing import Any, Dict, Iterable, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.cot_generator import generate_cot_with_verification, generate_template_cot
from scripts.utils.data_formatter import build_messages, format_assistant_reply
from scripts.utils.puzzle_generator import write_all_synthetic
from scripts.utils.solvers import solve_puzzle

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


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
        "--skip-template-cot",
        action="store_true",
        default=False,
        help="Do NOT include real train.csv rows with template (non-API) CoTs. "
             "Recommended when using GRPO in Stage 2: template CoTs are low quality and "
             "pollute training with weak reasoning chains. Default: False (include them).",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Alias for --skip-cot + --skip-template-cot (compat with older notebooks).",
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
    parser.add_argument(
        "--pseudo-label-file",
        type=Path,
        default=None,
        help="Optional JSONL of solver-distilled pseudo-labels from 02b_pseudolabel_test.py.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/test.csv"),
        help="Public test CSV used only for distribution-alignment reporting.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Sort the final JSONL ascending by token count (easy -> hard) and "
            "with solver-distilled CoTs ranked before template / synthetic ones. "
            "Helps a 30B reasoner converge cleaner inside a single epoch."
        ),
    )
    args = parser.parse_args()

    if args.synthetic_only:
        args.skip_cot = True
        args.skip_template_cot = True
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
    real_records: List[Dict[str, Any]] = []
    solver_records: List[Dict[str, Any]] = []

    # Locate the train CSV (categorized preferred, raw fallback)
    train_csv_path = train_cat if train_cat.is_file() else (data_dir / "train.csv")

    # ------------------------------------------------------------
    # Pass 1: programmatic solver on real train.csv (highest CoT quality).
    # Track which row indices were solved so we don't reprocess them
    # in the API or template passes below.
    # ------------------------------------------------------------
    solved_indices: set[int] = set()
    if train_csv_path.is_file():
        df_solver = pd.read_csv(train_csv_path)
        if "puzzle_type" not in df_solver.columns:
            df_solver["puzzle_type"] = "other"
        if args.limit_train > 0:
            df_solver_iter = df_solver.head(args.limit_train)
        else:
            df_solver_iter = df_solver
        for idx, row in tqdm(df_solver_iter.iterrows(), total=len(df_solver_iter), desc="Solver"):
            prompt = str(row["prompt"])
            gold = str(row["answer"]).strip()
            if not gold or gold == "nan":
                continue
            try:
                res_s = solve_puzzle(prompt)
            except Exception:
                res_s = None
            if res_s is None:
                continue
            ptype, ans, cot_text = res_s
            if not answers_match(gold, ans):
                continue
            puzzle_type = str(row.get("puzzle_type") or ptype)
            rec = row_from_csv(row, cot_text, gold, puzzle_type)
            rec["meta"]["cot_source"] = "solver"
            solver_records.append(rec)
            solved_indices.add(int(idx))
        print(f"Solver-verified real examples: {len(solver_records)}")

    if not args.skip_cot:
        if not train_csv_path.is_file():
            raise SystemExit(
                f"Missing {train_csv_path}. Run 01_eda.py first or pass --skip-cot."
            )
        df = pd.read_csv(train_csv_path)
        if args.limit_train > 0:
            df = df.head(args.limit_train)
        # Skip rows already solved in Pass 1
        df_unsolved = df[~df.index.isin(solved_indices)]
        for _, row in tqdm(df_unsolved.iterrows(), total=len(df_unsolved), desc="CoT (API)"):
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
            rec["meta"]["cot_source"] = "api"
            api_records.append(rec)
        print(f"API-verified examples: {len(api_records)}")
    else:
        print("Skipping API CoT (--skip-cot).")

    # Include real train.csv with template CoTs ONLY for rows the solver+API didn't cover.
    if train_csv_path.is_file() and not getattr(args, "skip_template_cot", False):
        df_real = pd.read_csv(train_csv_path)
        if "puzzle_type" not in df_real.columns:
            df_real["puzzle_type"] = "other"
        already_covered = solved_indices | {
            int(i) for i, r in df_real.iterrows()
            if any(rec["meta"].get("id") == str(r.get("id", "")) for rec in api_records)
        }
        df_remaining = df_real[~df_real.index.isin(already_covered)]
        added = 0
        for _, row in tqdm(df_remaining.iterrows(), total=len(df_remaining), desc="Template CoT"):
            prompt = str(row["prompt"])
            gold = str(row["answer"]).strip()
            if not gold or gold == "nan":
                continue
            puzzle_type = str(row.get("puzzle_type", "other"))
            cot_text = generate_template_cot(prompt, gold)
            rec = row_from_csv(row, cot_text, gold, puzzle_type)
            rec["meta"]["cot_source"] = "template"
            real_records.append(rec)
            added += 1
        print(f"Real train examples (template CoT): {added}")
    elif getattr(args, "skip_template_cot", False):
        print("Skipping template CoT for real train.csv (--skip-template-cot). GRPO will handle real data.")
    else:
        print(f"No train CSV found at {train_csv_path}, skipping real data.")

    # Optional: solver-distilled pseudo-labels from public test.csv (02b output).
    pseudo_records: List[Dict[str, Any]] = []
    if args.pseudo_label_file is not None and args.pseudo_label_file.is_file():
        pseudo_records = load_jsonl(args.pseudo_label_file)
        print(f"Loaded pseudo-labels (solver on test.csv): {len(pseudo_records)}")
    elif args.pseudo_label_file is not None:
        print(f"Pseudo-label file not found at {args.pseudo_label_file}; skipping.")

    print(
        f"\nCoT source breakdown -- "
        f"solver: {len(solver_records)}, api: {len(api_records)}, "
        f"template: {len(real_records)}, pseudo_test: {len(pseudo_records)}"
    )

    # Train vs synthetic vs test puzzle-type distribution alignment check.
    def _classify(prompt: str) -> str:
        # Lightweight classifier consistent with 01_eda.classify_puzzle_type
        t = prompt.lower()
        if re.search(r"\b(bit|binary|8-?bit|nibble|xor|rotate|complement)\b", t):
            return "bit_manipulation"
        if re.search(r"\b(cipher|encrypt|decrypt|caesar|vigen|substitution|substitute)\b", t):
            return "text_cipher"
        if re.search(r"\b(equation|polynomial|modulo|algebra|f\(|function|digit sum)\b", t):
            return "algebraic"
        if re.search(r"\b(sequence|term|next in|arithmetic|geometric|fibonacci)\b", t):
            return "sequence"
        return "other"

    def _user_prompt(rec: Dict[str, Any]) -> str:
        for m in rec.get("messages", []):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    synth_counts = Counter(_classify(_user_prompt(r)) for r in synthetic_records)
    real_pool = solver_records + api_records + real_records
    real_counts = Counter(_classify(_user_prompt(r)) for r in real_pool)
    test_counts: Counter = Counter()
    if args.test_csv.is_file():
        df_test = pd.read_csv(args.test_csv)
        if "prompt" in df_test.columns:
            test_counts = Counter(_classify(str(p)) for p in df_test["prompt"])

    all_types = sorted(set(synth_counts) | set(real_counts) | set(test_counts))
    if all_types:
        print("\nDistribution alignment (puzzle_type):")
        synth_total = sum(synth_counts.values()) or 1
        real_total = sum(real_counts.values()) or 1
        test_total = sum(test_counts.values()) or 1
        header = f"{'type':<18s} {'synth':>10s} {'real':>10s} {'test':>10s}   {'shift_test_vs_train':>22s}"
        print(header)
        print("-" * len(header))
        for t in all_types:
            s_pct = 100 * synth_counts.get(t, 0) / synth_total
            r_pct = 100 * real_counts.get(t, 0) / real_total
            te_pct = 100 * test_counts.get(t, 0) / test_total
            # Compare test against the training pool the model actually sees (synth + real)
            train_pct = 100 * (synth_counts.get(t, 0) + real_counts.get(t, 0)) / (synth_total + real_total)
            delta = te_pct - train_pct
            flag = "  <-- mismatch" if abs(delta) >= 10 else ""
            print(
                f"{t:<18s} {s_pct:>9.1f}% {r_pct:>9.1f}% {te_pct:>9.1f}%   "
                f"{delta:>+21.1f}%{flag}"
            )

    merged = synthetic_records + solver_records + api_records + real_records + pseudo_records
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
            r.setdefault("meta", {})["_token_count"] = ntok
            filtered.append(r)

    balanced = balance_records(filtered, args.max_per_type)

    if args.curriculum:
        # Easy -> hard within one epoch. Primary: cot_source priority (solver
        # CoTs are cleanest and fastest), secondary: token count ascending.
        source_priority = {
            "solver": 0,
            "api": 1,
            "template": 2,
            "synthetic": 3,
            "pseudo": 4,
        }

        def _curric_key(rec: Dict[str, Any]) -> Tuple[int, int]:
            meta = rec.get("meta") or {}
            src = str(meta.get("cot_source") or meta.get("source") or "synthetic")
            # Synthetic records use meta.source = synthetic kind; treat all those
            # as synthetic for ordering.
            if src not in source_priority:
                src = "synthetic"
            return source_priority[src], int(meta.get("_token_count", 0))

        balanced = sorted(balanced, key=_curric_key)
        print(
            "Curriculum on: ordered by (cot_source priority, token count). "
            f"First 5 sources: "
            f"{[(b.get('meta') or {}).get('cot_source') for b in balanced[:5]]}"
        )

    # Strip the helper key before writing so the output JSONL stays clean.
    for r in balanced:
        meta = r.get("meta")
        if isinstance(meta, dict):
            meta.pop("_token_count", None)

    counts = Counter(infer_puzzle_type_from_meta(x) for x in balanced)
    print("Final counts by type:", dict(counts))
    save_jsonl(args.output, balanced)
    print(f"Wrote {args.output} ({len(balanced)} rows)")


if __name__ == "__main__":
    main()
