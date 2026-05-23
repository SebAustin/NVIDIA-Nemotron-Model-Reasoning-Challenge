#!/usr/bin/env bash
# Phase 1.3 + 1.4 + 1.5: build the offline training data on your laptop.
#
# Outputs (relative to repo root):
#   kaggle/data/pseudo_test.jsonl   <- solver-verified pseudo-labels on test.csv
#   kaggle/data/train_sft.jsonl     <- final SFT dataset (synthetic + solver +
#                                      template + pseudo + teacher-verified CoTs)
#   kaggle/data/synthetic/*.jsonl   <- per-family synthetic shards
#   kaggle/data/reports/*.csv       <- distribution-alignment + pseudo-label reports
#
# Backends and approx. cost for a full train.csv run:
#   deepseek    DeepSeek V3 via OpenAI-compatible endpoint     ~$1-3   <- RECOMMENDED
#   anthropic   Claude Sonnet 4                                ~$30-80
#   openai      GPT-4o (or any OpenAI-compatible URL+model)    varies
#
# Cost levers regardless of backend:
#   QUICK_TEST=1                      cap train coverage to 30 rows (~$0.01 with deepseek)
#   COT_LIMIT=<n>                     cap train coverage to <n> rows
#   COT_MAX_TOKENS=<n>                cap output tokens per CoT call (default 2048)
#   SKIP_TEACHER=1                    skip the teacher API entirely (solver+template+synthetic only)
#
# Requirements before running:
#   - kaggle/data/train.csv and kaggle/data/test.csv exist
#   - pip install -r kaggle/requirements.txt
#   - One of:
#       export DEEPSEEK_API_KEY=sk-...     (for `deepseek` backend)
#       export ANTHROPIC_API_KEY=sk-ant-...
#       export OPENAI_API_KEY=sk-...       (for `openai` backend; can also point at
#                                           any OpenAI-compatible endpoint via
#                                           OPENAI_BASE_URL=https://...)
#
# Usage examples:
#   # Recommended: DeepSeek V3, full coverage, ~$1-3
#   export DEEPSEEK_API_KEY=sk-...
#   bash kaggle/scripts/build_offline_data.sh deepseek
#
#   # Quick smoke test (~$0.01 with deepseek):
#   QUICK_TEST=1 bash kaggle/scripts/build_offline_data.sh deepseek
#
#   # Cap to 500 unsolved rows:
#   COT_LIMIT=500 bash kaggle/scripts/build_offline_data.sh deepseek
#
#   # Anthropic Claude Sonnet 4 (original, expensive):
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash kaggle/scripts/build_offline_data.sh anthropic
#
#   # No teacher at all (free; solver + template + synthetic + pseudo only):
#   SKIP_TEACHER=1 bash kaggle/scripts/build_offline_data.sh

set -euo pipefail

BACKEND="${1:-deepseek}"

MODEL_DEFAULT_ANTHROPIC="claude-sonnet-4-20250514"
MODEL_DEFAULT_OPENAI="gpt-4o-mini"
MODEL_DEFAULT_DEEPSEEK="deepseek-chat"   # DeepSeek V3; swap to `deepseek-reasoner` for R1

case "$BACKEND" in
  deepseek)
    COT_MODEL="${2:-$MODEL_DEFAULT_DEEPSEEK}"
    # DeepSeek exposes an OpenAI-compatible /chat/completions endpoint, so we
    # use the openai backend in cot_generator.py with a redirected base URL.
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com/v1}"
    # Accept either DEEPSEEK_API_KEY (preferred) or OPENAI_API_KEY (fallback).
    if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
      export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
    fi
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "ERROR: DEEPSEEK_API_KEY (or OPENAI_API_KEY) is not set." >&2
      echo "  export DEEPSEEK_API_KEY=sk-..." >&2
      exit 1
    fi
    COT_BACKEND_ARG="openai"
    ;;
  anthropic)
    COT_MODEL="${2:-$MODEL_DEFAULT_ANTHROPIC}"
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "ERROR: ANTHROPIC_API_KEY is not set." >&2
      echo "  export ANTHROPIC_API_KEY=sk-ant-..." >&2
      exit 1
    fi
    COT_BACKEND_ARG="anthropic"
    ;;
  openai)
    COT_MODEL="${2:-$MODEL_DEFAULT_OPENAI}"
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "ERROR: OPENAI_API_KEY is not set." >&2
      echo "  export OPENAI_API_KEY=sk-..." >&2
      exit 1
    fi
    COT_BACKEND_ARG="openai"
    ;;
  *)
    echo "ERROR: unknown backend '$BACKEND'. Use one of: deepseek | anthropic | openai." >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="kaggle/data"
mkdir -p "$DATA_DIR/synthetic" "$DATA_DIR/reports"

if [[ ! -f "$DATA_DIR/train.csv" ]]; then
  echo "ERROR: $DATA_DIR/train.csv missing." >&2
  echo "  Put the Kaggle competition CSVs under $DATA_DIR/ first." >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/test.csv" ]]; then
  echo "ERROR: $DATA_DIR/test.csv missing." >&2
  exit 1
fi

echo "===== Phase 1.4 - solver pseudo-labels on test.csv ====="
python3 kaggle/scripts/02b_pseudolabel_test.py \
  --test-csv "$DATA_DIR/test.csv" \
  --output   "$DATA_DIR/pseudo_test.jsonl" \
  --report-dir "$DATA_DIR/reports"

# --- Phase 1.3 args ---
COT_MAX_TOKENS="${COT_MAX_TOKENS:-2048}"      # was 4096; 2048 covers ~all CoTs and halves cost
if [[ "${QUICK_TEST:-0}" == "1" ]]; then
  COT_LIMIT="${COT_LIMIT:-30}"
fi

PREP_ARGS=(
  --data-dir "$DATA_DIR"
  --synthetic-dir "$DATA_DIR/synthetic"
  --output "$DATA_DIR/train_sft.jsonl"
  --tokenizer-model "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  --max-tokens-per-example 3840
  --max-per-type 6000
  --synthetic-per-kind 2500
  --bit 4000
  --cipher 4000
  --algebraic 500
  --sequence 500
  --curriculum
  --pseudo-label-file "$DATA_DIR/pseudo_test.jsonl"
)

if [[ "${SKIP_TEACHER:-0}" == "1" ]]; then
  echo "===== Phase 1.3 - assemble SFT JSONL (NO teacher API; solver + template + synthetic) ====="
  PREP_ARGS+=(--skip-cot)
else
  echo "===== Phase 1.3 - teacher-distilled CoTs + assemble SFT JSONL ====="
  echo "       backend=$BACKEND  model=$COT_MODEL  max_tokens=$COT_MAX_TOKENS  base_url=${OPENAI_BASE_URL:-default}"
  PREP_ARGS+=(
    --skip-template-cot
    --cot-backend "$COT_BACKEND_ARG"
    --cot-model "$COT_MODEL"
    --cot-max-tokens "$COT_MAX_TOKENS"
  )
  if [[ -n "${COT_LIMIT:-}" ]]; then
    echo "       coverage cap: --limit-train $COT_LIMIT"
    PREP_ARGS+=(--limit-train "$COT_LIMIT")
  fi
fi

python3 kaggle/scripts/02_prepare_data.py "${PREP_ARGS[@]}"

echo
echo "===== Done. Outputs: ====="
wc -l "$DATA_DIR/train_sft.jsonl"  "$DATA_DIR/pseudo_test.jsonl" \
      "$DATA_DIR"/synthetic/*.jsonl 2>/dev/null || true
echo
echo "Next: upload to Kaggle Dataset (Phase 1.7):"
echo "  python3 kaggle/scripts/upload_train_data_dataset.py \\"
echo "      --slug your-username/nemotron-train-data --create"
