#!/usr/bin/env bash
# Run the Nemotron pipeline locally (EDA, data prep, package). Training requires GPU — run that on Kaggle.
# From project root: ./run_local.sh   or   bash run_local.sh

set -e
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
DATA_DIR="data"

echo "=== Local Nemotron Pipeline ==="

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Install (use lighter deps for CPU/macOS; no mamba-ssm, bitsandbytes, unsloth)
if ! python -c "import transformers" 2>/dev/null; then
    echo "Installing requirements-local.txt ..."
    pip install -q -r requirements-local.txt
fi

# Check data
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/train.csv" ] || [ ! -f "$DATA_DIR/test.csv" ]; then
    echo ""
    echo "Missing data: add train.csv and test.csv to $DATA_DIR/"
    echo "  Download from: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data"
    echo ""
    exit 1
fi

# Run pipeline (skip train + eval — need GPU; skip package if you prefer)
echo ""
echo "Running: EDA → data prep (synthetic only) → package"
python run_all.py --synthetic-only --skip-train --skip-eval
echo ""
echo "Done. submission.zip ready (uses synthetic data only)."
echo "For full training, run on Kaggle via kaggle_notebook.ipynb"
