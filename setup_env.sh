#!/usr/bin/env bash
# Create a virtual environment with everything needed for local dev (EDA, data prep, packaging, download script).
# Training requires Kaggle or Linux+CUDA; mamba-ssm, bitsandbytes are omitted on macOS.
#
# Run from project root: ./setup_env.sh   or   bash setup_env.sh

set -e
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "Using Python: $(which $PYTHON)"
$PYTHON --version

if [[ -d "$VENV_DIR" ]]; then
  echo "Removing existing $VENV_DIR ..."
  rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment at $VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"

echo "Activating and upgrading pip ..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

case "$(uname -s)" in
  Darwin)
    echo "macOS detected — installing local deps (no mamba-ssm, bitsandbytes; training on Kaggle)"
    pip install -r requirements-local.txt
    ;;
  Linux)
    echo "Linux detected — installing full requirements (including mamba-ssm, bitsandbytes)"
    pip install torch torchvision
    pip install -r requirements.txt
    pip install kagglehub
    ;;
  *)
    echo "Unknown OS — installing requirements-local.txt"
    pip install -r requirements-local.txt
    ;;
esac

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run: python run_all.py --skip-train --skip-eval   (EDA, data prep, packaging)"
echo "Or:       python scripts/download_nemotron_kagglehub.py   (on Kaggle)"
