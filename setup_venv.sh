#!/usr/bin/env bash
# Create a Python virtual environment and install all dependencies for the Nemotron Reasoning Challenge.
# Run from project root: ./setup_venv.sh   or   bash setup_venv.sh

set -e
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "Using Python: $(which $PYTHON)"
$PYTHON --version

echo "Creating virtual environment at $VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"

echo "Activating and upgrading pip ..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "Installing requirements ..."
pip install -r requirements.txt

echo "Done. Activate with: source $VENV_DIR/bin/activate"
