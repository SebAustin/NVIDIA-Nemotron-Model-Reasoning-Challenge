#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install -U pip setuptools wheel
.venv/bin/pip install -r requirements.txt

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Optional Unsloth (CUDA/Linux, may fail on macOS):"
echo "  .venv/bin/pip install -r requirements-unsloth.txt"
echo ""
echo "Mamba (Nemotron load / training) — Linux+CUDA only, not typical macOS:"
echo "  .venv/bin/pip install -r requirements-mamba.txt"
