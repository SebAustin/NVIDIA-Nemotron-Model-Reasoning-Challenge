#!/usr/bin/env bash
# Offline install of the Nemotron training stack from pre-downloaded wheels.
# Usage: bash install_offline.sh [WHEEL_DIR]   (defaults to the script's dir)
set -euo pipefail

DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "[offline-install] using wheel dir: $DIR"

# 1) pip-installable training stack (torch/cuda are taken from Kaggle's image)
pip install --no-index --find-links="$DIR" \
    transformers peft trl datasets accelerate bitsandbytes psutil sentencepiece safetensors

# 2) torch-ABI-locked: install causal-conv1d BEFORE mamba-ssm (mamba imports it),
#    --no-deps so pip doesn't try to fetch torch/triton over the network.
shopt -s nullglob
cc=("$DIR"/causal_conv1d-*.whl)
mm=("$DIR"/mamba_ssm-*.whl)
if [[ ${#cc[@]} -eq 0 || ${#mm[@]} -eq 0 ]]; then
  echo "[offline-install] WARNING: mamba_ssm/causal_conv1d wheels missing." >&2
  echo "  Run fetch_torch_locked_wheels.py once on Kaggle (internet ON) first." >&2
else
  pip install --no-index --no-deps "${cc[@]}" "${mm[@]}"
fi

echo "[offline-install] verifying imports..."
python - <<'PY'
import importlib
for m in ("transformers","peft","trl","datasets","accelerate","bitsandbytes",
          "mamba_ssm","causal_conv1d"):
    try:
        importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
PY
echo "[offline-install] done."
