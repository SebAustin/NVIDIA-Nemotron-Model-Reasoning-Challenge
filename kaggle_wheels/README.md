# Offline wheels for the Nemotron training run (Kaggle, no internet)

Pre-downloaded Python wheels so `scripts/03_train_lora.py` can install its deps on a
Kaggle GPU notebook with **internet OFF** (as required at submission time).

## Platform assumptions
- Kaggle image = **Linux x86_64, Python 3.12** (its image now tracks Colab's).
- Wheels here are `cp312` / `manylinux` / `py3-none`. They do **not** include
  `torch` or the CUDA runtime — Kaggle already ships those, and we must match
  Kaggle's torch (not override it).

## What's here
- The pip-installable training stack (version-tolerant, pre-fetched from a Mac):
  `transformers (<5)`, `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`
  (0.48.x, with CUDA binaries), `psutil`, `safetensors`, `sentencepiece`, and their
  pure/`manylinux` dependencies. See `requirements-offline.txt`.
- `fetch_torch_locked_wheels.py` — fetches the **torch-ABI-locked** packages
  (`mamba-ssm`, `causal-conv1d`) that MUST match Kaggle's exact torch build and
  therefore can't be picked from a Mac.
- `install_offline.sh` — installs everything in the correct order.

## How to build the complete dataset (one-time, internet ON)
1. New Kaggle notebook, GPU on, **internet ON**. Upload this folder (or attach it).
2. From inside the folder, run:
   ```bash
   python fetch_torch_locked_wheels.py --dest .
   ```
   It detects the live torch/CUDA/Python/ABI and downloads the exact
   `mamba_ssm-*` and `causal_conv1d-*` wheels next to the others.
3. **Save Version** → output becomes a Kaggle Dataset. Attach that dataset to your
   training notebook.

## How to install in the training notebook (internet OFF)
```bash
bash /kaggle/input/<your-dataset>/install_offline.sh /kaggle/input/<your-dataset>
```
or manually:
```bash
DIR=/kaggle/input/<your-dataset>
pip install --no-index --find-links="$DIR" \
    transformers peft trl datasets accelerate bitsandbytes psutil sentencepiece safetensors
# torch-locked, no-deps, causal-conv1d BEFORE mamba-ssm:
pip install --no-index --no-deps "$DIR"/causal_conv1d-*.whl "$DIR"/mamba_ssm-*.whl
```

## Notes
- `torch` is intentionally absent — use Kaggle's. If the `mamba_ssm` import fails
  with an ABI/symbol error, Kaggle's torch differs from the wheel; re-run
  `fetch_torch_locked_wheels.py` (it re-reads the live torch) and re-save.
- **vLLM (Phase 4 eval) is not bundled** — it is also torch-locked and heavy. Run
  eval in a notebook with internet ON, or use Kaggle's preinstalled vLLM. The
  offline bundle targets the training → adapter path (Phases 3 & 5).
