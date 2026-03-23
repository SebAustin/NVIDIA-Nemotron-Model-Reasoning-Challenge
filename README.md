# NVIDIA Nemotron Model Reasoning Challenge

Pipeline: EDA → supervised data (CoT + synthetic) → QLoRA SFT → vLLM eval → `submission.zip`.

## Setup

1. Place Kaggle files as `data/train.csv` and `data/test.csv`.
2. (Optional) Copy the competition **public metric** into `scripts/utils/competition_metric.py` with a function `scores(gold: str, pred: str) -> bool` so evaluation matches the leaderboard exactly.
3. Python **3.10–3.12** recommended (3.14 is often missing prebuilt wheels; more builds from source).

```bash
./setup_venv.sh
# or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

For vLLM evaluation only, use a separate env with `requirements-vllm.txt` if needed.

**macOS / Apple Silicon:** `requirements.txt` is enough for **EDA + data prep** (`01_eda.py`, `02_prepare_data.py`). Do **not** `pip install causal-conv1d` / `mamba-ssm` on Mac unless you have a working CUDA or vendor-specific setup — builds usually fail (`nvcc` missing, or `NameError: bare_metal_version` in `causal-conv1d`). For **Nemotron training**, use **Kaggle GPU** and `requirements-mamba.txt` there.

## Environment variables

| Variable | Used by |
|----------|---------|
| `NEMOTRON_MAX_MEMORY_CPU` | `03_train_lora.py` auto `max_memory` host-RAM cap (e.g. `8GiB`) if SIGKILL during load |
| `ANTHROPIC_API_KEY` | `02_prepare_data.py` when `--cot-backend anthropic` |
| `OPENAI_API_KEY` | `02_prepare_data.py` when `--cot-backend openai` |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible base URL |
| `WANDB_API_KEY` | Training logs when `USE_WANDB=1` |

## Run order

```bash
python scripts/01_eda.py --data-dir data --report-dir data/reports
python scripts/02_prepare_data.py --data-dir data --skip-cot  # or configure API for full CoT
python scripts/03_train_lora.py --data-path data/train_sft.jsonl --output-dir lora_adapter
python scripts/04_evaluate.py --adapter-path lora_adapter --data-dir data
python scripts/05_package_submission.py --adapter-dir lora_adapter
```

On Kaggle (after `kagglehub.model_download`), prefer: `--model-path <cache> --no-quant --lora-target-mode kaggle_nemotron --lora-alpha 16 --force-peft` (see **Kaggle** section).

Use `--help` on each script for options.

## Kaggle

Use [`kaggle_workflow.ipynb`](kaggle_workflow.ipynb) on a GPU notebook. **Re-upload your Kaggle code dataset whenever `scripts/` changes** (an old `02_prepare_data.py` only understands flags like `--synthetic-only` / `--bit` and will reject `--data-dir`). The current `02_prepare_data.py` supports both styles: full flags (`--data-dir`, `--tokenizer-model`, `--synthetic-per-kind`, …) and aliases (`--synthetic-only` → `--skip-cot`, `--max-cot` → `--limit-train`, per-type `--bit` / `--cipher` / `--algebraic` / `--sequence`).

Defaults in the notebook config point at a scripts dataset and at `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/` for `train.csv` (adjust paths if your mounts differ). Base weights load via competition mount and/or `kagglehub` (`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`). Training uses **`transformers>=4.45,<5`** (v5 can OOM loading 4-bit on 16GB GPUs), **`--lora-target-mode kaggle_nemotron`** (alpha 16), and **4-bit QLoRA** on T4 (`USE_BF16_FULL = False`).

**Mamba dependency:** loading the full Nemotron checkpoint requires **`mamba-ssm`** and **`causal-conv1d`** (the `kaggle_workflow.ipynb` pip cell installs them). If pip fails to build wheels, align versions with the official competition starter notebook (`import mamba_ssm`).

**Offline wheel datasets (optional):** You can attach a dataset of `.whl` files (e.g. [dennisfong/nvidia-nemotron-offline-packages](https://www.kaggle.com/datasets/dennisfong/nvidia-nemotron-offline-packages)), then install from the mount instead of PyPI. After adding it, check the exact path under **`/kaggle/input/`** (Kaggle rewrites the folder name). Example pattern:

```bash
WHEELS=/kaggle/input/<folder-from-inputs-tab>   # e.g. nvidia-nemotron-offline-packages
pip install --no-index --find-links="file://$WHEELS" <package-names...>
```

Still enforce this repo’s constraints where they matter (**`transformers>=4.45,<5`**, compatible **`torch`** with Kaggle’s CUDA). If the bundle is missing a dependency (e.g. `polars`, `kagglehub`), fall back to `pip install` for those only.

**MoE + `device_map="auto"`:** bf16 loads may spill weights to disk; `03_train_lora.py` passes an absolute **`offload_folder`** (and logs a `build=` line). Re-sync scripts if that line is missing. **TRL:** recent `SFTConfig` uses **`max_length`** instead of **`max_seq_length`**; the training script adapts via `inspect` so multiple TRL versions work.

**T4 / ~15GB VRAM:** do **not** use **`--no-quant`** + long context — you will OOM. Use **4-bit QLoRA** (`USE_BF16_FULL = False`) and **`--max-seq-length`** around **2048**. If the weight bar reaches ~25% then OOMs, you likely have **transformers 5.x**; pin **`<5`** in the pip cell. **`SIGKILL` / exit code -9** during “Loading checkpoint shards” is usually **host RAM**: do not set **`cpu` to hundreds of GiB** in `max_memory` on Kaggle; `03_train_lora.py` now caps auto **`cpu`** from real RAM (override with env **`NEMOTRON_MAX_MEMORY_CPU`** e.g. `8GiB` if needed). For **2× T4** with bf16, optional `'{"0":"13GiB","1":"13GiB","cpu":"12GiB","disk":"120GiB"}'`.

## Notes

- Base model: `nvidia/Nemotron-3-Nano-30B-A3B-BF16` (LoRA rank ≤ 32).
- Training uses Unsloth when import succeeds (HF id + 4-bit); otherwise PEFT. Use `--no-quant` for full bf16 weights (e.g. Kaggle Hub cache).
- `02_prepare_data.py` can generate synthetic-only data with `--skip-cot` for a quick baseline without API calls.
