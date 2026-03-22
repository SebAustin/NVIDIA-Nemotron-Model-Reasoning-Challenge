# NVIDIA Nemotron Reasoning Challenge

Pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/) on Kaggle: train a LoRA adapter (rank ≤ 32) for **Nemotron-3-Nano-30B-A3B-BF16** to maximize accuracy on logical reasoning puzzles.

## Clone and run (from GitHub)

**Quick local run** (EDA + data prep + package; no GPU):

```bash
git clone https://github.com/SebAustin/nemotron-reasoning-challenge.git
cd nemotron-reasoning-challenge
# Add train.csv and test.csv to data/ (download from Kaggle competition page)
./run_local.sh
```

This creates a venv, installs `requirements-local.txt`, and runs the pipeline with `--synthetic-only --skip-train --skip-eval`. Produces `submission.zip` (synthetic data only). Training requires GPU — use Kaggle.

**Manual setup** (for full control):

```bash
bash setup_venv.sh
source .venv/bin/activate   # .venv\Scripts\activate on Windows
# Add train.csv and test.csv to data/, then:
python run_all.py --synthetic-only --skip-train --skip-eval
# Or run full pipeline on Kaggle via kaggle_notebook.ipynb
```

## Notebooks

- **`kaggle_notebook.ipynb`** — Run on [Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) (add competition data, project, and model dataset).
- **`anaconda_notebook.ipynb`** — Run on [Anaconda Cloud](https://nb.anaconda.com/). Create a project, upload the notebook + data, choose a PyTorch+GPU runtime, and run all cells.

**Pip does not use the GPU:** `pip install` is CPU-only; **0% GPU in the sidebar during install is expected**. GPUs are used during **training**. **Disk** on `/kaggle/working` is capped (~57 GiB); HF cache + pip + `hf_offload/` can overflow it — use `NEMOTRON_INPUT_PATH` from a dataset when possible and clear `.pip-cache` / `hf_offload` if needed.

**Input dataset:** Add the **competition dataset** to the notebook (via “Add Data” on the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) page). It must include **train.csv** (columns: `id`, `prompt`, `answer`) and **test.csv** (columns: `id`, `prompt`). You also need to provide the project code: either set `GITHUB_REPO` to a public repo URL (and enable Internet), or add a Kaggle Dataset that contains this project (e.g. a zip with `scripts/`, `run_all.py`, `requirements.txt`). See the notebook’s first cell for details. Keep **USE_PEFT_ONLY = True** to avoid Unsloth CUDA kernel errors on newer Kaggle GPUs (e.g. Blackwell).

**Internet:** `pip install` needs PyPI unless you use offline wheels (see below). Turn **Internet ON** in the notebook sidebar, **save/commit**, then run — *only if the competition allows Internet for your accelerator*.

**mamba_ssm / torch ABI on Kaggle:** If you see `ImportError: selective_scan_cuda... undefined symbol ... c10_cuda`, pip upgraded **torch** while **mamba_ssm** was built for an older torch. The notebook reinstalls `mamba-ssm` and `causal-conv1d` after `pip install`; **`requirements-kaggle-peft.txt` omits `torch`** so Kaggle’s image torch is preserved. Add **`HF_TOKEN`** in Secrets to avoid Hub rate limits.

**torchvision vs torch:** If you see `RuntimeError: operator torchvision::nms does not exist` or `Could not import module 'PreTrainedModel'`, **torchvision** does not match **torch** (common after partial pip upgrades). The notebook runs a **torchvision `--force-reinstall`** using the PyTorch **CUDA wheel index** derived from `torch.version.cuda`. Re-run the install cell or manually: `pip install torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu124` (adjust `cu124` to your CUDA, e.g. `cu121`, `cu126`).

**Faster installs on T4:** The notebook uses **`--prefer-binary`**, **`PIP_CACHE_DIR` under `/kaggle/working/.pip-cache`**, and optional **`SKIP_MAMBA_REINSTALL=True`** after the first successful run (skips the slow mamba rebuild). Each new Kaggle session still reinstalls from PyPI unless you attach a **wheel dataset** (`PIP_FIND_LINKS_DIR` + offline).

**GTX (and similar):** Kaggle may **disable Internet entirely** for this competition on GTX. You cannot work around that in settings. Use **`requirements-kaggle-peft.txt`** (PEFT + synthetic-only) and a **Kaggle Dataset of wheels**:

```bash
# On a Linux machine with Python 3.12 (match Kaggle) and CUDA, from this repo:
pip download -r requirements-kaggle-peft.txt -d ./pip-wheels
```

Upload `pip-wheels/` as a private dataset, add it to the notebook, then set `PIP_FIND_LINKS_DIR` and `PIP_OFFLINE=True` in `kaggle_notebook.ipynb`. Also set **`NEMOTRON_INPUT_PATH`** to a dataset containing **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** (HF file layout) so training does not need Hugging Face. Prefer a **GPU that allows Internet** (e.g. T4) if you want simpler `pip install` without a wheelhouse.

## Setup (one-time)

Create a virtual environment and install dependencies:

```bash
# From project root
bash setup_venv.sh
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
```

Use a different Python or venv path: `PYTHON=python3.11 VENV_DIR=./myenv bash setup_venv.sh`

**vLLM (Phase 4 evaluation only):** vLLM is not in the default requirements because it often fails to build from source on macOS or without CUDA. On a **Linux machine with NVIDIA GPU and CUDA**, install it after the main requirements:

```bash
pip install -r requirements-vllm.txt
```

If that fails (no matching pre-built wheel), try: `pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`. You can skip Phase 4 locally and run evaluation on Kaggle or a cloud GPU instead.

## Prerequisites

1. **Competition data**: Download `train.csv` and `test.csv` from Kaggle into `data/`.
2. **CoT generation** (Phase 2): Set `ANTHROPIC_API_KEY` (or configure another API) for chain-of-thought generation.
3. **GPU**: Training and vLLM evaluation require a GPU. The 30B MoE in 4-bit needs **2× GPU T4** on Kaggle (one 15GB card is not enough). Loading uses **`max_memory`** caps so weights split across GPUs + CPU (`hf_offload/`): defaults **7GiB on GPU0 / 9GiB on GPU1** (parallel load can spike VRAM on one card). Override with `SFT_MAX_MEMORY_GB_PER_GPU` (same for all GPUs), or `SFT_MAX_MEMORY_GB_GPU0` / `SFT_MAX_MEMORY_GB_GPU1`. If load still OOMs, try `SFT_CUDA_MEMORY_FRACTION=0.78` or lower the GiB caps further. `PYTORCH_CUDA_ALLOC_CONF` is set **before** `import torch` in `03_train_lora.py`. Training defaults: `SFT_MAX_SEQ_LENGTH=2048`, optional `LORA_R=16`. **Pip** conflict warnings on Kaggle are usually harmless.

## Project layout

```
data/                 # train.csv, test.csv, train_categorized.csv, train_sft.jsonl, synthetic/
scripts/
  01_eda.py           # EDA and puzzle-type categorization
  02_prepare_data.py  # CoT generation + synthetic data → train_sft.jsonl
  03_train_lora.py    # LoRA SFT (and optional GRPO)
  04_evaluate.py      # vLLM evaluation and error analysis
  05_package_submission.py
  utils/
    answer_extractor.py
    cot_generator.py
    data_formatter.py
    puzzle_generator.py
lora_output/          # Training checkpoints
lora_adapter/         # Final adapter for submission
submission.zip        # Packaged submission
```

## Usage

**Run the full pipeline** (with venv activated):

```bash
python run_all.py
```

Options: `--skip-eda`, `--skip-prepare`, `--synthetic-only`, `--skip-train`, `--skip-eval`, `--skip-package`, `--max-cot N`. Example: synthetic-only data then skip training/eval:

```bash
python run_all.py --synthetic-only --skip-train --skip-eval
```

**Or run steps individually:**

1. **EDA**: `python scripts/01_eda.py` (requires `data/train.csv`, `data/test.csv`).
2. **Prepare data**: `python scripts/02_prepare_data.py` (produces `data/train_sft.jsonl`).
3. **Train**: `python scripts/03_train_lora.py` (writes `lora_adapter/`).
4. **Evaluate**: `python scripts/04_evaluate.py`.
5. **Package**: `python scripts/05_package_submission.py` → `submission.zip`.

## Iteration (after first submission)

- Weak puzzle type → add more synthetic data for that type and retrain.
- Format errors (missing `\boxed{}`) → add format-focused examples or GRPO with format penalty.
- Overfitting → fewer epochs, more dropout, more diverse data.
- Run GRPO (Stage 2) in `03_train_lora.py` after SFT baseline.

## Pushing this repo to GitHub

- Do **not** commit `data/train.csv`, `data/test.csv`, or generated data (they are in `.gitignore`). Download them from Kaggle after cloning.
- Do **not** commit `.venv/`, `lora_adapter/`, or `submission.zip` (ignored).
- From the project root: `git init`, `git add .`, `git commit -m "Initial commit"`, then add your remote and push.
