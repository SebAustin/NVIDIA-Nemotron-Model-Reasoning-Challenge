# NVIDIA Nemotron Reasoning Challenge

Pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/) on Kaggle: train a LoRA adapter (rank ≤ 32) for **Nemotron-3-Nano-30B-A3B-BF16** to maximize accuracy on logical reasoning puzzles.

## Clone and run (from GitHub)

```bash
git clone https://github.com/SebAustin/nemotron-reasoning-challenge.git
cd nemotron-reasoning-challenge
bash setup_venv.sh
source .venv/bin/activate
# Add train.csv and test.csv to data/ (from Kaggle), then:
python run_all.py --skip-train --skip-eval   # local (no GPU)
# Or run full pipeline on Kaggle via kaggle_notebook.ipynb (set GITHUB_REPO to this repo).
```

## Kaggle notebook

Use **`kaggle_notebook.ipynb`** to run the full pipeline on Kaggle and produce `submission.zip`.

**Input dataset:** Add the **competition dataset** to the notebook (via “Add Data” on the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) page). It must include **train.csv** (columns: `id`, `prompt`, `answer`) and **test.csv** (columns: `id`, `prompt`). You also need to provide the project code: either set `GITHUB_REPO` to a public repo URL (and enable Internet), or add a Kaggle Dataset that contains this project (e.g. a zip with `scripts/`, `run_all.py`, `requirements.txt`). See the notebook’s first cell for details.

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
3. **GPU**: Training and vLLM evaluation require a GPU (e.g. ~24GB VRAM for 4-bit LoRA).

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
