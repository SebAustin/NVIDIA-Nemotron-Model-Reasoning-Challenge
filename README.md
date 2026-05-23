# NVIDIA Nemotron Reasoning Challenge

![Kaggle](https://img.shields.io/badge/Kaggle-Nemotron%20Reasoning%20Challenge-20BEFF?logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Base model](https://img.shields.io/badge/Base-Nemotron--3--Nano--30B-76B900?logo=nvidia&logoColor=white)
![Method](https://img.shields.io/badge/Method-LoRA%20%E2%86%92%20GRPO-orange)
![License](https://img.shields.io/badge/License-Nemotron%20Open%20Model-lightgrey)

End-to-end fine-tuning pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle. Trains a LoRA adapter on top of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` to improve few-shot rule-induction reasoning over bit manipulation, ciphers, numeric sequences, and related puzzle families.

> **TL;DR** — A reproducible, numbered-script pipeline that builds offline training data on a laptop, runs supervised fine-tuning (SFT) followed by GRPO reinforcement on Kaggle GPUs, and packages a competition-ready LoRA adapter. Current leaderboard score: **0.60**.

## Contents

- [What the competition actually scores](#what-the-competition-actually-scores)
- [Pipeline](#pipeline)
- [Repository layout](#repository-layout)
- [How to reproduce](#how-to-reproduce)
- [Key technical decisions and the rationale](#key-technical-decisions-and-the-rationale)
- [Honest status](#honest-status)
- [Credits and references](#credits-and-references)
- [License](#license)

| Metric | Value |
|---|---|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (30B MoE, hybrid Mamba + Transformer) |
| Adaptation | LoRA, rank 32 (competition cap) |
| Training compute | Kaggle GPU L4 x 4 (~96 GB total VRAM) |
| Stages | SFT v2 + GRPO Stage 2 |
| Current LB score | 0.60 |
| Target (Progress-Prize winning band) | 0.85+ |

The repository is organized as a reproducible pipeline of numbered scripts driven by two Kaggle notebooks (`kaggle/kaggle_training.ipynb` for training, `kaggle/kaggle_submission.ipynb` for packaging). Offline data generation runs on a laptop and ships to Kaggle as a Dataset; Kaggle training is internet-off after the first run.

## What the competition actually scores

Submissions ship a LoRA adapter only; Kaggle's evaluation harness loads it into the base model and runs inference on a hidden test set. The grading rule (confirmed against the Progress-Prize-winning reference and reproduced in [`kaggle/scripts/utils/competition_metric.py`](kaggle/scripts/utils/competition_metric.py)) is:

1. Binary strings (`[01]+`): strict case-insensitive string equality (leading zeros matter).
2. Otherwise: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` on float casts.
3. Otherwise: case-insensitive string equality.

Answers are extracted from the **last** `\boxed{...}` in the model's response. The chat template (`enable_thinking=True`) emits `<|im_start|>assistant\n<think>\n` and expects the model's completion to be `{reasoning}\n</think>\n\boxed{ANSWER}<|im_end|>`. Our training format matches this exactly (see [`kaggle/scripts/utils/data_formatter.py`](kaggle/scripts/utils/data_formatter.py)).

## Pipeline

```
+-------------------+      +-------------------+      +-------------------+
|  Phase 0          |      |  Phase 1          |      |  Phase 2          |
|  Diagnostics      | ---> |  Offline data     | ---> |  SFT v2 on Kaggle |
|  (metric, parity) |      |  (laptop, ~$1-3)  |      |  (~8-10 h)        |
+-------------------+      +-------------------+      +-------------------+
                                                              |
                                                              v
                                                     +-------------------+
                                                     |  Phase 3          |
                                                     |  GRPO Stage 2     |
                                                     |  (~6-8 h)         |
                                                     +-------------------+
                                                              |
                                                              v
                                                     +-------------------+
                                                     |  Phase 4          |
                                                     |  Multi-seed soup  |
                                                     |  + submit         |
                                                     +-------------------+
```

### Phase 0 - Diagnostics and alignment

Before spending any GPU compute, these checks kill the silent-failure modes that make every downstream run lie about its score.

| Step | Script | What it does |
|---|---|---|
| 0.1 | [`utils/competition_metric.py`](kaggle/scripts/utils/competition_metric.py) | Implements the verified competition metric (binary-string, numeric tolerance, case-insensitive fallback). Replaces the previous placeholder that over-counted matches locally. |
| 0.2 | [`05_package_submission.py`](kaggle/scripts/05_package_submission.py) | Aligns tokenizer-inclusion policy between the packaging script and the submission notebook so what gets shipped is what gets evaluated. |
| 0.3 | [`04_evaluate.py`](kaggle/scripts/04_evaluate.py) | Per-family eval on the current adapter (`--samples-per-family 50`, writes `eval_by_family.csv`). Classifier recognizes the 9 actual test categories: `bit_manipulation`, `cipher`, `gravity`, `unit_conversion`, `numeral`, `cryptarithm_*`, `equation_numeric_*`. |
| 0.4 | [`00_check_prompt_parity.py`](kaggle/scripts/00_check_prompt_parity.py) | Token-aligns training and eval prompts. Caught the original `<think>\n` vs `<think></think>` mismatch in the Nemotron chat template - a silent training/inference divergence that produced the wrong learning signal. |

### Phase 1 - Offline data build

Runs on a laptop. Produces `kaggle/data/train_sft.jsonl`, uploaded to Kaggle as a Dataset (`upload_train_data_dataset.py`).

```bash
export DEEPSEEK_API_KEY=sk-...
bash kaggle/scripts/build_offline_data.sh deepseek
```

Cost: ~$1-3 with DeepSeek V3, ~5x cheaper than Claude Sonnet 4. Knobs documented in [`build_offline_data.sh`](kaggle/scripts/build_offline_data.sh): `QUICK_TEST=1`, `COT_LIMIT=N`, `SKIP_TEACHER=1`, `COT_MAX_TOKENS=N`.

The data merges five sources, deduped and curriculum-sorted:

| Source | Generator | What it adds |
|---|---|---|
| Synthetic | [`utils/puzzle_generator.py`](kaggle/scripts/utils/puzzle_generator.py) | 9k+ procedurally-generated examples across 4 families (38 sub-modes) |
| Solver CoTs | [`utils/solvers.py`](kaggle/scripts/utils/solvers.py) | Deterministic reasoning traces for any row a programmatic solver can crack, verified against gold |
| Teacher CoTs | [`utils/cot_generator.py`](kaggle/scripts/utils/cot_generator.py) | API-distilled CoTs (DeepSeek V3 / Claude / GPT-4o) for rows the solver can't, kept only if the teacher's `\boxed{}` matches gold |
| Template CoTs | [`utils/cot_generator.generate_template_cot`](kaggle/scripts/utils/cot_generator.py) | Fallback scaffolding for solver- and API-uncovered rows |
| Test pseudo-labels | [`02b_pseudolabel_test.py`](kaggle/scripts/02b_pseudolabel_test.py) | Solver-verified labels on `test.csv` prompts (deterministic teacher; no leakage of hidden labels) |

The dedupe step is a normalized-prompt SHA1 fingerprint; the curriculum step sorts ascending by `(cot_source priority, token count)` so solver CoTs and shorter prompts come first within a single epoch.

### Phase 2 - SFT v2

```python
# kaggle/kaggle_training.ipynb config cell:
TRAIN_MAX_SEQ    = 4096
LORA_TARGET_MODE = "kaggle_nemotron"  # in_proj/out_proj + attn + MoE
LORA_ALPHA       = 64                  # rsLoRA on -> effective scaling ~11.3
NUM_EPOCHS       = 2.0
LR               = 1e-4
COMPLETION_ONLY  = True                # mask system+user, train only on assistant tokens
```

Implementation: [`03_train_lora.py`](kaggle/scripts/03_train_lora.py). Uses TRL `SFTTrainer` with `DataCollatorForCompletionOnlyLM` and a probe that asserts the assistant-turn marker actually appears in the tokenized prompt (otherwise the collator silently masks every label and the run is a no-op). bf16 full-precision base; LoRA layers in bf16; rsLoRA + NEFTune (alpha=5.0).

### Phase 3 - GRPO Stage 2

```python
# kaggle/kaggle_training.ipynb config cell:
RUN_GRPO              = True
GRPO_EPOCHS           = 1.0
GRPO_LR               = 5e-6
GRPO_LIMIT            = 600
GRPO_NUM_GENERATIONS  = 4
GRPO_MAX_NEW_TOKENS   = 1024
```

Implementation: [`08_grpo.py`](kaggle/scripts/08_grpo.py). Warm-starts from the SFT v2 adapter and optimizes two rewards:

- `reward_correctness`: `+1.0` if the boxed answer matches gold, `+0.1` if format is present but wrong, `-0.5` if `\boxed{}` is missing.
- `reward_format`: `+0.2` if the response ends with `}`, `+0.05` for `\boxed{` only, `-0.2` otherwise.

No teacher API needed - rewards come from `train.csv` gold answers. The notebook reassigns `ADAPTER_DIR` to the GRPO output on success, so Phase 6 (mirror) and Phase 7 (publish) ship the GRPO adapter; if GRPO fails (OOM, etc.) the notebook falls back to the SFT adapter.

### Phase 4 - Multi-seed soup

Runs SFT (and optionally GRPO) with multiple seeds, then averages LoRA A/B matrices (Wortsman et al. "Model Soups"). Implementation: [`07_multi_seed_average.py`](kaggle/scripts/07_multi_seed_average.py).

Optional: hard-example mining + DPO via [`06_mine_hard.py`](kaggle/scripts/06_mine_hard.py) -> [`06b_build_dpo_pairs.py`](kaggle/scripts/06b_build_dpo_pairs.py) -> [`07_dpo.py`](kaggle/scripts/07_dpo.py). Gated by teacher API access.

## Repository layout

```
.
|-- kaggle/
|   |-- kaggle_training.ipynb        # main training pipeline (Phases 0-7)
|   |-- kaggle_submission.ipynb      # packages submission.zip from a LoRA adapter
|   |-- kaggle_debug_workflow.ipynb  # scratch notebook for isolating failures
|   |-- kaggle_build_training_wheels.ipynb
|   |-- kaggle_build_vllm_wheels.ipynb
|   |-- colab_workflow.ipynb         # historical Colab variants (+ _v2, _v3, _v4)
|   |-- local_gtx_workflow.ipynb     # local consumer-GPU workflow
|   |-- udacity_workflow.ipynb       # alt cloud-GPU workflow
|   |-- requirements.txt             # training-side: torch, transformers, peft, trl, ...
|   |-- requirements-mamba.txt       # mamba_ssm + causal_conv1d (Linux+CUDA only)
|   |-- requirements-vllm.txt        # inference-side: vllm + minimal deps
|   |-- dataset-metadata.json        # scripts Kaggle Dataset
|   |-- kernel-metadata.json         # submission kernel config
|   |
|   |-- scripts/
|   |   |-- 00_check_prompt_parity.py
|   |   |-- 01_eda.py
|   |   |-- 02_prepare_data.py       # builds train_sft.jsonl
|   |   |-- 02b_pseudolabel_test.py  # solver pseudo-labels on test.csv
|   |   |-- 03_train_lora.py         # SFT training
|   |   |-- 04_evaluate.py           # per-family vLLM eval
|   |   |-- 05_package_submission.py
|   |   |-- 06_kfold_cv.py
|   |   |-- 06_mine_hard.py
|   |   |-- 06b_build_dpo_pairs.py
|   |   |-- 07_dpo.py
|   |   |-- 07_multi_seed_average.py
|   |   |-- 08_grpo.py               # GRPO Stage 2 (the score booster)
|   |   |-- 08_upload_adapter_dataset.py
|   |   |-- 09_verify_kaggle_inputs.py
|   |   |-- 10_upload_base_model.py
|   |   |-- 11_upload_scripts_dataset.py
|   |   |-- 12_build_nemotron_wheels_dataset.py
|   |   |-- build_offline_data.sh    # laptop helper for Phase 1
|   |   |-- upload_train_data_dataset.py
|   |   `-- utils/
|   |       |-- answer_extractor.py
|   |       |-- competition_metric.py
|   |       |-- cot_generator.py     # teacher API client + template CoT
|   |       |-- data_formatter.py    # chat template + assistant reply formatter
|   |       |-- lora_merge.py
|   |       |-- puzzle_generator.py  # synthetic puzzle generators
|   |       `-- solvers.py           # deterministic puzzle solvers
|   `-- data/                        # competition CSVs + generated data (gitignored)
|       |-- synthetic/               # per-family synthetic puzzle jsonl (generated)
|       `-- reports/                 # eval breakdowns, dedupe stats, alignment checks
|
|-- kaggle_lora_dataset/
|   `-- dataset-metadata.json        # LoRA adapter Kaggle Dataset
|
|-- .gitignore
`-- README.md
```

Everything under `kaggle/data/` (competition CSVs, generated `train_sft.jsonl`, synthetic puzzle files) is intentionally **not** committed — it is regenerated from the [How to reproduce](#how-to-reproduce) steps. Only the code, notebooks, and config metadata are versioned.

## How to reproduce

### Prereqs

- Python 3.11+ (laptop) for the offline data step.
- A Kaggle account with API credentials at `~/.kaggle/kaggle.json`.
- (Optional) A DeepSeek API key for the teacher-CoT pass: [platform.deepseek.com](https://platform.deepseek.com). Anthropic / OpenAI also supported.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r kaggle/requirements.txt
```

### Run

```bash
# 1. (Optional) Verify training and eval produce the same prompt prefix.
python3 kaggle/scripts/00_check_prompt_parity.py

# 2. Stage competition CSVs.
mkdir -p kaggle/data
cp /path/to/train.csv /path/to/test.csv kaggle/data/

# 3. Build offline training data (laptop).
export DEEPSEEK_API_KEY=sk-...
QUICK_TEST=1 bash kaggle/scripts/build_offline_data.sh   # ~$0.01 smoke test
bash kaggle/scripts/build_offline_data.sh                # full run, ~$1-3

# 4. Upload as Kaggle Dataset.
python3 kaggle/scripts/upload_train_data_dataset.py \
    --slug <your-username>/nemotron-train-data --create

# 5. Open kaggle/kaggle_training.ipynb on Kaggle, attach the four datasets:
#    - competition data, base model, scripts dataset, the train-data dataset above
#    Settings: GPU L4 x 4, Internet ON for first run.
#    Run All. ~16-20h end-to-end (SFT v2 + GRPO Stage 2).

# 6. Open kaggle/kaggle_submission.ipynb on Kaggle, attach the new adapter version,
#    Run All -> submit /kaggle/working/submission.zip.
```

## Key technical decisions and the rationale

- **Chat-template alignment with the eval harness.** The Nemotron template under `enable_thinking=True` emits `<|im_start|>assistant\n<think>\n` at the end of an inference prompt but `<|im_start|>assistant\n<think></think>` before a saved assistant turn. Originally our training data passed the assistant message through `apply_chat_template(add_generation_prompt=False)`, which produced the latter - so the model was learning conditional `P(completion | <think></think>...)` while the harness queries `P(completion | <think>\n...)`. The current pipeline builds the prompt with `add_generation_prompt=True` over system+user only and then concatenates the assistant content + EOS, matching what the harness emits exactly.
- **Completion-only loss with a verified response template.** TRL's `DataCollatorForCompletionOnlyLM` masks every label as -100 if its response-template ids are not a token-id subsequence of the tokenized prompt. We probe the chat template at startup and assert subsequence membership before training begins, so silent no-op runs are impossible.
- **rsLoRA with effective scaling ~11.3.** With LoRA rank capped at 32, plain alpha-tuning runs out of headroom quickly. rsLoRA (`use_rslora=True`) scales the LoRA update by `alpha / sqrt(r)` instead of `alpha / r`, which is more numerically stable under the rank cap. We pair it with `alpha=64`.
- **GRPO with gold-answer rewards.** The competition test set is held out but training labels are public, so GRPO can directly optimize answer correctness with no teacher API in the loop. The defensive dispatch in `08_grpo.py` (`reward_funcs` vs `reward_fn`, `processing_class` vs `tokenizer`, `num_generations` vs `num_return_sequences`) covers TRL API drift across versions >= 0.12.
- **Submission package is intentionally minimal.** The adapter zip contains weights + tokenizer config + chat template. Kaggle's harness applies the LoRA against the base model and uses whichever chat template ships with the adapter, so the chat-template file is the lever for whether `<think>` reasoning is enabled.

## Honest status

| Submission | LB |
|---|---|
| Baseline (synthetic-only, single-seed SFT) | 0.58 |
| Full pipeline (SFT v2 + GRPO, chat-template fix, DeepSeek CoTs, curriculum) | 0.60 |
| Progress-Prize winning band ([`tonghuikang/nemotron`](https://github.com/tonghuikang/nemotron)) | 0.85 - 0.88 |

The +0.02 gain over baseline suggests the bottleneck is no longer hyperparameters or chat-template alignment. Per-family eval on the 0.60 adapter is the next diagnostic; the working hypothesis is **category coverage** (5 of the 9 actual test categories - `gravity`, `unit_conversion`, `numeral`, `cryptarithm_*`, `equation_numeric_*` - currently have no synthetic generator in `puzzle_generator.py`). Porting the Progress-Prize winner's per-category reasoners is the next planned change.

## Credits and references

- [tonghuikang/nemotron](https://github.com/tonghuikang/nemotron) - Progress-Prize winning submission (Kaggle LB 0.85). Used as a reference for the chat-template format, the metric, and the canonical test-category taxonomy.
- [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) - base model.
- TRL, PEFT, transformers, vLLM - core training and inference stack.

## License

Code in this repository is provided as-is for the competition. The base model is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
