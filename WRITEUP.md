# NVIDIA Nemotron Model Reasoning Challenge — Method Writeup

A LoRA adapter for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` that teaches the
base model to solve "Alice's Wonderland" few-shot **rule-induction** puzzles. The
pipeline is: EDA → per-family solvers → solver-grounded + synthetic SFT data →
8-bit QLoRA → metric-mirroring eval → packaged adapter.

## 1. Verified ground truth (the source of truth, not assumptions)

All confirmed against the HF Hub, the model tokenizer, and the official
`ryanholbrook/nvidia-nemotron-submission-demo` notebook:

| Item | Verified value |
|---|---|
| Base model id | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (31.6B, `nemotron_h` hybrid Mamba-2 + GQA + MoE) |
| Kaggle model mount | `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` |
| Data mount | `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/` |
| Chat template | ChatML + thinking; gen prompt ends `…<\|im_start\|>assistant\n<think>\n` |
| Eval prompt | **No system prompt**; user = puzzle + suffix ``"\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`"``; `add_generation_prompt=True, enable_thinking=True` |
| Decoding | greedy: `temperature=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192, max_lora_rank=32` |
| Answer extraction | last `\boxed{…}` → heuristic → last numeric |
| Scoring | exact string **OR** relative tolerance `1e-2` |
| Official adapter | `LoraConfig(r=32, lora_alpha=16, target_modules=r".*\.(in_proj\|out_proj\|up_proj\|down_proj)$", lora_dropout=0.05, bias="none")` |
| Submission | `submission.zip` with `adapter_config.json` + weights **at the zip root** |

> Correction applied repo-wide: the brief's `nvidia/Nemotron-3-Nano-30B-A3B-BF16`
> (missing the `NVIDIA-` prefix) 404s; the correct id is used everywhere.

## 2. The data (9,500 train rows; hidden test)

Six balanced families (~1,575–1,602 rows each), all opening "In Alice's
Wonderland, …", each a *few-shot rule-induction* task (infer the hidden rule from
input→output examples, apply to a query):

| Family | Task | Answer |
|---|---|---|
| `bit_manipulation` | hidden 8-bit→8-bit transform | 8-bit string |
| `gravity` | `d = ½·g·t²`, g secretly changed | number |
| `unit_conversion` | hidden linear scaling | number |
| `encryption` | monoalphabetic letter substitution | text |
| `numeral` | integer → Roman numerals | Roman string |
| `equation` | per-row secret symbol-operator semantics | symbol string |

## 3. Per-family deterministic solvers (`scripts/utils/solvers.py`)

Each solver parses the examples, infers the rule, and predicts the query. Validated
on every train row under the **real metric** (exact OR rel-tol 1e-2):

| Family | Solver | Solve rate |
|---|---|---|
| gravity | least-squares fit of `g = 2d/t²` | **100.0%** |
| unit_conversion | linear `y = k·x (+ b)` fit | **100.0%** |
| numeral | standard Roman numeral encoder | **100.0%** |
| encryption | aligned substitution map + Wonderland-vocabulary fill for letters unseen in a row's examples | **99.4%** |
| bit_manipulation | GF(2) affine-per-bit (+ leave-one-out) and an op library (rotate/shift/xor/and/or/not/maj/add) | ~10% 🚩 |
| equation | per-operator op library (concat / reverse / interleave) | ~8% 🚩 |

**🚩 Flagged families.** `bit_manipulation` rules include nonlinear majority/choice
functions and deeper compositions; `equation` assigns *different* secret semantics
to operators per row. Neither is deterministically solvable to 100% (consistent
with the competition being hard — the public LB-0.85 winner relied on the model +
investigation, not pure solvers). They are handled by **forward synthetic
generation** (§4): we generate puzzles from *known* rules, yielding unlimited
clean reasoning traces.

The encryption vocabulary trick (build the plaintext vocabulary across all rows,
then pattern-match test words to recover letters unseen in a given row's examples)
lifted that family from 38% → 99.4%.

## 4. SFT data construction (`scripts/02_prepare_data.py`)

Output: `data/train_sft.jsonl` (~12,500 examples), every example verified so its
`\boxed{}` answer round-trips through the metric (0 mismatches).

- **Real rows.** Solvable families get a concise, solver-grounded `<think>`
  reasoning trace (rule stated → applied → answer). Hard-family rows and any solver
  miss become **direct-answer** examples anchored on the gold answer (no fabricated
  reasoning).
- **Synthetic rows.** `puzzle_generator.py` emits puzzles for all six families
  from known rules, using the *exact* real prompt templates, with correct answers
  and reasoning. Hard families (`bit_manipulation`, `equation`) are **upweighted 3×**.
- **Format.** Matches eval exactly: no system prompt, boxed suffix on the user
  turn, assistant target `<think>\n{reasoning}\n</think>\n\boxed{answer}`.
- **Mix.** ~75% reasoning / 25% direct-answer (preserves base reasoning per
  Nemotron LoRA guidance); dedup + length filter; all seeds fixed.

## 5. Training (`scripts/03_train_lora.py`) — Kaggle 2×T4 (8-bit + offload)

4-bit QLoRA is unreliable on this hybrid Mamba model, so we load **8-bit**
(`load_in_8bit` + `llm_int8_enable_fp32_cpu_offload`), shard with
`device_map="auto"` + `max_memory` caps, and offload to CPU/disk.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before importing torch;
gradient checkpointing + bf16 compute.

- **LoRA:** `r=16` (env `LORA_R`, ≤32), `alpha=2r`, dropout 0, bias none.
  `target_modules` default to the **official demo regex**
  (`in_proj|out_proj|up_proj|down_proj`) — verified against `model.named_modules()`
  and asserted to **never** match the MoE router/gate.
- **Safety rails:** a **smoke test** (1% data, 10 steps) proves the memory config
  fits before the full run; **per-category min-logprob** is logged each epoch and
  categories not approaching 0 are flagged for upweighting next round; on CUDA OOM
  the script writes **`FALLBACK.md`** (rent one A100/H100, identical config, copy
  the adapter back) rather than silently truncating the model.

VRAM note: 31.6B in int8 ≈ 31.6 GB of weights alone vs 32 GB across 2×T4, so CPU
offload is mandatory and OOM is the expected #1 risk — the FALLBACK path exists for
exactly this.

## 6. Evaluation (`scripts/04_evaluate.py`)

Mirrors the metric exactly: 10% stratified per-category holdout, vLLM greedy with
the competition decoding params, the eval prompt template, the same extraction
chain, exact-or-rel-1e-2 scoring. Emits overall + per-category accuracy and a CSV
of every miss (`id, category, gold, pred, raw_output`). `--base-only` measures the
pre-LoRA baseline for ablation.

## 7. Packaging (`scripts/05_package_submission.py`)

Verifies `adapter_config.json` (r≤32) and writes `submission.zip` with the adapter
files at the **zip root** (matching the demo), printing the final listing and
`r/alpha/target_modules/base` for a sanity check.

## 8. The loop

Run 1→5, read per-category accuracy + min-logprob, upweight the worst 1–2 families
in the synthetic mix (`--hard-multiplier`, or add a family to `HARD`), retrain,
repeat until the held-out score plateaus.

## 9. How to run

Local (no GPU) — data is fully reproducible:
```bash
python scripts/01_eda.py --data-dir data
python scripts/02_prepare_data.py --data-dir data   # -> data/train_sft.jsonl
```
Kaggle GPU (training + eval), after attaching the model + competition data:
```bash
python scripts/03_train_lora.py --data-path data/train_sft.jsonl --output-dir lora_adapter
python scripts/04_evaluate.py   --adapter-path lora_adapter --data-dir data
python scripts/05_package_submission.py --adapter-dir lora_adapter   # -> submission.zip
```

## 10. Ablations / per-category results

| Run | Overall holdout | bit | gravity | unit | enc | numeral | equation | Notes |
|---|---|---|---|---|---|---|---|---|
| Solver-only (oracle on train) | 69.6% | 10% | 100% | 100% | 99% | 100% | 8% | upper bound from deterministic solvers |
| Base model (no adapter) | _TBD_ | | | | | | | run `04 --base-only` |
| LoRA r16, demo targets, 3× hard | _TBD_ | | | | | | | first training run |

_Fill the TBD rows after the first Kaggle training + eval run; this table is the
per-loop scoreboard (overall acc, per-category delta, VRAM peak, wall-clock)._
