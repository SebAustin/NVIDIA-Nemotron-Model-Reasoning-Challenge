# Train on Modal (A100-80GB, bf16)

The 31.6B base model is ~63 GB in bf16 — it fits on a single **A100-80GB** but not
on 2×T4 (32 GB) or a 40 GB A100 in bf16. This runs the one training step on Modal
and leaves the adapter in a Volume; do data-prep was-here too (it's self-contained).

## One-time setup
```bash
pip install modal
modal setup                       # auth
modal secret create nemotron KAGGLE_TOKEN=KGAT_xxx   # + HF_TOKEN=hf_xxx if base model needs auth
```

## Run
```bash
modal run modal/train_modal.py
modal volume get nemotron-out lora_adapter ./lora_adapter   # download the trained adapter
```
Then upload `lora_adapter/` as a Kaggle dataset and package with
`scripts/05_package_submission.py` (see the root README / WRITEUP).

## Notes
- The image pins `torch==2.6.0` and installs matching prebuilt `mamba_ssm` /
  `causal_conv1d` wheels (no source build).
- The base model downloads from HF on first run and is cached in the `hf-cache`
  Volume for reruns. If the download 401s, add `HF_TOKEN` to the `nemotron` secret.
- Runtime ~1–3 h; A100-80GB on Modal is ~$3–4/h.
- Knobs pass through as env: `LORA_R`, `NUM_EPOCHS`, `LEARNING_RATE`,
  `SFT_MAX_SEQ_LENGTH` (set them in the `@app.function` env or the secret).
