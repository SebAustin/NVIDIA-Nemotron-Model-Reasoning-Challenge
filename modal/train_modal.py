"""Train the Nemotron LoRA adapter on a Modal A100-80GB (bf16, no offload).

Self-contained: pulls the competition data with your Kaggle token, builds the SFT
data, trains, and writes lora_adapter/ to a Modal Volume you can download.

Setup (once)
------------
    pip install modal
    modal setup
    # one secret holding your Kaggle API token (and optionally an HF token if the
    # base model download needs auth):
    modal secret create nemotron KAGGLE_TOKEN=KGAT_xxx   # add HF_TOKEN=hf_xxx if needed

Run
---
    modal run modal/train_modal.py
    # then download the adapter:
    modal volume get nemotron-out lora_adapter ./lora_adapter

You can run these from anywhere, including a Google Colab cell
(`!pip install modal && modal token set ... && modal run ...`) — the A100 is
Modal's, so Colab's own GPU is irrelevant.
"""

from __future__ import annotations

import modal

REPO = "https://github.com/SebAustin/NVIDIA-Nemotron-Model-Reasoning-Challenge"
BRANCH = "build/nemotron-pipeline"
TORCH = "torch==2.6.0"          # cu124; mamba/causal-conv1d built from source against it
CUDA_TAG = "12.4.1-devel-ubuntu22.04"   # devel image => nvcc available to build mamba

# Build mamba_ssm + causal_conv1d FROM SOURCE against this exact torch (prebuilt
# wheels' c10 symbols don't reliably match), targeting A100 (sm_80).
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        TORCH,
        "transformers>=4.45,<5", "peft", "trl", "datasets", "accelerate",
        "bitsandbytes", "psutil", "pandas", "numpy", "sentencepiece",
        "huggingface_hub", "hf_transfer", "einops",
        "ninja", "packaging", "wheel", "setuptools",  # build tools for --no-build-isolation
    )
    .run_commands(
        f"git clone -b {BRANCH} {REPO} /repo",
        "CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0 "
        "CUDA_HOME=/usr/local/cuda pip install --no-build-isolation causal-conv1d",
        "MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0 "
        "CUDA_HOME=/usr/local/cuda pip install --no-build-isolation mamba-ssm",
        "python -c 'import causal_conv1d, mamba_ssm; print(\"mamba build OK\")'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)

app = modal.App("nemotron-lora")
out_vol = modal.Volume.from_name("nemotron-out", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("nemotron")],
    volumes={"/out": out_vol, "/root/.cache/huggingface": hf_vol},
)
def train():
    import os
    import subprocess
    import urllib.request

    os.chdir("/repo")
    os.makedirs("data", exist_ok=True)

    # 1) competition data via the Kaggle KGAT bearer token
    tok = os.environ["KAGGLE_TOKEN"]
    comp = "nvidia-nemotron-model-reasoning-challenge"
    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{comp}/train.csv"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req) as r, open("data/train.csv", "wb") as f:
        f.write(r.read())
    print("downloaded train.csv:", os.path.getsize("data/train.csv"), "bytes")

    # 2) build the SFT data (EDA + prepare)
    subprocess.run(["python", "scripts/01_eda.py", "--data-dir", "data"], check=True)
    subprocess.run(["python", "scripts/02_prepare_data.py", "--data-dir", "data"], check=True)

    # 3) train bf16 on the A100 (no 8-bit, no offload), save to the output Volume
    env = {**os.environ, "NEMOTRON_MAX_MEMORY_GPU": "78GiB"}
    subprocess.run(
        ["python", "scripts/03_train_lora.py",
         "--data-path", "data/train_sft.jsonl",
         "--output-dir", "/out/lora_adapter",
         "--no-8bit", "--no-smoke"],
        check=True, env=env,
    )
    out_vol.commit()
    print("DONE -> volume 'nemotron-out':/lora_adapter")
    print("download with:  modal volume get nemotron-out lora_adapter ./lora_adapter")


@app.local_entrypoint()
def main():
    train.remote()
