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
TORCH = "torch==2.7.0"          # 2.7+ Linux wheels are cxx11abi=TRUE (provides the
                                # __cxx11 c10 symbols the mamba wheels need; 2.6 is abiFALSE)

# Prebuilt mamba/causal-conv1d wheels for torch 2.7 cp312, cxx11abi=TRUE.
_GH = "https://github.com/{}/releases/download/{}/{}"
CAUSAL_WHL = _GH.format(
    "Dao-AILab/causal-conv1d", "v1.6.2.post1",
    "causal_conv1d-1.6.2.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
MAMBA_WHL = _GH.format(
    "state-spaces/mamba", "v2.3.2.post1",
    "mamba_ssm-2.3.2.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        TORCH,
        "transformers>=4.45,<5", "peft", "trl", "datasets", "accelerate",
        "bitsandbytes", "psutil", "pandas", "numpy", "sentencepiece",
        "huggingface_hub", "hf_transfer", "einops",
    )
    .run_commands(
        f"git clone -b {BRANCH} {REPO} /repo",
        f"pip install --no-deps '{CAUSAL_WHL}' '{MAMBA_WHL}'",
        # verify the .so symbols resolve against this torch (no GPU needed)
        "python -c 'import causal_conv1d, mamba_ssm; print(\"mamba import OK\")'",
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
