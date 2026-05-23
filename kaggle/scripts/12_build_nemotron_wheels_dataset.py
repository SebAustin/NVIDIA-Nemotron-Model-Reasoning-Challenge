#!/usr/bin/env python3
"""Build a single Kaggle Dataset that contains every wheel needed to run the
Nemotron training + inference pipeline fully offline (no internet on the
Kaggle GPU notebook).

Why this exists
---------------
Kaggle GPU notebooks frequently can't enable internet (phone verification,
competition rules, etc.). Both kaggle_training.ipynb and kaggle_submission.ipynb
need a handful of pip packages that aren't in Kaggle's stock image. We pre-stage
them as wheels in a Kaggle Dataset so the notebooks can install with
``--no-index --find-links /kaggle/input/nemotron-wheels``.

Usage
-----
First-time create:

    python kaggle/scripts/12_build_nemotron_wheels_dataset.py --create

Subsequent updates:

    python kaggle/scripts/12_build_nemotron_wheels_dataset.py -m "bump versions"

Notes
-----
* Run this from your laptop (with internet). It uses your own ``pip`` to
  download the right linux/x86_64 wheels for Python 3.12 (Kaggle's runtime).
* Requires the ``kaggle`` CLI and credentials at ``~/.kaggle/kaggle.json``.
* ``bitsandbytes``, ``vllm``, and a few transitive deps are CUDA/Linux specific;
  this script asks pip explicitly for ``manylinux2014_x86_64`` wheels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _ensure_ca_bundle() -> None:
    """Workaround for Homebrew Python 3.14 + pip certifi path flakiness.

    Both pip's vendored certifi and the kaggle CLI's certifi are symlinks
    into ``/opt/homebrew/Cellar/...`` which can break mid-run if brew
    re-installs certifi/python in the background. Pin REQUESTS_CA_BUNDLE
    (and friends) to a stable system bundle so both processes succeed.
    """
    candidates = [
        "/etc/ssl/cert.pem",
        "/usr/local/etc/openssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
    ]
    try:
        import certifi  # type: ignore[import-not-found]
        candidates.insert(0, certifi.where())
    except Exception:
        pass
    for c in candidates:
        if c and Path(c).is_file():
            for var in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
                os.environ.setdefault(var, c)
            print(f"[ca-bundle] using {c}", flush=True)
            return
    print("[ca-bundle] WARNING: no usable CA bundle found.", flush=True)

# Package set required by both notebooks.
TRAINING_REQS = [
    "transformers>=4.45.0,<5",
    "peft>=0.13.0",
    "trl>=0.11.0",
    "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0",
    "datasets>=3.0.0",
    "sentencepiece",
    "tokenizers",
    "safetensors",
]

INFERENCE_REQS = [
    # vllm 0.17.0 (released March 2026) is built for torch 2.10, which is
    # exactly what Kaggle's RTX Pro 6000 GPU container ships. Using this
    # version avoids overriding Kaggle's torch/xformers/triton/numpy/etc.
    # and the cascading ABI mismatches that come with that.
    "vllm==0.17.0",
]

# torch 2.4.0+cu121 + matching torchvision so the TRAINING notebook can pin
# torch back from Kaggle's bleeding-edge base image (torch 2.10) to the
# version that the rest of the prebuilt CUDA wheel stack
# (mamba_ssm / causal_conv1d / xformers 0.0.27.post2 / vllm 0.6.3 /
# triton 3.1) was built against. PyPI doesn't host the `+cu121` local-version
# wheels; they live on the pytorch.org index.
TORCH_PIN_REQS = [
    "torch==2.4.0+cu121",
    "torchvision==0.19.0+cu121",
]
TORCH_PIN_INDEX = "https://download.pytorch.org/whl/cu121"

# Kaggle CLI itself, so kaggle_training.ipynb Phase 7 can publish the LoRA
# adapter without needing internet.
TOOLING_REQS = [
    "kaggle",
]

DEFAULT_PYTHON_VERSION = "3.12"
# Different packages on PyPI publish under different manylinux tags. We pass
# them all so pip can match whichever each wheel uses (vllm uses manylinux1,
# bitsandbytes uses manylinux_2_24, transformers/torch usually manylinux1
# or manylinux2014, etc.).
DEFAULT_PLATFORMS = [
    "manylinux1_x86_64",
    "manylinux2010_x86_64",
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_24_x86_64",
    "manylinux_2_27_x86_64",
    "manylinux_2_28_x86_64",
    "manylinux_2_31_x86_64",
    "manylinux_2_34_x86_64",
    "linux_x86_64",
]


def _run(cmd: list[str], **kw) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


PUBLIC_PYPI = "https://pypi.org/simple/"

# Some vLLM transitive deps publish only sdists on PyPI (notably pyairports,
# pulled in by outlines). Since they're pure Python we build the wheel locally
# with `pip wheel` first, then point the platform-restricted download at the
# wheelhouse via --find-links so pip can satisfy them without sdists.
PURE_PY_SDIST_ONLY_DEPS = [
    "pyairports",
]


def _pip_wheel(spec: str, dest: Path, index_url: str) -> None:
    _run(
        [
            sys.executable, "-m", "pip", "wheel", spec,
            "--wheel-dir", str(dest),
            "--no-deps",
            "--index-url", index_url,
        ]
    )


def _pip_download(reqs: list[str], dest: Path, py: str, platforms: list[str],
                  index_url: str) -> None:
    """Download manylinux + python <py> wheels for <reqs> into <dest>.

    Pins the index URL because corporate PyPI mirrors often don't proxy the
    bitsandbytes/vllm/torch CUDA wheels we need for Kaggle.
    """
    plat_args: list[str] = []
    for p in platforms:
        plat_args += ["--platform", p]
    base = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(dest),
        "--index-url", index_url,
        "--prefer-binary",
        "--python-version", py,
        "--only-binary=:all:",
        "--find-links", str(dest),
    ]
    _run(base + plat_args + reqs)


def _pip_download_no_deps(reqs: list[str], dest: Path, py: str,
                          platforms: list[str], index_url: str) -> None:
    """Download wheels for <reqs> WITHOUT resolving deps, into <dest>.

    Used to override pin conflicts (e.g. vllm pins numpy<2 but xformers
    cp312 + Kaggle's torch C extensions need numpy>=2 ABI).
    """
    plat_args: list[str] = []
    for p in platforms:
        plat_args += ["--platform", p]
    _run(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(dest),
            "--index-url", index_url,
            "--prefer-binary",
            "--python-version", py,
            "--only-binary=:all:",
            "--no-deps",
            *plat_args,
            *reqs,
        ]
    )


def _resolve_default_slug() -> str | None:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.is_file():
        try:
            return f"{json.loads(cred.read_text())['username']}/nemotron-wheels"
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        default=_resolve_default_slug(),
        help="Dataset slug, e.g. 'username/nemotron-wheels'. Defaults to "
        "<your username>/nemotron-wheels if ~/.kaggle/kaggle.json is present.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="First publish (datasets create). Without this flag, a new "
        "version is pushed.",
    )
    parser.add_argument(
        "--message", "-m",
        default="update nemotron wheels",
    )
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        help=f"Default {DEFAULT_PYTHON_VERSION} (Kaggle GPU runtime).",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Don't include vLLM wheels (smaller dataset; only training will work).",
    )
    parser.add_argument(
        "--index-url",
        default=PUBLIC_PYPI,
        help="Override the pip index URL. Defaults to https://pypi.org/simple/ "
        "to bypass corporate mirrors that don't proxy CUDA wheels.",
    )
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    _ensure_ca_bundle()

    if not args.slug or "/" not in args.slug:
        parser.error("--slug required, e.g. 'username/nemotron-wheels'")

    staging = Path(tempfile.mkdtemp(prefix="kaggle-nemotron-wheels-"))
    print(f"Staging dir: {staging}", flush=True)

    reqs = TRAINING_REQS + TOOLING_REQS + ([] if args.skip_vllm else INFERENCE_REQS)

    # Pre-build wheels for pure-Python sdist-only deps so the platform-locked
    # download below can resolve them via --find-links.
    if not args.skip_vllm:
        for dep in PURE_PY_SDIST_ONLY_DEPS:
            print(f"\n[pip wheel] {dep}")
            _pip_wheel(dep, staging, args.index_url)

    # Download torch 2.4.0+cu121 + torchvision 0.19.0+cu121 from pytorch.org.
    # These are needed by the TRAINING notebook's _pin_torch_2_4() step when
    # the Kaggle base image torch is newer than 2.4 (which it now is, on the
    # Python 3.12 RTX Pro 6000 image). --no-deps because torch's metadata
    # pulls numpy and we manage numpy explicitly elsewhere.
    print("\n[pip download] torch 2.4.0+cu121 + torchvision 0.19.0+cu121 "
          "(from pytorch.org/whl/cu121)")
    _pip_download_no_deps(
        TORCH_PIN_REQS, staging, args.python_version, DEFAULT_PLATFORMS,
        TORCH_PIN_INDEX,
    )

    _pip_download(
        reqs, staging, args.python_version, DEFAULT_PLATFORMS, args.index_url
    )

    # Write the metadata Kaggle CLI needs.
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": args.slug.split("/", 1)[1],
                "id": args.slug,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
    )

    files = [p for p in staging.iterdir() if p.is_file()]
    wheels = [p for p in files if p.suffix in (".whl", ".tar.gz")]
    print(f"\nDownloaded {len(wheels)} wheel/sdist files "
          f"({sum(p.stat().st_size for p in wheels) / 1e6:.0f} MB):", flush=True)
    for w in sorted(wheels):
        print(f"  {w.name}  ({w.stat().st_size/1e6:.1f} MB)")

    try:
        if args.create:
            cmd = ["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"]
            if args.public:
                cmd.append("--public")
            _run(cmd)
        else:
            _run(["kaggle", "datasets", "version", "-p", str(staging),
                  "-m", args.message, "-r", "zip"])
        print(f"\nDataset URL: https://www.kaggle.com/datasets/{args.slug}")
        print(
            "Next: in kaggle_training.ipynb (and kaggle_submission.ipynb), "
            "click 'Add Data -> Your Datasets' and attach this dataset."
        )
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
