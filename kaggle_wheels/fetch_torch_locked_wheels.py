#!/usr/bin/env python3
"""Fetch torch-ABI-locked wheels (mamba-ssm + causal-conv1d) for the LIVE env.

Run this ONCE on a Kaggle notebook with **internet ON**, in the directory that
holds the rest of the offline wheels (e.g. the unzipped dataset). It detects the
running torch / CUDA / Python / C++ ABI and downloads the exact-matching prebuilt
wheels from the official GitHub releases, so they are guaranteed to import on this
machine. Then "Save Version" to bake them into your offline dataset.

Why this can't be done from a Mac: these two packages publish a wheel per
(torch X.Y, cuda major, cpython, cxx11abi) combination, and Kaggle's torch build
is only knowable on Kaggle (its image now tracks Colab's, Python 3.12).

    python fetch_torch_locked_wheels.py            # into current dir
    python fetch_torch_locked_wheels.py --dest .   # explicit
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPOS = {
    "mamba_ssm": ("state-spaces/mamba", "v2.3.2.post1"),
    "causal_conv1d": ("Dao-AILab/causal-conv1d", "v1.6.2.post1"),
}


def _api(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "nemotron-wheels"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def env_tags():
    import torch

    vi = sys.version_info
    py = f"cp{vi.major}{vi.minor}"
    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    cuda = (torch.version.cuda or "12.0").split(".")[0]
    cu = f"cu{cuda}"
    try:
        abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"
    except Exception:
        abi = "TRUE"
    print(f"[env] python={py} torch={torch_mm} {cu} cxx11abi={abi} (torch {torch.__version__})")
    return py, torch_mm, cu, abi


def list_assets(repo: str, tag: str):
    try:
        rel = _api(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    except Exception:
        rel = _api(f"https://api.github.com/repos/{repo}/releases/latest")
    return [(a["name"], a["browser_download_url"]) for a in rel.get("assets", [])]


def choose(assets, py, torch_mm, cu, abi, arch="x86_64"):
    """Pick the best asset: exact match first, then relax abi, then nearest torch
    minor (same major), then any cu major."""
    cands = [(n, u) for n, u in assets
             if f"-{py}-{py}-linux_{arch}.whl" in n and "+cu" in n]

    def score(name):
        # returns a tuple; higher is better, or None to reject
        try:
            tag = name.split("+", 1)[1].split("-", 1)[0]  # e.g. cu12torch2.6cxx11abiTRUE
        except Exception:
            return None
        exact_cu = f"{cu}torch" in tag
        exact_abi = f"cxx11abi{abi}" in tag
        exact_torch = f"torch{torch_mm}cxx11abi" in tag
        return (int(exact_torch), int(exact_cu), int(exact_abi), name)

    scored = [(score(n), n, u) for n, u in cands]
    scored = [s for s in scored if s[0] is not None]
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1], scored[0][2]


def download(url: str, dest: Path) -> Path:
    out = dest / url.split("/")[-1]
    print(f"[get] {out.name}")
    urllib.request.urlretrieve(url, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("."))
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    py, torch_mm, cu, abi = env_tags()
    ok = True
    for pkg, (repo, tag) in REPOS.items():
        assets = list_assets(repo, tag)
        pick = choose(assets, py, torch_mm, cu, abi)
        if not pick:
            print(f"[WARN] no {pkg} wheel matched {py}/{cu}/torch{torch_mm}. "
                  f"Assets available: {[n for n, _ in assets][:8]} ...")
            ok = False
            continue
        name, url = pick
        if f"torch{torch_mm}cxx11abi{abi}" not in name:
            print(f"[note] exact torch{torch_mm}/abi{abi} not found; using nearest: {name}")
        download(url, args.dest)
    print("\nDone." if ok else "\nDone with warnings (see above).")
    print("Next: install everything offline with install_offline.sh / the README command.")


if __name__ == "__main__":
    main()
