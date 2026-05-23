"""Resolve Nemotron Nano weights on Kaggle (competition Model input, nested mounts)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _hf_model_dir_has_tokenizer(d: Path) -> bool:
    if not (d / "config.json").is_file():
        return False
    return (
        (d / "tokenizer.json").is_file()
        or (d / "tokenizer_config.json").is_file()
        or (d / "tokenizer.model").is_file()
        or (d / "spiece.model").is_file()
    )


def _config_looks_nemotron_nano(config_path: Path) -> bool:
    try:
        with config_path.open(encoding="utf-8") as f:
            c = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    blob = (
        str(c.get("model_type", ""))
        + " "
        + str(c.get("architectures", []))
    ).lower()
    return "nemotron" in blob and "nano" in blob


def find_kaggle_competition_nemotron_dir() -> str | None:
    """
    Find a local folder with config.json + tokenizer files under /kaggle/input.
    Handles metric/.../transformers/default and arbitrary slug directory names.
    """
    for direct in (
        Path("/kaggle/input/nemotron-3-nano-30b-a3b-bf16"),
        Path("/kaggle/input/metric-nemotron-3-nano-30b-a3b-bf16"),
    ):
        if _hf_model_dir_has_tokenizer(direct):
            return str(direct.resolve())

    root_in = Path("/kaggle/input")
    if not root_in.is_dir():
        return None

    scored: list[tuple[int, int, Path]] = []

    def path_score(p: Path) -> int:
        low = str(p).lower()
        s = 0
        if "nemotron" in low:
            s += 15
        if "nano" in low:
            s += 8
        if "metric" in low or "models" in low:
            s += 4
        if "transformers" in low:
            s += 2
        cfg = p / "config.json"
        if cfg.is_file() and _config_looks_nemotron_nano(cfg):
            s += 40
        return s

    try:
        for cfg in root_in.rglob("config.json"):
            parent = cfg.parent
            if not _hf_model_dir_has_tokenizer(parent):
                continue
            rp = parent.resolve()
            scored.append((path_score(rp), len(str(rp)), rp))
    except OSError:
        return None

    if not scored:
        return None

    scored.sort(key=lambda t: (-t[0], t[1]))
    best_score, _, best_path = scored[0]
    if best_score > 0:
        return str(best_path)
    if len(scored) == 1:
        return str(scored[0][2])
    return None


def resolve_model_weights_path(cli_path: str) -> str:
    """
    Prefer a local directory with config.json; then env NEMOTRON_MODEL_PATH;
    then Kaggle competition mount. Otherwise keep cli_path (Hub id or path).
    """
    p = Path(cli_path)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    envp = os.environ.get("NEMOTRON_MODEL_PATH", "").strip()
    if envp:
        ep = Path(envp)
        if ep.is_dir() and (ep / "config.json").is_file():
            return str(ep.resolve())
    found = find_kaggle_competition_nemotron_dir()
    if found:
        return found
    return cli_path
