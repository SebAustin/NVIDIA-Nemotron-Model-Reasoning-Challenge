"""Average multiple PEFT LoRA adapters into a single adapter.

Per Kaggle Grandmasters Playbook tip #7 (multi-seed training): training the
same architecture with different random seeds and averaging predictions
yields a measurable accuracy bump. Kaggle only accepts ONE LoRA adapter, so
prediction-time averaging isn't possible; instead we average the LoRA
weights themselves before submission.

LoRA matrices are linear projections (down: A, up: B; ΔW = B @ A scaled by
alpha/r). Averaging multiple `(A, B)` pairs that started from the same base
model and similar configs gives a stable, well-behaved combined adapter
without retraining. Empirically (Wortsman et al., 'Model Soups' 2022),
weight averaging across runs that share initialization is roughly
equivalent to a small ensemble.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


def _list_safetensor_files(adapter_dir: Path) -> List[Path]:
    candidates = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if not candidates:
        candidates = sorted(adapter_dir.glob("adapter_model*.bin"))
    return candidates


def _load_state(path: Path) -> Dict[str, "torch.Tensor"]:  # noqa: F821 (lazy)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    import torch

    return torch.load(path, map_location="cpu")


def _save_state(state: Dict[str, "torch.Tensor"], out_path: Path) -> None:  # noqa: F821
    if out_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(out_path))
    else:
        import torch

        torch.save(state, out_path)


def average_adapters(
    adapter_paths: Iterable[Path],
    output_dir: Path,
    weights: List[float] | None = None,
) -> Dict[str, float]:
    """Average the LoRA `A` / `B` matrices across N adapter directories.

    Args:
        adapter_paths: directories containing PEFT adapters (each must have
            `adapter_config.json` and either `adapter_model.safetensors` or
            `adapter_model.bin`).
        output_dir: destination for the averaged adapter. The first adapter's
            `adapter_config.json` is copied verbatim (configs must match).
        weights: optional per-adapter weights (will be normalized). Defaults
            to uniform 1/N.

    Returns:
        Dict with summary fields ('n_adapters', 'n_tensors', 'output_dir').
    """
    import torch

    paths = [Path(p) for p in adapter_paths]
    if not paths:
        raise ValueError("Need at least one adapter path")

    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    else:
        if len(weights) != len(paths):
            raise ValueError("weights must have same length as adapter_paths")
        s = float(sum(weights))
        if s <= 0:
            raise ValueError("weights must sum to a positive number")
        weights = [w / s for w in weights]

    # Validate configs match.
    cfgs = []
    for p in paths:
        cfg_path = p / "adapter_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing adapter_config.json in {p}")
        cfgs.append(json.loads(cfg_path.read_text()))

    ref = cfgs[0]
    for i, c in enumerate(cfgs[1:], start=1):
        for k in ("r", "lora_alpha", "target_modules", "task_type", "base_model_name_or_path"):
            if c.get(k) != ref.get(k):
                # 'target_modules' may be a list with stable order
                if isinstance(c.get(k), list) and isinstance(ref.get(k), list):
                    if sorted(c[k]) == sorted(ref[k]):
                        continue
                raise ValueError(
                    f"Adapter config mismatch on key '{k}': "
                    f"{paths[0]} -> {ref.get(k)} vs {paths[i]} -> {c.get(k)}"
                )

    # Load all states.
    states: List[Dict[str, "torch.Tensor"]] = []
    for p in paths:
        files = _list_safetensor_files(p)
        if not files:
            raise FileNotFoundError(f"No adapter weights file in {p}")
        if len(files) > 1:
            raise NotImplementedError(
                f"Sharded adapter weights not supported (found {len(files)} in {p})"
            )
        states.append(_load_state(files[0]))

    keys = set(states[0].keys())
    for i, s in enumerate(states[1:], start=1):
        if set(s.keys()) != keys:
            missing = keys.symmetric_difference(s.keys())
            raise ValueError(
                f"Adapter tensor keys differ between {paths[0]} and {paths[i]}: "
                f"{sorted(missing)[:5]}..."
            )

    averaged: Dict[str, "torch.Tensor"] = {}
    for k in keys:
        ref_t = states[0][k]
        if not torch.is_floating_point(ref_t):
            # Non-float (e.g. integer index) tensors: take first; they should match.
            averaged[k] = ref_t.clone()
            continue
        acc = torch.zeros_like(ref_t, dtype=torch.float32)
        for w, st in zip(weights, states):
            t = st[k]
            if t.shape != ref_t.shape:
                raise ValueError(f"Shape mismatch for {k}: {t.shape} vs {ref_t.shape}")
            acc = acc + w * t.to(torch.float32)
        averaged[k] = acc.to(ref_t.dtype)

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_safe = output_dir / "adapter_model.safetensors"
    _save_state(averaged, out_safe)
    shutil.copy2(paths[0] / "adapter_config.json", output_dir / "adapter_config.json")

    # Carry over README.md / tokenizer files if present (PEFT convention).
    for fname in ("README.md", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        src = paths[0] / fname
        if src.is_file():
            shutil.copy2(src, output_dir / fname)

    return {
        "n_adapters": float(len(paths)),
        "n_tensors": float(len(averaged)),
        "output_dir": str(output_dir),
    }


__all__ = ["average_adapters"]
