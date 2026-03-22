"""Shared helpers for loading Nemotron model/tokenizer from local path or HF."""
import os


def resolve_model_path(path: str) -> str:
    """Resolve model path: if dir has config.json use it; else check subfolders (e.g. NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)."""
    if not path or not os.path.isdir(path):
        return path
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    for name in os.listdir(path):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
            return sub
    return path


def local_load_kwargs(path: str) -> dict:
    """Use local_files_only=True when path is a local dir to avoid HF downloads (saves disk on Kaggle)."""
    return {"local_files_only": True} if path and os.path.isdir(path) else {}
