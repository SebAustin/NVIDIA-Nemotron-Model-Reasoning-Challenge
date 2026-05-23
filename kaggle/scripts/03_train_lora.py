#!/usr/bin/env python3
"""Phase 3: QLoRA SFT with Unsloth (preferred) or Hugging Face PEFT fallback."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import site
import stat
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ensure_pip_target_pth_processed() -> None:
    """
    `pip install --target WORK_ROOT/.pip_tf4` writes `nvidia_cutlass_dsl.pth` into that directory.
    Prepended PYTHONPATH alone does not process `.pth` files; `site.addsitedir` does, which exposes
    the real CuTe DSL `cutlass` package under `nvidia_cutlass_dsl/python_packages`.
    """
    seen: set[str] = set()
    candidates: list[Path] = []
    for raw in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        p = Path(raw.strip()).expanduser().resolve()
        if p.is_dir() and any(p.glob("*.pth")):
            candidates.append(p)
    cwd_tf4 = (Path.cwd() / ".pip_tf4").resolve()
    if cwd_tf4.is_dir() and any(cwd_tf4.glob("*.pth")):
        candidates.append(cwd_tf4)
    for d in candidates:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        site.addsitedir(key)
        print(f"[03_train_lora] site.addsitedir({d}) (.pth e.g. nvidia-cutlass-dsl → import cutlass)", flush=True)


def _prepare_cutlass_import_for_nemotron() -> None:
    """
    Nemotron needs NVIDIA's CuTe DSL `cutlass` package (`Int32`, `cutlass.cute`, …) from
    `nvidia-cutlass-dsl` + `nvidia-cutlass-dsl-libs-base` (see `.pth` under site-packages).

    Never alias `sys.modules['cutlass']` to `cutlass_cppgen`: model code does
    `from cutlass import Int32`, which only exists on the DSL package. Undo that alias if present
    (e.g. from an older notebook) so a real `import cutlass` can load.
    """
    v = os.environ.get("NEMOTRON_DISABLE_CUTLASS_SHIM", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return
    existing = sys.modules.get("cutlass")
    if existing is not None and hasattr(existing, "Int32"):
        return
    if existing is not None and getattr(existing, "__name__", None) == "cutlass_cppgen":
        del sys.modules["cutlass"]
        sys.modules.pop("cutlass.cute", None)


def _neutralize_wandb_before_trl_import() -> None:
    """
    TRL imports SFTTrainer → BaseTrainer calls transformers' is_wandb_available(), which
    imports wandb. Colab/Kaggle images sometimes ship a broken wandb/protobuf pair
    (ImportError: wandb.proto...). Unless USE_WANDB=1, skip wandb entirely.
    """
    if os.environ.get("USE_WANDB", "").lower() in ("1", "true", "yes"):
        return
    os.environ.setdefault("WANDB_DISABLED", "true")
    try:
        import transformers.integrations.integration_utils as _iu

        _iu.is_wandb_available = lambda: False  # type: ignore[method-assign]
    except Exception:
        pass


_neutralize_wandb_before_trl_import()

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer


def _sft_config_kwargs(max_seq_length: int, *, packing: bool = False) -> dict:
    """TRL renames max_seq_length → max_length on SFTConfig in recent versions."""
    sig = inspect.signature(SFTConfig.__init__).parameters
    out: dict = {}
    if "max_seq_length" in sig:
        out["max_seq_length"] = max_seq_length
    elif "max_length" in sig:
        out["max_length"] = max_seq_length
    if "dataset_text_field" in sig:
        out["dataset_text_field"] = None
    if "packing" in sig:
        out["packing"] = packing
    return out


def _kwargs_for_callable(func, kwargs: dict) -> dict:
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _nemotron_kaggle_patches_enabled() -> bool:
    v = os.environ.get("NEMOTRON_KAGGLE_PATCHES", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return Path("/kaggle/input").exists()


def _resolve_ptxas_blackwell_src() -> Path | None:
    """Ryan Holbrook utility notebook — see https://www.kaggle.com/code/ryanholbrook/nvidia-utility-script"""
    env = (os.environ.get("NVIDIA_UTILITY_SCRIPT_DIR") or "").strip()
    if env:
        p = Path(env) / "triton/backends/nvidia/bin/ptxas-blackwell"
        if p.is_file():
            return p
    for _slug in ("nvidia_utility_script", "nvidia-utility-script"):
        fixed = (
            Path("/kaggle/usr/lib/notebooks/ryanholbrook")
            / _slug
            / "triton/backends/nvidia/bin/ptxas-blackwell"
        )
        if fixed.is_file():
            return fixed
    root = Path("/kaggle/input")
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            q = child / "triton/backends/nvidia/bin/ptxas-blackwell"
            if q.is_file():
                return q
    return None


def apply_nemotron_kaggle_env_patches() -> None:
    """Copy ptxas-blackwell to /tmp when the NVIDIA utility bundle is present."""
    if not _nemotron_kaggle_patches_enabled():
        return
    src = _resolve_ptxas_blackwell_src()
    dst = Path("/tmp/ptxas-blackwell")
    if src is None:
        return
    try:
        shutil.copy2(src, dst)
        mode = dst.stat().st_mode
        dst.chmod(mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", str(dst))
        os.environ.setdefault("TRITON_PTXAS_PATH", str(dst))
        print(f"[03_train_lora] Triton: copied ptxas-blackwell -> {dst} (from {src})", flush=True)
    except OSError as e:
        print(f"[03_train_lora] Triton env patch skipped: {e!r}", flush=True)


def _patch_moe_dtype_mismatch() -> None:
    """Fix index_add_ BFloat16/Float mismatch in Nemotron-H MoE layers.

    The MoE routing computes weighted_output in float32 (via autocast) but
    final_hidden_states stays in bf16, causing:
        RuntimeError: index_add_(): self (BFloat16) and source (Float)
        must have the same scalar type
    We monkey-patch every NemotronH class that has an `moe` method to cast
    weighted_output to the target dtype before index_add_.
    """
    import types

    patched = 0
    for name, mod in list(sys.modules.items()):
        if "modeling_nemotron" not in name:
            continue
        for attr_name in dir(mod):
            cls = getattr(mod, attr_name, None)
            if not isinstance(cls, type):
                continue
            orig_moe = getattr(cls, "moe", None)
            if orig_moe is None or not callable(orig_moe):
                continue
            import inspect
            src = ""
            try:
                src = inspect.getsource(orig_moe)
            except (OSError, TypeError):
                pass
            if "index_add_" not in src:
                continue

            _orig = orig_moe

            def _safe_moe(self, hidden_states, topk_indices, topk_weights, _orig_fn=_orig):
                result = _orig_fn(self, hidden_states, topk_indices, topk_weights)
                return result

            try:
                src_lines = src.split("\n")
                indent = ""
                for line in src_lines:
                    stripped = line.lstrip()
                    if stripped.startswith("final_hidden_states.index_add_"):
                        indent = line[: len(line) - len(stripped)]
                        break

                new_src = src.replace(
                    "final_hidden_states.index_add_(0, token_indices, weighted_output)",
                    "final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))",
                )
                if new_src != src:
                    ns: dict = {}
                    exec(compile(new_src, f"<patched {attr_name}.moe>", "exec"), mod.__dict__, ns)
                    patched_fn = ns.get("moe")
                    if patched_fn is not None:
                        setattr(cls, "moe", patched_fn)
                        patched += 1
                        continue
            except Exception:
                pass

            def _make_wrapper(original):
                def _wrapper(self, hidden_states, topk_indices, topk_weights):
                    orig_index_add = torch.Tensor.index_add_

                    def _safe_index_add(self_t, dim, index, source, *a, **kw):
                        return orig_index_add(self_t, dim, index, source.to(self_t.dtype), *a, **kw)

                    torch.Tensor.index_add_ = _safe_index_add
                    try:
                        return original(self, hidden_states, topk_indices, topk_weights)
                    finally:
                        torch.Tensor.index_add_ = orig_index_add

                return _wrapper

            setattr(cls, "moe", _make_wrapper(_orig))
            patched += 1

    if patched:
        print(f"[03_train_lora] MoE dtype patch applied ({patched} class(es))", flush=True)


def patch_nemotron_modules_after_load() -> None:
    """Pure RMSNorm + disable Nemotron fast CUDA path (Triton/ptxas issues on some Kaggle images)."""
    if not _nemotron_kaggle_patches_enabled():
        return
    import torch.nn.functional as F

    def _pure_rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        upcast=True,
    ):
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype)

    touched = 0
    for name, mod in list(sys.modules.items()):
        if not name:
            continue
        if hasattr(mod, "rmsnorm_fn"):
            mod.rmsnorm_fn = _pure_rmsnorm_fn
            touched += 1
        if "modeling_nemotron" in name and hasattr(mod, "is_fast_path_available"):
            try:
                setattr(mod, "is_fast_path_available", False)
                touched += 1
            except Exception:
                pass

    _patch_moe_dtype_mismatch()

    if touched:
        print(f"[03_train_lora] Nemotron runtime patches applied (hooks={touched})", flush=True)


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# Bump when changing load/offload behavior (check logs on Kaggle to confirm sync).
_TRAIN_LORA_BUILD = "2026-04-07+high-gpu-low-cpu"

# Default HF-style targets (good for many Llama-like MoE checkpoints)
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Kaggle competition baseline (matches public notebooks using Nemotron-3 Nano PyTorch modules)
# NemotronH is hybrid Mamba+attn+MoE: in_proj/out_proj (Mamba), q/k/v/o_proj (attn),
# gate/up/down_proj (MoE experts). Wider target => more expressive adapter.
KAGGLE_LORA_TARGET_REGEX = (
    r".*\.(in_proj|out_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from kaggle_nemotron_paths import resolve_model_weights_path  # noqa: E402


def _ensure_base_model_in_adapter(save_dir: Path, base_id: str) -> None:
    cfg_path = save_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = base_id
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_model_unsloth(max_seq_length: int):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer, "unsloth"


def _lora_target_modules(mode: str):
    if mode == "kaggle_nemotron":
        return KAGGLE_LORA_TARGET_REGEX
    if mode == "hf_linear":
        return LORA_TARGET_MODULES
    raise ValueError(f"Unknown lora target mode: {mode}")


def _from_pretrained_dtype_kw(model_cls) -> dict:
    """Use `dtype` when supported (transformers deprecation of torch_dtype)."""
    params = inspect.signature(model_cls.from_pretrained).parameters
    if "dtype" in params:
        return {"dtype": torch.bfloat16}
    if "torch_dtype" in params:
        return {"torch_dtype": torch.bfloat16}
    return {}


def _cuda_total_bytes(device_index: int = 0) -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(device_index).total_memory)
    except Exception:
        return None


def _parse_gib_from_memory_string(val) -> float | None:
    """Parse Accelerate-style memory strings (e.g. '14GiB', '8000MiB') to GiB float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) / (1024**3) if val > 1_000_000_000 else float(val)
    s = str(val).strip().upper().replace(" ", "")
    try:
        if s.endswith("GIB"):
            return float(s[:-3])
        if s.endswith("MIB"):
            return float(s[:-3]) / 1024.0
        if s.endswith("G"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1024.0
    except ValueError:
        return None
    return None


def _clamp_max_memory_cpu_for_host_ram(mm: dict, *, quantize: bool) -> dict:
    """
    Safety-net: ensure max_memory['cpu'] does not exceed what the host can handle.
    With the high-GPU / low-CPU strategy the requested cpu cap is typically already
    small (2 GiB), so this rarely fires.  It remains useful when users override
    --max-memory-json with a large cpu value.
    """
    v = os.environ.get("NEMOTRON_SKIP_CPU_MAX_MEMORY_CLAMP", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return mm
    if "cpu" not in mm:
        return mm
    user_gib = _parse_gib_from_memory_string(mm["cpu"])
    if user_gib is None:
        return mm
    try:
        import psutil

        vm = psutil.virtual_memory()
        avail_gib = vm.available / (1024**3)
        total_gib = vm.total / (1024**3)
        avail_cap = max(0.75, avail_gib * 0.20)
        total_cap = max(0.75, total_gib * 0.11)
        safe_ceiling = max(0.75, min(avail_cap, total_cap))
        _env_hard = os.environ.get("NEMOTRON_SHARD_LOAD_MAX_CPU_GIB", "").strip()
        if _env_hard:
            safe_ceiling = min(safe_ceiling, float(_env_hard))
        cap = min(user_gib, safe_ceiling)
    except Exception:
        cap = min(user_gib, 1.75)
    if user_gib <= cap + 0.05:
        return mm
    out = dict(mm)
    cr = max(0.75, round(cap, 2))
    if abs(cr - int(cr)) < 0.001:
        out["cpu"] = f"{int(round(cr))}GiB"
    else:
        out["cpu"] = f"{cr:g}GiB"
    print(
        f"[03_train_lora] clamped max_memory['cpu'] {mm['cpu']!r} -> {out['cpu']!r} "
        f"(host RAM safety net; set NEMOTRON_SKIP_CPU_MAX_MEMORY_CLAMP=1 to disable).",
        flush=True,
    )
    return out


def _normalize_max_memory(mm: dict) -> dict:
    """
    Accelerate only treats *integer* keys as GPU ids (see get_max_memory: gpu_devices =
    keys where isinstance(k, int)). JSON uses string keys, so '{"0":"6GiB"}' must become
    {0: bytes, "cpu": bytes} or the GPU cap is ignored and the full ~15GiB 4-bit model
    lands on a 15GiB card → OOM.
    """
    out: dict = {}
    for k, v in mm.items():
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = v
        else:
            out[k] = v
    return out


def _auto_cpu_max_memory_cap() -> str:
    """
    Accelerate packs offloaded weights into *host RAM* up to the "cpu" cap.
    High cpu caps accumulate quantized weights in RAM alongside the ~4.6 GiB
    per-shard decode buffer, causing OOM.  Keep cpu low so most weights go to
    GPU (fast) or disk (safe).
    """
    env = os.environ.get("NEMOTRON_MAX_MEMORY_CPU", "").strip()
    if env:
        return env if "GiB" in env or "GB" in env.upper() else f"{env}GiB"
    _kaggle = Path("/kaggle/input").is_dir()
    _vram = _cuda_total_bytes(0)
    _tight = _vram is not None and _vram < 22 * 1024**3
    if _kaggle and _tight:
        return "2GiB"
    try:
        import psutil

        total_gib = psutil.virtual_memory().total / (1024**3)
        cap = int(total_gib * 0.32)
        cap = max(6, min(cap, 14))
        return f"{cap}GiB"
    except Exception:
        return "10GiB"


def _auto_max_memory_for_quantized_load() -> dict[int | str, str] | None:
    """
    30B in 4-bit is ~15 GiB of weights alone.  Put as much as possible on GPU
    so quantized weights leave CPU RAM immediately during shard dispatch.
    Low CPU cap + disk spill keeps host RAM safe for the ~4.6 GiB per-shard
    decode buffer.
    """
    total = _cuda_total_bytes(0)
    if total is None or total >= 22 * 1024**3:
        return None
    gib = total / (1024**3)
    gpu_gib = max(4, min(12, int(gib * 0.65)))
    return {
        0: f"{gpu_gib}GiB",
        "cpu": _auto_cpu_max_memory_cap(),
        "disk": "150GiB",
    }


def _load_pretrained_single_or_fallback(
    model_path: str,
    base_kw: dict,
    *,
    kaggle_quantized: bool = False,
) -> "AutoModelForCausalLM":
    """Load model with attention implementation selection.

    On Kaggle T4 with quantization (``kaggle_quantized=True``), use a single
    attempt with no explicit attention implementation (model default / eager)
    so a partially-loaded model never overlaps in RSS with a second attempt.
    NemotronH does not support sdpa; eager is the safe choice.
    On larger GPUs, try flash-attn first then fall back.
    """
    import gc

    from transformers import AutoModelForCausalLM

    if kaggle_quantized:
        attn_choices: list[str | None] = [None]
    elif base_kw.get("max_memory") is not None:
        attn_choices = [None]
    else:
        attn_choices = ["flash_attention_2", "sdpa", None]

    print(f"[03_train_lora] attn load order: {attn_choices}", flush=True)

    model = None
    last_err: Exception | None = None
    for attn in attn_choices:
        try:
            kw = dict(base_kw)
            if attn:
                kw["attn_implementation"] = attn
            model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
            break
        except Exception as e:
            last_err = e
            del model
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if kaggle_quantized:
                raise
            continue
    if model is None:
        if last_err is not None:
            raise last_err
        raise RuntimeError("from_pretrained failed with no model and no error")
    return model


def load_model_peft(
    model_path: str,
    *,
    quantize: bool,
    lora_alpha: int,
    lora_target_mode: str,
    offload_folder: Path,
    max_memory: dict | None = None,
    no_auto_max_memory: bool = False,
    use_rslora: bool = True,
    lora_r: int = 32,
    lora_dropout: float = 0.05,
):
    import gc

    import transformers
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tv = getattr(transformers, "__version__", "0")
    try:
        t_major = int(tv.split(".", 1)[0])
    except ValueError:
        t_major = 0
    if t_major >= 5 and quantize:
        print(
            "[03_train_lora] WARNING: transformers>=5 + 4-bit often OOMs on ~16GB GPUs "
            "(HF issue: quantized weights materialize on GPU before quant). "
            "Pin: pip install 'transformers>=4.45,<5'. Current:",
            tv,
            flush=True,
        )
        _vram = _cuda_total_bytes(0)
        _allow_t5 = os.environ.get("NEMOTRON_ALLOW_TRANSFORMERS5_4BIT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if (
            _vram is not None
            and _vram < 22 * 1024**3
            and not _allow_t5
        ):
            raise RuntimeError(
                "Refusing 4-bit load with transformers>=5 on a <22GB GPU (typical CUDA OOM during shard load). "
                "Install transformers 4.x: pip install -U 'transformers>=4.45,<5' "
                "or set NEMOTRON_ALLOW_TRANSFORMERS5_4BIT=1 to try anyway. "
                f"Current transformers=={tv}"
            )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload_folder = Path(offload_folder).expanduser().resolve()
    offload_folder.mkdir(parents=True, exist_ok=True)
    offload_str = str(offload_folder)
    print(f"[03_train_lora] offload_folder={offload_str}", flush=True)

    effective_mm = _normalize_max_memory(max_memory) if max_memory else None
    if effective_mm is None and quantize and not no_auto_max_memory:
        effective_mm = _auto_max_memory_for_quantized_load()
        if effective_mm is not None:
            print(
                f"[03_train_lora] auto max_memory (small GPU + 4-bit): {effective_mm}",
                flush=True,
            )
    if effective_mm is not None:
        effective_mm = _clamp_max_memory_cpu_for_host_ram(effective_mm, quantize=quantize)
    if Path("/kaggle/input").is_dir() and effective_mm is not None:
        print(
            f"[03_train_lora] max_memory passed to from_pretrained: {effective_mm}",
            flush=True,
        )

    _sig = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    base_kw: dict = dict(
        device_map="auto",
        trust_remote_code=True,
        offload_folder=offload_str,
    )
    # QLoRA: avoid bf16 dtype on the full skeleton — let BitsAndBytesConfig own compute dtype.
    if not quantize:
        base_kw.update(_from_pretrained_dtype_kw(AutoModelForCausalLM))
    if "low_cpu_mem_usage" in _sig:
        base_kw["low_cpu_mem_usage"] = True
    if "max_workers" in _sig:
        base_kw["max_workers"] = 1
    if effective_mm is not None and "max_memory" in _sig:
        base_kw["max_memory"] = effective_mm
    if quantize:
        _skip = ["lm_head"]
        if "nemotron" in model_path.lower() or "nemotron" in MODEL_ID.lower():
            _skip += ["in_proj", "out_proj", "conv1d", "x_proj", "dt_proj"]
            print(
                "[03_train_lora] Nemotron detected: skipping quantization for Mamba "
                "projection modules (Triton kernels call F.linear on raw weights)",
                flush=True,
            )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=_skip,
        )
        base_kw["quantization_config"] = bnb

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _kaggle_q = Path("/kaggle/input").is_dir() and quantize and effective_mm is not None
    _prev_torch_threads: int | None = None
    _prev_torch_interop: int | None = None
    if _kaggle_q:
        try:
            _prev_torch_threads = torch.get_num_threads()
            torch.set_num_threads(1)
        except RuntimeError:
            _prev_torch_threads = None
        try:
            if hasattr(torch, "get_num_interop_threads"):
                _prev_torch_interop = torch.get_num_interop_threads()
                torch.set_num_interop_threads(1)
        except RuntimeError:
            _prev_torch_interop = None
    try:
        model = _load_pretrained_single_or_fallback(
            model_path, base_kw, kaggle_quantized=_kaggle_q,
        )
    finally:
        if _prev_torch_interop is not None and hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(_prev_torch_interop)
            except RuntimeError:
                pass
        if _prev_torch_threads is not None:
            try:
                torch.set_num_threads(_prev_torch_threads)
            except RuntimeError:
                pass

    if quantize:
        model = prepare_model_for_kbit_training(model)

    targets = _lora_target_modules(lora_target_mode)
    _lc_sig = inspect.signature(LoraConfig.__init__).parameters
    _lc_kw: dict = dict(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    if "use_rslora" in _lc_sig:
        _lc_kw["use_rslora"] = use_rslora
    lora_config = LoraConfig(**_lc_kw)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, "peft"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for Nemotron")
    parser.add_argument("--data-path", type=Path, default=Path("data/train_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("lora_adapter"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("lora_output"))
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_ID,
        help="HF model id or local path (e.g. kagglehub.model_download(...)).",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Load base in bf16 without 4-bit quantization (typical Kaggle kagglehub weights).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha; competition notebooks often use 16 with kaggle_nemotron targets.",
    )
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (max 32 per rules).")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout (0.05–0.1 typical)."
    )
    parser.add_argument(
        "--no-rslora",
        action="store_true",
        help="Disable RSLoRA (rsLoRA stabilizes larger ranks; enabled by default).",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable SFT packing (better throughput + more gradient signal per step).",
    )
    parser.add_argument(
        "--neftune-alpha",
        type=float,
        default=5.0,
        help="NEFTune embedding noise alpha (0 to disable; 5 is the standard +1–2pp SFT boost).",
    )
    parser.add_argument(
        "--completion-only",
        action="store_true",
        help="Mask prompt tokens, compute loss only on assistant reply (TRL DataCollatorForCompletionOnlyLM).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Gradient clipping max-norm."
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="Fraction of total steps for LR warmup."
    )
    parser.add_argument(
        "--lora-target-mode",
        choices=("hf_linear", "kaggle_nemotron"),
        default="hf_linear",
        help="kaggle_nemotron: regex in_proj/out_proj/up/down_proj (Kaggle starter pattern).",
    )
    parser.add_argument(
        "--adapter-base-name",
        type=str,
        default=MODEL_ID,
        help="Written to adapter_config.json base_model_name_or_path for submission tooling.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=None,
        help="Disk directory for accelerate when MoE weights spill off GPU (default: <checkpoint-dir>/hf_offload).",
    )
    parser.add_argument(
        "--max-memory-json",
        type=str,
        default=None,
        help='Optional accelerate max_memory JSON (numeric keys may be strings; script coerces to int). '
        'Example: \'{"0":"5GiB","cpu":"240GiB"}\'.',
    )
    parser.add_argument(
        "--no-auto-max-memory",
        action="store_true",
        help="Do not infer max_memory for <22GB GPUs when using 4-bit (can OOM loading 30B on T4).",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-peft", action="store_true", help="Skip Unsloth")
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 on Kaggle/CUDA to avoid fork deadlocks; was 2).",
    )
    parser.add_argument(
        "--no-nemotron-kaggle-patches",
        action="store_true",
        help="Disable Triton/ptxas copy + Nemotron RMSNorm/fast-path workarounds (see NEMOTRON_KAGGLE_PATCHES).",
    )
    args = parser.parse_args()
    if args.no_nemotron_kaggle_patches:
        os.environ["NEMOTRON_KAGGLE_PATCHES"] = "0"

    use_wandb = os.environ.get("USE_WANDB", "").lower() in ("1", "true", "yes")
    report_to = "wandb" if use_wandb else "none"

    if not args.data_path.is_file():
        raise SystemExit(f"Missing dataset {args.data_path}. Run 02_prepare_data.py first.")

    print(f"[03_train_lora] build={_TRAIN_LORA_BUILD}", flush=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    apply_nemotron_kaggle_env_patches()

    offload_folder = args.offload_folder or (args.checkpoint_dir / "hf_offload")
    env_off = os.environ.get("HF_OFFLOAD_FOLDER")
    if env_off:
        offload_folder = Path(env_off)

    max_memory = None
    if args.max_memory_json:
        max_memory = json.loads(args.max_memory_json)
        if not isinstance(max_memory, dict):
            raise SystemExit("--max-memory-json must be a JSON object")

    ds = load_dataset("json", data_files=str(args.data_path), split="train")
    ds = ds.train_test_split(test_size=args.test_size, seed=args.seed)

    model_path = resolve_model_weights_path(args.model_path)
    if model_path != args.model_path:
        print(
            f"[03_train_lora] --model-path resolved: {args.model_path!r} -> {model_path!r}",
            flush=True,
        )

    _ensure_pip_target_pth_processed()
    _prepare_cutlass_import_for_nemotron()

    model = None
    tokenizer = None
    backend = ""
    use_unsloth = (
        not args.force_peft
        and not args.no_quant
        and model_path == MODEL_ID
    )
    if use_unsloth:
        try:
            model, tokenizer, backend = load_model_unsloth(args.max_seq_length)
            print("Loaded model via Unsloth")
        except Exception as e:
            print(f"Unsloth load failed ({e!r}); falling back to PEFT.")
    if model is None:
        try:
            model, tokenizer, backend = load_model_peft(
                model_path,
                quantize=not args.no_quant,
                lora_alpha=args.lora_alpha,
                lora_target_mode=args.lora_target_mode,
                offload_folder=offload_folder,
                max_memory=max_memory,
                no_auto_max_memory=args.no_auto_max_memory,
                use_rslora=not args.no_rslora,
                lora_r=args.lora_r,
                lora_dropout=args.lora_dropout,
            )
            q = "PEFT + 4-bit" if not args.no_quant else "PEFT + bf16 (no quant)"
            print(f"Loaded model via {q}")
        except Exception as e:
            err = str(e).lower()
            if "mamba" in err:
                raise SystemExit(
                    "PEFT load failed: Nemotron-3 includes Mamba layers and needs `mamba-ssm` "
                    "(and usually `causal-conv1d`). On Kaggle, add to your pip cell:\n"
                    "  pip install mamba-ssm causal-conv1d\n"
                    f"Original error: {e!r}"
                ) from e
            _mn = getattr(e, "name", None)
            if (
                "cutlass" in err
                or _mn in ("cutlass", "cutlass.cute")
                or ("cannot import name" in err and "cutlass" in err)
            ):
                raise SystemExit(
                    "PEFT load failed: Nemotron needs NVIDIA **CuTe DSL** `cutlass` (`Int32`, `cutlass.cute`, …).\n"
                    "Install **`nvidia-cutlass-dsl`** (pulls **`nvidia-cutlass-dsl-libs-base`**). "
                    "Not the same as **`nvidia-cutlass`** (`cutlass_cppgen` / `pycute`).\n"
                    "Kaggle: Install + T4 **`--target .pip_tf4`** pip line must include **`nvidia-cutlass-dsl`**. "
                    "After `pip --target`, **`nvidia_cutlass_dsl.pth`** must live in `.pip_tf4`; "
                    "`03_train_lora.py` runs **`site.addsitedir`** on that dir so `.pth` is applied "
                    "(PYTHONPATH alone does not load `.pth`).\n"
                    "Do **not** `pip install cutlass` (unrelated PyPI project). "
                    "NEMOTRON_DISABLE_CUTLASS_SHIM=1 skips cutlass alias cleanup only.\n"
                    f"Original error: {e!r}"
                ) from e
            if "offload_folder" in err or "offloaded to the disk" in err:
                raise SystemExit(
                    "PEFT load failed: MoE + device_map often needs a disk offload dir. "
                    "This script defaults to <checkpoint-dir>/hf_offload; set explicitly with:\n"
                    "  --offload-folder /kaggle/working/project/lora_output/hf_offload\n"
                    "Or drop --no-quant to use 4-bit QLoRA and reduce GPU/disk offload.\n"
                    f"Original error: {e!r}"
                ) from e
            if "not a valid model identifier" in err or "is not a local folder" in err:
                raise SystemExit(
                    "PEFT load failed: Hugging Face could not load this --model-path.\n"
                    "On Kaggle, add the competition **Model** input (often `/kaggle/input/nemotron-3-nano-30b-a3b-bf16/` "
                    "or under `/kaggle/input/models/...`) or set env NEMOTRON_MODEL_PATH.\n"
                    "This script auto-searches those paths for config.json.\n"
                    f"Original error: {e!r}"
                ) from e
            if "out of memory" in err:
                raise SystemExit(
                    "PEFT load failed: CUDA OOM while loading weights.\n"
                    "1) Pin transformers 4.x (v5 can materialize FP weights on GPU before 4-bit quant — "
                    "max_memory may not help):  pip install -U 'transformers>=4.45,<5'\n"
                    "2) Integer GPU keys in max_memory (script auto-sets this). Tighten GPU/cpu, e.g.\n"
                    '     --max-memory-json \'{"0":"10GiB","cpu":"2GiB","disk":"150GiB"}\'\n'
                    "   Host RAM OOM (Linux SIGKILL during shards): raise GPU cap + lower cpu.\n"
                    "3) Avoid --no-quant on T4.\n"
                    f"Original error: {e!r}"
                ) from e
            if "refusing 4-bit load" in err:
                raise SystemExit(
                    f"{e}\n"
                    "Re-run the notebook **Install** cell (T4_PYPI_TRANSFORMERS_4X) or before training run:\n"
                    "  python -m pip install -U --prefer-binary 'transformers>=4.45,<5'\n"
                    "Or set NEMOTRON_ALLOW_TRANSFORMERS5_4BIT=1 (v5 on T4 usually CUDA OOMs)."
                ) from e
            raise SystemExit(
                f"PEFT load failed ({e!r}).\n"
                "If you still see 'provide offload_folder' or 'Try --force-peft', your Kaggle copy of "
                "scripts/03_train_lora.py is OUTDATED — re-upload the dataset and re-run Bootstrap.\n"
                "Otherwise try: explicit --offload-folder, --max-memory-json, or remove --no-quant."
            ) from e

    patch_nemotron_modules_after_load()

    def formatting_func(example):
        # Phase 0.4 chat-template fix. The Nemotron template emits an empty
        # `<think></think>` placeholder when there's an assistant message and
        # `add_generation_prompt=False`. At eval time the harness uses
        # `add_generation_prompt=True`, which instead emits an open `<think>\n`
        # block. That mismatch was caught by 00_check_prompt_parity.
        #
        # Fix: build the prompt (system + user) with add_generation_prompt=True,
        # then concatenate the assistant content + EOS. The assistant content
        # already has shape `{reasoning}\n</think>\n\\boxed{ANSWER}`, so the
        # final tokenization matches the eval prompt exactly through the open
        # `<think>` boundary.
        messages = example["messages"]
        prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
        assistant_msg = next(
            (m for m in messages if m.get("role") == "assistant"), None
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        if assistant_msg is None:
            return prompt_text
        eos = tokenizer.eos_token or "<|im_end|>"
        return f"{prompt_text}{assistant_msg['content']}{eos}\n"

    common_kwargs = dict(
        output_dir=str(args.checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=torch.cuda.is_available(),
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=report_to,
        dataloader_num_workers=args.dataloader_workers,
        max_grad_norm=args.max_grad_norm,
    )
    if args.neftune_alpha and args.neftune_alpha > 0:
        common_kwargs["neftune_noise_alpha"] = float(args.neftune_alpha)
    if hasattr(TrainingArguments, "eval_strategy"):
        common_kwargs["eval_strategy"] = "epoch"
        common_kwargs["load_best_model_at_end"] = True
        common_kwargs["metric_for_best_model"] = "eval_loss"
    else:
        common_kwargs["evaluation_strategy"] = "epoch"
        common_kwargs["load_best_model_at_end"] = True
        common_kwargs["metric_for_best_model"] = "eval_loss"

    if torch.cuda.is_available() and "gradient_checkpointing_kwargs" in inspect.signature(
        SFTConfig.__init__
    ).parameters:
        common_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": True}

    cfg_kw = {**common_kwargs, **_sft_config_kwargs(args.max_seq_length, packing=args.packing)}
    cfg_kw = _kwargs_for_callable(SFTConfig.__init__, cfg_kw)
    try:
        training_args = SFTConfig(**cfg_kw)
    except TypeError:
        cfg_kw.pop("packing", None)
        training_args = SFTConfig(**cfg_kw)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        formatting_func=formatting_func,
    )

    if args.completion_only and not args.packing:
        # Compute loss only on assistant tokens. Huge accuracy gain on small SFT sets.
        # Response template must match what the chat template emits right before the
        # assistant turn. Nemotron chat template uses "<|assistant|>\n"; we fall back
        # to a generic "assistant\n" marker if the specific token is not found in the
        # tokenized prompt.
        from trl import DataCollatorForCompletionOnlyLM

        _candidates = [
            "<|assistant|>\n",
            "<|im_start|>assistant\n",
            "\nassistant\n",
            "\n<|assistant|>\n",
            "<extra_id_1>Assistant\n",
            "<extra_id_1>assistant\n",
        ]
        probe = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        chosen = next((c for c in _candidates if c in probe), None)
        if chosen is None:
            raise SystemExit(
                "[03_train_lora] --completion-only: no known assistant-turn "
                "marker found in the chat-template probe.\n"
                f"  candidates tried: {_candidates}\n"
                f"  probe text (repr): {probe!r}\n"
                "Action: inspect the tokenizer's chat_template (or "
                "tokenizer_config.json) to find the literal string emitted "
                "right before the assistant turn, then append it to the "
                "_candidates list in kaggle/scripts/03_train_lora.py.\n"
                "Without a matching template, DataCollatorForCompletionOnlyLM "
                "would mask EVERY label (-100) and the run would learn nothing."
            )

        probe_ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
        template_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]

        def _is_subseq(needle: list, hay: list) -> bool:
            n = len(needle)
            if n == 0 or n > len(hay):
                return False
            for i in range(len(hay) - n + 1):
                if hay[i:i + n] == needle:
                    return True
            return False

        if not _is_subseq(template_ids, probe_ids):
            raise SystemExit(
                "[03_train_lora] --completion-only: response_template "
                f"{chosen!r} appears as a SUBSTRING in the chat-template "
                f"probe but its tokenized ids {template_ids} are NOT a "
                f"subsequence of the tokenized probe (first 60 ids: "
                f"{probe_ids[:60]}...).\n"
                "DataCollatorForCompletionOnlyLM matches on token-id "
                "subsequence, so it would silently mask every label and the "
                "run would learn nothing.\n"
                "Fix: pick a response_template whose tokenization survives "
                "BPE merging in this tokenizer (e.g. prefix it with the same "
                "leading whitespace the chat template emits, like '\\n')."
            )

        print(
            f"[03_train_lora] completion-only loss: response_template={chosen!r} "
            f"(token ids {template_ids}, verified as subsequence in tokenized probe)",
            flush=True,
        )
        collator = DataCollatorForCompletionOnlyLM(response_template=chosen, tokenizer=tokenizer)
        trainer_kwargs["data_collator"] = collator

    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(
        "Starting trainer.train() — the progress bar may stay at 0% for a long time on the "
        "**first** step (MoE + disk offload + first backward). If it never moves after ~30–60 min, "
        "set --dataloader-workers 0 and ensure GPU RAM is not thrashing.",
        flush=True,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    _ensure_base_model_in_adapter(args.output_dir, args.adapter_base_name)
    print(f"Saved LoRA adapter to {args.output_dir} (backend={backend})")


if __name__ == "__main__":
    main()
