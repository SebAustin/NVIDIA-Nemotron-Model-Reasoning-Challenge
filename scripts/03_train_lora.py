#!/usr/bin/env python3
"""Phase 3: Kaggle-fit 8-bit QLoRA SFT for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Designed for 2x T4 (32 GB total): load the base model in 8-bit, shard across both
GPUs with ``device_map="auto"`` + ``max_memory`` caps and CPU/disk offload, train
a small LoRA adapter with gradient checkpointing. 4-bit QLoRA is NOT reliable on
this hybrid Mamba model, hence 8-bit.

Key behaviours required by the brief:
  * ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` set BEFORE importing torch.
  * LoRA target_modules default to the OFFICIAL demo regex
    ``.*\\.(in_proj|out_proj|up_proj|down_proj)$`` (Mamba + MLP; never the MoE
    router/gate). Verified against ``model.named_modules()`` before training.
  * A tiny smoke test (1% data, few steps) runs FIRST to prove the memory config
    fits before the full run.
  * Per-category training loss AND min-logprob are logged each epoch; categories
    whose min-logprob has not approached 0 are printed (upweight-next candidates).
  * On CUDA OOM (after offload tuning) we stop and write FALLBACK.md telling you to
    run the identical config on one rented A100/H100 and copy the adapter back.

Env knobs: LORA_R (default 16, max 32), LORA_ALPHA (default 2*R),
SFT_MAX_SEQ_LENGTH (default 1536), LORA_TARGET_REGEX, NUM_EPOCHS, LEARNING_RATE,
PER_DEVICE_BATCH, GRAD_ACCUM, NEMOTRON_MAX_MEMORY_GPU (e.g. 13GiB),
NEMOTRON_MAX_MEMORY_CPU (e.g. 24GiB), SEED.
"""

from __future__ import annotations

# --- MUST precede `import torch` -------------------------------------------
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# out_proj is consumed RAW by the fused Mamba kernel (F.linear on its weight), which
# breaks if it's quantized AND bypasses any LoRA on it — so we keep out_proj bf16
# (skip-quantized) and LoRA in_proj/up_proj/down_proj (all applied as real modules).
DEFAULT_TARGET_REGEX = r".*\.(in_proj|up_proj|down_proj)$"
SKIP_QUANT_MODULES = ["out_proj"]
FORBIDDEN_SUBSTRINGS = ("router", "gate", "expert_gate", "lm_head")  # never LoRA the router


# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def human_max_memory() -> Dict:
    """Build a device->capacity map for device_map='auto' + offload."""
    import torch

    gpu_cap = os.environ.get("NEMOTRON_MAX_MEMORY_GPU", "13GiB")
    cpu_cap = os.environ.get("NEMOTRON_MAX_MEMORY_CPU")
    if cpu_cap is None:
        try:
            import psutil

            avail = int(psutil.virtual_memory().available / (1024**3))
            cpu_cap = f"{max(8, min(avail - 4, 28))}GiB"
        except Exception:
            cpu_cap = "16GiB"
    mm: Dict = {}
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(n):
        mm[i] = gpu_cap
    mm["cpu"] = cpu_cap
    return mm


def resolve_model_source() -> str:
    """Model weights location: --model-path/MODEL_PATH env, else a Kaggle model
    mount if present (avoids a ~60GB HF re-download), else the HF id."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    import glob

    for pat in (
        "/kaggle/input/**/config.json",
        "/kaggle/input/**/nemotron*/**/config.json",
    ):
        hits = glob.glob(pat, recursive=True)
        hits = [h for h in hits if "nemotron" in h.lower()]
        if hits:
            return str(Path(hits[0]).parent)
    return MODEL_ID


def load_base_model(offload_dir: Path, model_src: str, quant: str = "8bit"):
    """quant: 'none' (bf16, big GPU), '8bit' (2xT4 + offload), '4bit' (NF4 QLoRA,
    fits a single 40GB A100; compute in bf16 so dtypes stay consistent)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[lora] loading base model from: {model_src} (quant={quant})")
    tok = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    offload_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        device_map="auto",
        max_memory=human_max_memory(),
        offload_folder=str(offload_dir),
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=SKIP_QUANT_MODULES,  # out_proj must stay bf16 for the fused kernel
        )
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # keep MoE math in bf16
            llm_int8_enable_fp32_cpu_offload=True,  # tolerate CPU spill on small GPUs
            llm_int8_skip_modules=SKIP_QUANT_MODULES,  # out_proj must stay bf16 for the fused kernel
        )
    # else 'none': bf16 weights straight onto a big GPU (A100/H100 80GB)
    model = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)
    model.config.use_cache = False
    return model, tok


def maybe_force_torch_forward() -> None:
    """On pre-Ampere GPUs (T4/sm_75) the Mamba-2 SSD Triton kernels fail to
    compile ('Unsupported conversion from bf16 to f16'). Force the model's
    pure-PyTorch ``torch_forward`` path by flipping the dynamic module's
    ``is_fast_path_available`` global to False. Set FORCE_TORCH_FORWARD=0 to skip."""
    import sys

    if os.environ.get("FORCE_TORCH_FORWARD", "auto") == "0":
        return
    caps = []
    try:
        import torch

        caps = [torch.cuda.get_device_capability(i)
                for i in range(torch.cuda.device_count())]
        pre_ampere = any(c < (8, 0) for c in caps)
    except Exception:
        pre_ampere = False
    force = os.environ.get("FORCE_TORCH_FORWARD") == "1" or pre_ampere
    if not force:
        return
    patched = 0
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_nemotron_h") and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False
            patched += 1
    print(f"[lora] forced torch_forward (disabled Mamba fast path) on {patched} "
          f"module(s); GPU caps={caps}")


def patch_moe_dtype() -> None:
    """NemotronHMOE.moe accumulates into a bf16 tensor via index_add_ but the
    per-expert ``weighted_output`` can be fp32 (autocast) -> dtype error. Replace
    ``moe`` with a version that casts to the accumulator dtype before index_add_."""
    import sys

    import torch

    def _safe_moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        ).permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_output = expert(hidden_states[token_indices])
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output.to(final_hidden_states.dtype)
                )
            else:
                # feed the dummy in the ACTIVATION dtype (hidden_states), not the
                # weight dtype — a 4-bit layer's .weight is packed uint8.
                dummy_out = expert(torch.zeros_like(hidden_states[0]).unsqueeze(0))
                final_hidden_states = final_hidden_states + dummy_out.to(
                    final_hidden_states.dtype
                )
        return final_hidden_states.type(hidden_states.dtype)

    patched = 0
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_nemotron_h"):
            cls = getattr(mod, "NemotronHMOE", None)
            if cls is not None:
                cls.moe = _safe_moe
                patched += 1
    if patched:
        print(f"[lora] patched NemotronHMOE.moe for dtype-safe index_add_ ({patched})")


def verify_target_modules(model, regex: str) -> List[str]:
    """Return the module names matched by `regex`; abort if it hits the router."""
    pat = re.compile(regex)
    matched = [name for name, _ in model.named_modules() if pat.fullmatch(name)]
    if not matched:
        raise SystemExit(
            f"target regex {regex!r} matched NO modules. Inspect model.named_modules()."
        )
    bad = [m for m in matched if any(f in m for f in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise SystemExit(
            f"target regex matched forbidden modules (router/gate/lm_head): {bad[:5]}"
        )
    leaves = sorted({m.split(".")[-1] for m in matched})
    print(f"[lora] target regex matches {len(matched)} modules; leaf types: {leaves}")
    return matched


def build_peft_model(model, r: int, alpha: int, target_regex: str):
    from peft import LoraConfig, get_peft_model

    # NOTE: deliberately NOT calling prepare_model_for_kbit_training — it upcasts
    # layernorms to fp32, which clashes with the bf16 LoRA / fused Mamba kernel. We
    # keep everything bf16 (autocast_adapter_dtype=False) and set up grad
    # checkpointing + input-require-grads AFTER get_peft_model (the canonical order;
    # with use_reentrant=False the block INPUT must require grad, not just the LoRA
    # params inside it).
    model.config.use_cache = False
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_regex,  # regex string (PEFT supports this)
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg, autocast_adapter_dtype=False)
    # Gradient checkpointing OFF by default: this model's custom checkpoint path +
    # the fused Mamba kernel breaks grad flow ("element 0 ... does not require grad").
    # Without it the LoRA params build a normal graph. Set GRAD_CHECKPOINT=1 to re-enable.
    if os.environ.get("GRAD_CHECKPOINT", "0") == "1":
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
def load_sft_records(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_text_dataset(records: List[dict], tokenizer):
    """Render each {messages} to a single training string via the chat template."""
    from datasets import Dataset

    texts, cats = [], []
    for r in records:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        cats.append((r.get("meta") or {}).get("category", "other"))
    return Dataset.from_dict({"text": texts, "category": cats})


def make_sft_config(output_dir, max_seq_len, epochs, lr, bsz, accum, **extra):
    """Construct SFTConfig, adapting to TRL versions (max_length vs max_seq_length)."""
    import inspect

    from trl import SFTConfig

    params = set(inspect.signature(SFTConfig.__init__).parameters)
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        # Keep this in sync with build_peft_model: OFF by default for this model
        # (checkpointing + fused Mamba kernel breaks LoRA grad flow). GRAD_CHECKPOINT=1 re-enables.
        gradient_checkpointing=os.environ.get("GRAD_CHECKPOINT", "0") == "1",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        **extra,
    )
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_len
    elif "max_length" in params:
        kwargs["max_length"] = max_seq_len
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return SFTConfig(**kwargs)


def completion_collator(tokenizer):
    """Mask everything up to and including the assistant turn header so loss is on
    the reasoning + answer only."""
    try:
        from trl import DataCollatorForCompletionOnlyLM

        return DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n",
            tokenizer=tokenizer,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[lora] completion-only collator unavailable ({e!r}); full-text loss.")
        return None


# ---------------------------------------------------------------------------
class PerCategoryLogprobProbe:
    """Computes per-category MIN target-token logprob over a small fixed probe set
    and flags categories whose min-logprob has not approached 0."""

    def __init__(self, tokenizer, records, max_per_cat=8, threshold=-3.0):
        self.tok = tokenizer
        self.threshold = threshold
        by_cat: Dict[str, list] = {}
        for r in records:
            by_cat.setdefault((r.get("meta") or {}).get("category", "other"), []).append(r)
        self.probes = {c: rows[:max_per_cat] for c, rows in by_cat.items()}

    def __call__(self, model) -> Dict[str, float]:
        import torch

        results: Dict[str, float] = {}
        model.eval()
        with torch.no_grad():
            for cat, rows in self.probes.items():
                mins = []
                for r in rows:
                    full = self.tok.apply_chat_template(
                        r["messages"], tokenize=False, add_generation_prompt=False
                    )
                    prompt_only = self.tok.apply_chat_template(
                        r["messages"][:-1], tokenize=False, add_generation_prompt=True
                    )
                    ids = self.tok(full, return_tensors="pt").input_ids
                    p_len = self.tok(prompt_only, return_tensors="pt").input_ids.shape[1]
                    ids = ids.to(model.device)
                    logits = model(ids).logits[:, :-1, :]
                    targets = ids[:, 1:]
                    logp = torch.log_softmax(logits.float(), dim=-1)
                    tok_lp = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)[0]
                    comp_lp = tok_lp[p_len - 1:]
                    if comp_lp.numel():
                        mins.append(float(comp_lp.min()))
                if mins:
                    results[cat] = min(mins)
        model.train()
        return results


# ---------------------------------------------------------------------------
def write_fallback_md(reason: str, cfg: dict) -> None:
    txt = f"""# FALLBACK: train on a single rented A100/H100

The Kaggle 2x T4 8-bit + CPU-offload run hit an unrecoverable memory error:

    {reason}

This is the documented #1 failure mode (31.6B hybrid Mamba does not 8-bit-fit on
32 GB with room for activations). Do NOT truncate the model. Instead run the
*identical* config on one big GPU and copy the adapter back.

## Steps (Modal / RunPod / Lambda)
1. Provision 1x A100 80GB (or H100). CUDA 12.x, Python 3.10-3.12.
2. Install: `pip install -r requirements-mamba.txt` (mamba-ssm + causal-conv1d
   need a working nvcc) plus `transformers>=4.45,<5 peft trl datasets accelerate
   bitsandbytes`.
3. Copy `data/train_sft.jsonl` to the box.
4. Run the SAME script with a large GPU cap (no offload needed):

       NEMOTRON_MAX_MEMORY_GPU=78GiB LORA_R={cfg.get('r')} \\
       LORA_ALPHA={cfg.get('alpha')} SFT_MAX_SEQ_LENGTH={cfg.get('max_seq')} \\
       python scripts/03_train_lora.py --data-path data/train_sft.jsonl \\
         --output-dir lora_adapter

   On 80GB you can also load bf16 (drop 8-bit) for speed; the adapter is identical.
5. `scp` the resulting `lora_adapter/` back here and run Phase 4/5 locally.

Config at failure: {json.dumps(cfg, indent=2)}
"""
    Path("FALLBACK.md").write_text(txt, encoding="utf-8")
    print("Wrote FALLBACK.md")


def diagnose_grad(model, tokenizer, record) -> None:
    """One-shot grad-flow probe: localize why loss.requires_grad is False.
    Set DIAGNOSE_GRAD=1 to run this instead of training, then exits."""
    import torch

    model.train()
    text = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    enc["labels"] = enc["input_ids"].clone()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[diag] trainable tensors: {len(trainable)}")
    if trainable:
        n0, p0 = trainable[0]
        print(f"[diag] sample trainable: {n0} dtype={p0.dtype} device={p0.device} "
              f"is_leaf={p0.is_leaf}")

    # --- inspect one wrapped in_proj LoRA module directly ---
    for n, m in model.named_modules():
        if n.endswith("in_proj") and hasattr(m, "lora_A"):
            keys = list(m.lora_A.keys())
            print(f"[diag] {n}: active_adapters={getattr(m, 'active_adapters', '?')} "
                  f"merged={getattr(m, 'merged', '?')} "
                  f"disable={getattr(m, '_disable_adapters', '?')} lora_keys={keys}")
            la = m.lora_A[keys[0]]
            print(f"[diag]   lora_A.weight: dtype={la.weight.dtype} "
                  f"requires_grad={la.weight.requires_grad}")
            try:
                x = torch.randn(1, 4, m.in_features,
                                dtype=la.weight.dtype, device=la.weight.device,
                                requires_grad=True)
                y = m(x)
                print(f"[diag]   direct module forward: out.requires_grad="
                      f"{y.requires_grad} grad_fn={type(y.grad_fn).__name__ if y.grad_fn else None}")
            except Exception as e:
                print(f"[diag]   direct module forward FAILED: {type(e).__name__}: {e}")
            break

    # --- per-module grad tracing: find the first True->False transition ---
    def _rg(o):
        if isinstance(o, torch.Tensor):
            return o.requires_grad
        if isinstance(o, (tuple, list)):
            for e in o:
                if isinstance(e, torch.Tensor):
                    return e.requires_grad
        return None

    want = ("embeddings", "embed_tokens", "layers.0.mixer.in_proj",
            "layers.0.mixer", "layers.0", "layers.1", "layers.2",
            "layers.3.mixer", "norm_f", "final_layernorm", "lm_head")
    trace = []
    hooks = []
    for n, m in model.named_modules():
        if n.endswith(want):
            hooks.append(m.register_forward_hook(
                lambda mod, i, o, nm=n: trace.append((nm, _rg(o)))))

    # --- full forward ---
    out = model(**enc)
    for h in hooks:
        h.remove()
    print("[diag] per-module output.requires_grad (forward order):")
    for nm, rg in trace:
        print(f"[diag]    {rg}  {nm}")
    loss = out.loss
    logits = getattr(out, "logits", None)
    print(f"[diag] logits.requires_grad="
          f"{None if logits is None else logits.requires_grad}")
    print(f"[diag] loss={float(loss):.4f} requires_grad={loss.requires_grad} "
          f"grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}")
    if loss.requires_grad:
        loss.backward()
        got = sum(1 for _, p in trainable if p.grad is not None)
        print(f"[diag] params WITH grad after backward: {got}/{len(trainable)}")
    raise SystemExit("[diag] done — remove DIAGNOSE_GRAD to train")


def is_oom(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in s or "cuda oom" in s or "cublas" in s or "alloc" in s


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="8-bit QLoRA SFT (Kaggle-fit)")
    ap.add_argument("--data-path", type=Path, default=Path("data/train_sft.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("lora_adapter"))
    ap.add_argument("--offload-dir", type=Path, default=Path("offload"))
    ap.add_argument("--model-path", type=str, default=None,
                    help="Base model dir/HF id. Default: MODEL_PATH env, else a "
                         "Kaggle model mount if found, else the HF id.")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None,
                    help="Quantization: none=bf16 (80GB GPU), 8bit (2xT4+offload), "
                         "4bit=NF4 QLoRA (fits one 40GB A100). Default: QUANT env, "
                         "else 8bit; --no-8bit forces none.")
    ap.add_argument("--no-8bit", action="store_true",
                    help="Alias for --quant none (load bf16 on a big GPU).")
    ap.add_argument("--no-smoke", action="store_true", help="Skip the smoke test.")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    model_src = args.model_path or resolve_model_source()
    quant = args.quant or os.environ.get("QUANT") or ("none" if args.no_8bit else "8bit")

    r = min(int(os.environ.get("LORA_R", "16")), 32)
    alpha = int(os.environ.get("LORA_ALPHA", str(2 * r)))
    max_seq = int(os.environ.get("SFT_MAX_SEQ_LENGTH", "1536"))
    target_regex = os.environ.get("LORA_TARGET_REGEX", DEFAULT_TARGET_REGEX)
    epochs = float(os.environ.get("NUM_EPOCHS", "2"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-4"))
    bsz = int(os.environ.get("PER_DEVICE_BATCH", "1"))
    accum = int(os.environ.get("GRAD_ACCUM", "16"))
    seed = int(os.environ.get("SEED", "42"))
    cfg = {"r": r, "alpha": alpha, "max_seq": max_seq, "target_regex": target_regex,
           "epochs": epochs, "lr": lr, "bsz": bsz, "accum": accum, "seed": seed,
           "quant": quant}
    print("[lora] config:", json.dumps(cfg, indent=2))
    set_all_seeds(seed)

    if not args.data_path.is_file():
        raise SystemExit(f"Missing {args.data_path}. Run 02_prepare_data.py first.")
    records = load_sft_records(args.data_path)
    print(f"[lora] loaded {len(records)} SFT records")

    try:
        from trl import SFTTrainer

        model, tok = load_base_model(args.offload_dir, model_src, quant)
        maybe_force_torch_forward()
        patch_moe_dtype()
        verify_target_modules(model, target_regex)
        model = build_peft_model(model, r, alpha, target_regex)
        if os.environ.get("DIAGNOSE_GRAD") == "1":
            diagnose_grad(model, tok, records[0])
        full_ds = to_text_dataset(records, tok)
        collator = completion_collator(tok)
        probe = PerCategoryLogprobProbe(tok, records)

        def run(ds, epochs_, max_steps, tag):
            sft_cfg = make_sft_config(
                args.output_dir, max_seq, epochs_, lr, bsz, accum,
                **({"max_steps": max_steps} if max_steps else {}),
            )
            kw = {}
            if collator is not None:
                kw["data_collator"] = collator
            trainer = SFTTrainer(
                model=model, args=sft_cfg, train_dataset=ds,
                processing_class=tok, **kw,
            )
            print(f"[lora] === {tag} ===")
            trainer.train()
            return trainer

        # 1) smoke test: 1% of data, 10 steps, to prove the memory config fits
        if not args.no_smoke:
            k = max(8, len(full_ds) // 100)
            smoke_ds = full_ds.shuffle(seed=seed).select(range(min(k, len(full_ds))))
            run(smoke_ds, 1, 10, f"SMOKE TEST ({len(smoke_ds)} rows, 10 steps)")
            print("[lora] smoke test passed: memory config fits.")
            mins = probe(model)
            print("[lora] post-smoke min-logprob by category:",
                  {c: round(v, 3) for c, v in mins.items()})
            if args.smoke_only:
                return

        # 2) full run
        run(full_ds, epochs, 0, f"FULL TRAIN ({len(full_ds)} rows, {epochs} epochs)")

        mins = probe(model)
        print("\n[lora] FINAL min-logprob by category:")
        not_converged = []
        for c in sorted(mins):
            flag = (" <-- not approaching 0 (upweight next round)"
                    if mins[c] < probe.threshold else "")
            if flag:
                not_converged.append(c)
            print(f"    {c:18s} {mins[c]:8.3f}{flag}")
        if not_converged:
            print(f"[lora] upweight-next candidates: {not_converged}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.output_dir))
        tok.save_pretrained(str(args.output_dir))
        print(f"[lora] saved adapter -> {args.output_dir}")

    except (RuntimeError, MemoryError) as e:
        traceback.print_exc()
        if is_oom(e):
            write_fallback_md(str(e), cfg)
            raise SystemExit(
                "OOM after offload tuning. Wrote FALLBACK.md — run the identical "
                "config on a rented A100/H100 and copy lora_adapter/ back."
            )
        raise


if __name__ == "__main__":
    main()
