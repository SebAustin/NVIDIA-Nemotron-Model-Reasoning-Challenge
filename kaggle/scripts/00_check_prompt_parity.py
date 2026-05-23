#!/usr/bin/env python3
"""Phase 0.4: assert training and inference produce the same conditioning prefix.

This is the silent-failure mode where the adapter learns to generate the
assistant turn conditioned on context X, but the eval / Kaggle harness queries
the model with context Y. Any divergence in the prefix means the model is
being asked to extrapolate, not interpolate.

What we check:

1. The eval prompt (chat template applied with `add_generation_prompt=True`)
   should be a strict PREFIX of the training prompt (chat template applied
   with `add_generation_prompt=False` over messages that include the assistant
   turn).
2. The `--completion-only` response template (the marker that
   `DataCollatorForCompletionOnlyLM` uses to find the start of the assistant
   turn) should appear in the EXACT same token positions in both.

If either check fails, the adapter is being trained on a different conditional
than what gets queried at inference. Typical silent cost: ~0.02-0.10 LB.

Usage:
    python kaggle/scripts/00_check_prompt_parity.py \
        --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

    # Or if base model is already attached as a Kaggle Dataset:
    python kaggle/scripts/00_check_prompt_parity.py \
        --model-path /kaggle/input/.../nemotron-base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer

from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT


_PROBE_USER = (
    "Examples:\n01010001 -> 11011101\n00001001 -> 01101101\n"
    "Now produce the output for: 00110100"
)
_PROBE_ASSISTANT = "Reasoning here.\n</think>\n\\boxed{10010111}"

_ASSISTANT_MARKER_CANDIDATES = [
    "<|assistant|>\n",
    "<|im_start|>assistant\n",
    "\nassistant\n",
    "\n<|assistant|>\n",
    "<extra_id_1>Assistant\n",
    "<extra_id_1>assistant\n",
]


def _find_response_marker(probe: str) -> str:
    for cand in _ASSISTANT_MARKER_CANDIDATES:
        if cand in probe:
            return cand
    raise SystemExit(
        "No known assistant-turn marker found in chat-template probe.\n"
        f"  candidates tried: {_ASSISTANT_MARKER_CANDIDATES}\n"
        f"  probe text (repr): {probe!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.4 prompt parity check")
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="HF id or local path with tokenizer files",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=_PROBE_USER,
        help="User-turn probe text (default: a bit-manipulation example)",
    )
    parser.add_argument(
        "--assistant-content",
        type=str,
        default=_PROBE_ASSISTANT,
        help="Assistant-turn probe text used by the training-side template render",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Both training and eval use the same prompt (system + user) rendered with
    # add_generation_prompt=True. Training then appends the assistant content +
    # EOS. This mirrors 03_train_lora.formatting_func and 04_evaluate.build_prompt
    # exactly, post the Phase 0.4 fix.
    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": args.user_prompt},
    ]
    eval_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = tokenizer.eos_token or "<|im_end|>"
    train_text = f"{eval_text}{args.assistant_content}{eos}\n"

    train_ids = tokenizer(train_text, add_special_tokens=False)["input_ids"]
    eval_ids = tokenizer(eval_text, add_special_tokens=False)["input_ids"]

    print("=" * 70)
    print("Phase 0.4 prompt parity diagnostic")
    print("=" * 70)
    print(f"Tokenizer:       {args.model_path}")
    print(f"Training text:   {len(train_text)} chars, {len(train_ids)} tokens")
    print(f"Eval text:       {len(eval_text)} chars, {len(eval_ids)} tokens")

    marker = _find_response_marker(train_text)
    print(f"Response marker: {marker!r}")

    if not train_text.startswith(eval_text):
        # Tolerate trailing whitespace differences right at the boundary; the
        # important property is that the eval text is a prefix of the training
        # text up to (and ideally including) the assistant marker.
        common_chars = 0
        for a, b in zip(train_text, eval_text):
            if a != b:
                break
            common_chars += 1
        print(
            "\nFAIL: eval text is NOT a strict prefix of the training text.\n"
            f"  First divergence at char {common_chars} of {len(eval_text)}.\n"
            f"  Eval[..{common_chars}]:     {eval_text[:common_chars]!r}\n"
            f"  Eval next:                {eval_text[common_chars:common_chars+80]!r}\n"
            f"  Training next:            {train_text[common_chars:common_chars+80]!r}\n"
            "Action: align the system prompt and chat template render between\n"
            "  kaggle/scripts/utils/data_formatter.py build_messages() and\n"
            "  kaggle/scripts/04_evaluate.py build_prompt()."
        )
        raise SystemExit(1)

    # Token-level: eval_ids should be a prefix of train_ids.
    if eval_ids != train_ids[: len(eval_ids)]:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(eval_ids, train_ids)) if a != b),
            min(len(eval_ids), len(train_ids)),
        )
        print(
            "\nFAIL: eval token ids are not a prefix of training token ids.\n"
            f"  First divergence at token index {first_diff}.\n"
            f"  Eval[..{first_diff}]:    {eval_ids[max(0,first_diff-5):first_diff+5]}\n"
            f"  Train[..{first_diff}]:   {train_ids[max(0,first_diff-5):first_diff+5]}\n"
            "Action: this usually means the chat template emits different\n"
            "whitespace under add_generation_prompt={True,False}. Inspect the\n"
            "template's `{% if add_generation_prompt %}` branch."
        )
        raise SystemExit(2)

    # Locate the assistant marker in BOTH renders and confirm same token slice.
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]

    def _find_sub(haystack: list[int], needle: list[int]) -> int:
        if not needle:
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return -1

    train_marker_pos = _find_sub(train_ids, marker_ids)
    eval_marker_pos = _find_sub(eval_ids, marker_ids)

    if train_marker_pos < 0:
        print(
            "\nFAIL: response marker tokens do not appear as a subsequence of\n"
            "the training token ids. DataCollatorForCompletionOnlyLM would\n"
            "mask every label. Pick a different marker whose tokenization\n"
            "survives BPE merging."
        )
        raise SystemExit(3)

    if eval_marker_pos < 0:
        print(
            "\nWARN: response marker not present in eval ids. This is expected\n"
            "if `add_generation_prompt=True` emits a different shorthand than\n"
            "the explicit assistant turn. Verify the eval generation continues\n"
            "from the same place training did."
        )

    elif eval_marker_pos != train_marker_pos:
        print(
            f"\nFAIL: response marker at train position {train_marker_pos} but\n"
            f"eval position {eval_marker_pos}. Token alignment is broken.\n"
            "Action: same fix as above (chat template whitespace mismatch)."
        )
        raise SystemExit(4)

    print("\nPASS: eval prefix matches training prefix exactly (char + token).")
    print(f"  Marker '{marker.strip()}' at token index {train_marker_pos}.")
    print(
        f"  Tokens BEFORE marker (=context the model conditions on at gen "
        f"time): {train_marker_pos + len(marker_ids)}"
    )


if __name__ == "__main__":
    main()
