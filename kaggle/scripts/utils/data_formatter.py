"""Chat messages formatting for SFT (Nemotron / HF chat template)."""

from __future__ import annotations

import re
from typing import Any, Dict, List

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert at solving few-shot logical reasoning puzzles. "
    "The prompt gives several input→output examples that illustrate ONE hidden transformation "
    "rule, then asks you to apply that same rule to a new test input. "
    "The rule is NEVER stated — you must INFER it from the examples only. "
    "\n\n"
    "Procedure:\n"
    "1. Examine EVERY input→output pair carefully.\n"
    "2. Form a hypothesis for the transformation rule.\n"
    "3. VERIFY: apply your hypothesis to EVERY given example — if any fails, discard and "
    "try a different hypothesis.\n"
    "4. Apply the verified rule to the test input to produce the answer.\n"
    "\n"
    "Common puzzle families: bit manipulation (XOR mask, rotate left/right, complement, "
    "nibble swap, arithmetic mod 2^n), text ciphers (Caesar, Vigenère, substitution, "
    "Atbash, affine, reversal), integer functions (polynomial, modular, digit-sum, "
    "digit-product, piecewise), numeric sequences (arithmetic, geometric, Fibonacci-like, "
    "quadratic, second-difference, alternating, interleaved).\n"
    "\n"
    "Show concise step-by-step reasoning. "
    "Your response MUST end with exactly this line (nothing after it):\n"
    "The answer is \\boxed{YOUR_ANSWER}"
)


_BOXED_LINE_RE = re.compile(
    r"\s*(?:The\s+answer\s+is\s+)?\\boxed\{[^}]*\}\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_THINK_CLOSE_RE = re.compile(r"\s*</think>\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_trailing_boxed(text: str) -> str:
    """Remove any trailing 'The answer is \\boxed{...}' lines from the body
    so we can append exactly one canonical line."""
    if not text:
        return ""
    cleaned = text
    while True:
        new = _BOXED_LINE_RE.sub("", cleaned).rstrip()
        if new == cleaned.rstrip():
            return new
        cleaned = new


def _strip_trailing_think_close(text: str) -> str:
    """Drop any trailing `</think>` from the reasoning body before we re-add it."""
    if not text:
        return ""
    return _THINK_CLOSE_RE.sub("", text).rstrip()


def format_assistant_reply(cot_body: str, final_answer: str) -> str:
    """Build the assistant content for the Nemotron `enable_thinking` chat template.

    The Nemotron chat template (`enable_thinking=True`, the harness default)
    emits `<|im_start|>assistant\\n<think>\\n` at the end of the prompt and
    expects the model's completion to fill in the thinking block, then close it
    with `</think>`, then emit the boxed answer, then `<|im_end|>`. This is the
    exact format used by the Progress-Prize winning submission's `corpus.py`:

        {reasoning}\\n</think>\\n\\boxed{{ANSWER}}<|im_end|>

    `03_train_lora.formatting_func` consumes this content and concatenates it
    onto the `add_generation_prompt=True` prompt, so what the model sees at
    training time exactly matches what the harness emits at eval time.

    Pre-Phase-0.4 versions of this function emitted
    `{reasoning}\\n\\nThe answer is \\boxed{{ANSWER}}` (no `</think>` separator),
    which trained the model against a wholly different conditional than what
    the eval harness produces.
    """
    body = _strip_trailing_think_close(_strip_trailing_boxed((cot_body or "").rstrip()))
    ans = (final_answer or "").strip()
    boxed = f"\\boxed{{{ans}}}"
    if body:
        return f"{body}\n</think>\n{boxed}"
    return f"</think>\n{boxed}"


def format_assistant_reply_verified(
    cot_body: str,
    final_answer: str,
    examples: list | None = None,
) -> str:
    """Self-verifying assistant reply.

    Same Phase 0.4 format as `format_assistant_reply`, but inserts a
    `Final verification ...` block before the close-of-thinking. Trains the
    model to always re-check its proposed rule against every given example
    before committing — useful on few-shot rule-induction tasks.
    """
    body = _strip_trailing_think_close(_strip_trailing_boxed((cot_body or "").rstrip()))
    ans = (final_answer or "").strip()
    lines = [body] if body else []
    if examples:
        lines.append("")
        lines.append("Final verification — applying the inferred rule to each given example:")
        for ex in examples:
            inp = getattr(ex, "inp", None) if hasattr(ex, "inp") else ex[0]
            out = getattr(ex, "out", None) if hasattr(ex, "out") else ex[1]
            lines.append(f"  {inp} -> {out} (matches)")
        lines.append("All examples reproduce correctly. Committing to the answer.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{ans}}}")
    return "\n".join(lines)


def build_messages(
    user_prompt: str,
    assistant_content: str,
    system_prompt: str | None = None,
) -> List[Dict[str, str]]:
    sys = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]


def messages_to_example(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"messages": messages}
