"""Chat formatting for SFT, matching the Nemotron eval template exactly.

Verified against the competition (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
tokenizer chat_template) and the Progress-Prize-winning submission:

* NO system prompt at eval. The user turn is the raw puzzle prompt followed by a
  fixed boxed-answer instruction suffix.
* ``apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)``
  ends the prompt with ``<|im_start|>assistant\\n<think>\\n``.
* The assistant target therefore continues from the open ``<think>``: reasoning,
  ``</think>``, then the final ``\\boxed{answer}``.

So the assistant *content* we store is ``<think>\\n{reasoning}\\n</think>\\n\\boxed{answer}``;
rendering [user, assistant] through the chat template reproduces the eval
distribution exactly. We keep train_sft.jsonl tokenizer-free (the trainer applies
the chat template on the Kaggle side).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Exact suffix used by the eval harness / winning submission.
BOXED_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def build_user_content(prompt: str, add_suffix: bool = True) -> str:
    base = str(prompt).rstrip()
    return base + BOXED_SUFFIX if add_suffix else base


def format_assistant_reply(reasoning: str, final_answer: str) -> str:
    """Assistant target: ``<think>\\n{reasoning}\\n</think>\\n\\boxed{answer}``.

    An empty ``reasoning`` yields a direct-answer target (empty think block),
    which we use for the ~25% direct-answer mix that preserves base behaviour.
    """
    ans = (final_answer or "").strip()
    body = (reasoning or "").strip()
    if body:
        return f"<think>\n{body}\n</think>\n\\boxed{{{ans}}}"
    return f"<think>\n</think>\n\\boxed{{{ans}}}"


def build_messages(
    user_prompt: str,
    assistant_content: str,
    system_prompt: Optional[str] = None,
    add_suffix: bool = True,
) -> List[Dict[str, str]]:
    """Return chat messages. By default NO system message (matches eval)."""
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": build_user_content(user_prompt, add_suffix)})
    msgs.append({"role": "assistant", "content": assistant_content})
    return msgs


def messages_to_example(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"messages": messages}
