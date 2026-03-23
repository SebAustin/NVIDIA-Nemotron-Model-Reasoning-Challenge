"""Chat messages formatting for SFT (Nemotron / HF chat template)."""

from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert logical reasoning assistant. Analyze the given input-output examples "
    "to discover the underlying transformation rule. Think step-by-step: first identify the "
    "pattern, then verify it against all examples, then apply it to solve the test case. "
    "Always place your final answer inside \\boxed{}."
)


def format_assistant_reply(cot_body: str, final_answer: str) -> str:
    """Build assistant message: reasoning + mandatory boxed line."""
    body = (cot_body or "").rstrip()
    ans = (final_answer or "").strip()
    boxed = f"The answer is \\boxed{{{ans}}}"
    if body:
        return f"{body}\n\n{boxed}"
    return boxed


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
