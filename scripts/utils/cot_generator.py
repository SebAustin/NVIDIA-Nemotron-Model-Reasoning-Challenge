"""
Generate chain-of-thought reasoning traces via an external API (e.g. Claude).
Uses ANTHROPIC_API_KEY from environment; no keys in code.
"""
import os
from typing import Callable

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT


def call_anthropic(system: str, user: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    """Call Anthropic API. Requires ANTHROPIC_API_KEY in env."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text if msg.content else ""
    return text


def generate_cot_with_retries(
    puzzle_prompt: str,
    ground_truth_answer: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    call_model: Callable[[str, str], str] | None = None,
    max_retries: int = 3,
) -> tuple[str | None, bool]:
    """
    Generate CoT for one puzzle. Returns (assistant_content, correct).
    assistant_content includes CoT and final \\boxed{answer}.
    If no call_model is provided, uses call_anthropic.
    """
    if call_model is None:
        call_model = lambda sys, user: call_anthropic(sys, user)
    user_content = puzzle_prompt.strip()
    for attempt in range(max_retries):
        try:
            raw = call_model(system_prompt, user_content)
        except Exception as e:
            if attempt == max_retries - 1:
                return None, False
            continue
        extracted = extract_boxed_answer(raw)
        if answers_match(extracted, ground_truth_answer):
            # Normalize to end with \boxed{answer}
            if extracted is not None and not raw.strip().endswith("}"):
                if "\\boxed{" in raw:
                    raw = raw.strip()
                    if not raw.endswith("}"):
                        raw = raw + "\n\n\\boxed{" + extracted + "}"
                else:
                    raw = raw.strip() + "\n\n\\boxed{" + extracted + "}"
            return raw.strip(), True
    return None, False


def build_training_example(
    puzzle_prompt: str,
    ground_truth_answer: str,
    assistant_content: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """Build one JSONL training example with messages key."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle_prompt.strip()},
            {"role": "assistant", "content": assistant_content},
        ]
    }
