"""
Format chat messages for Nemotron using the model's chat template.
"""
from transformers import AutoTokenizer


def messages_to_nemotron_string(
    messages: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Convert a list of messages (system, user, assistant, ...) to a single
    string using the Nemotron chat template (for SFT training or inference).
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_messages(
    system_prompt: str,
    user_content: str,
    assistant_content: str | None = None,
) -> list[dict[str, str]]:
    """
    Build messages list for training: system, user, and optional assistant.
    """
    out: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if assistant_content is not None:
        out.append({"role": "assistant", "content": assistant_content})
    return out


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert logical reasoning assistant. Analyze the given input-output "
    "examples to discover the underlying transformation rule. Think step-by-step: "
    "first identify the pattern, then verify it against all examples, then apply it "
    "to solve the test case. Always place your final answer inside \\boxed{}."
)
