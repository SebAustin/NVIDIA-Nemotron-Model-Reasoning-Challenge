"""
Extract final answer from model output using \\boxed{}.
Align with competition metric when published.
"""
import re


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the last \\boxed{...} content from text.
    Handles nested braces in the content.
    """
    if not text or not isinstance(text, str):
        return None
    # Match \boxed{...} with possible nested braces
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """
    Compare predicted answer to ground truth: exact string match or
    numeric match within relative tolerance (1e-6 * |gt|).
    """
    if predicted is None:
        return False
    pred = predicted.strip()
    gt = ground_truth.strip()
    if pred == gt:
        return True
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        if gt_val == 0:
            return pred_val == 0
        return abs(pred_val - gt_val) <= 1e-6 * abs(gt_val)
    except (ValueError, TypeError):
        pass
    return False
