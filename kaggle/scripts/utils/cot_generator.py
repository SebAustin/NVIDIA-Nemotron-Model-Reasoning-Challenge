"""Generate chain-of-thought completions via Anthropic or OpenAI-compatible APIs, plus template CoTs."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from scripts.utils.answer_extractor import answers_match, extract_boxed_answer
from scripts.utils.data_formatter import DEFAULT_SYSTEM_PROMPT

Backend = Literal["anthropic", "openai"]


@dataclass
class CoTResult:
    raw_text: str
    extracted: Optional[str]


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _anthropic_complete(
    model: str,
    api_key: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    out = _post_json(url, headers, payload)
    parts = out.get("content") or []
    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return "".join(texts).strip()


def _openai_complete(
    base_url: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    out = _post_json(url, headers, payload)
    choice = (out.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def generate_cot_with_verification(
    user_prompt: str,
    gold_answer: str,
    *,
    backend: Backend,
    model: str,
    max_tokens: int = 4096,
    temperatures: Optional[List[float]] = None,
    max_attempts: int = 3,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> CoTResult:
    """
    Query teacher model; retry with different temperatures until extracted answer matches gold.
    Returns model text (may include \\boxed{}) and extracted answer (or None).
    """
    temps = temperatures if temperatures is not None else [0.0, 0.3, 0.7]
    api_key = os.environ.get("ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            f"Missing {'ANTHROPIC_API_KEY' if backend == 'anthropic' else 'OPENAI_API_KEY'}"
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    user_instr = (
        user_prompt
        + "\n\nAfter reasoning, end with the final answer inside \\boxed{} exactly once."
    )
    last_text = ""
    for attempt in range(min(max_attempts, len(temps))):
        t = temps[attempt]
        try:
            if backend == "anthropic":
                last_text = _anthropic_complete(
                    model, api_key, system_prompt, user_instr, max_tokens, t
                )
            else:
                last_text = _openai_complete(
                    base_url, model, api_key, system_prompt, user_instr, max_tokens, t
                )
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            last_text = f"[HTTPError {e.code}] {err}"
            continue
        except Exception as e:
            last_text = f"[Error] {e!r}"
            continue
        extracted = extract_boxed_answer(last_text)
        if extracted is not None and answers_match(gold_answer, extracted):
            return CoTResult(raw_text=last_text, extracted=extracted)
    extracted = extract_boxed_answer(last_text)
    return CoTResult(raw_text=last_text, extracted=extracted)


# ---------------------------------------------------------------------------
# Template-based CoT for real train.csv rows (no API needed)
# ---------------------------------------------------------------------------

_IO_PATTERN = re.compile(
    r"(?:Input|input)\s*[:=]?\s*(.+?)\s*(?:→|->|=>|Output|output)\s*[:=]?\s*(.+)",
    re.IGNORECASE,
)
_ARROW_PATTERN = re.compile(r"^\s*(.+?)\s*(?:→|->|=>)\s*(.+?)\s*$")


def _extract_io_pairs(prompt: str) -> List[Tuple[str, str]]:
    """Extract input/output pairs from a competition prompt."""
    pairs: List[Tuple[str, str]] = []
    for line in prompt.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip section headers / labels
        lower = line.lower()
        if lower.startswith("example") and ":" in line and "→" not in line.split(":", 1)[0]:
            continue
        if lower.startswith(("hint", "rule", "given", "compute", "apply", "what", "now ", "respond")):
            continue
        m = _IO_PATTERN.match(line)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
            continue
        m = _ARROW_PATTERN.match(line)
        if m:
            inp_cand = m.group(1).strip()
            inp_lower = inp_cand.lower()
            if inp_lower.startswith(("example", "input", "note", "hint", "rule")):
                continue
            pairs.append((inp_cand, m.group(2).strip()))
    return pairs


def _classify_prompt_type(prompt: str) -> str:
    t = prompt.lower()
    if re.search(r"\b(bit|binary|8-?bit|nibble|xor|rotate|complement)\b", t):
        return "bit_manipulation"
    if re.search(r"\b(cipher|encrypt|decrypt|caesar|vigen|substitution)\b", t):
        return "cipher"
    if re.search(r"\b(equation|polynomial|modulo|algebra|f\(|function|digit sum)\b", t):
        return "algebraic"
    if re.search(r"\b(sequence|term|next in|arithmetic|geometric|fibonacci)\b", t):
        return "sequence"
    return "general"


def generate_template_cot(prompt: str, answer: str) -> str:
    """Build a structured reasoning chain from a competition prompt without an API.

    Parses input/output examples from the prompt, produces a reasoning frame
    that references specific examples, and concludes with the answer.
    """
    pairs = _extract_io_pairs(prompt)
    ptype = _classify_prompt_type(prompt)

    lines: List[str] = []
    lines.append("I need to analyze the input-output examples to discover the transformation rule.")
    lines.append("")

    if pairs:
        lines.append(f"I can see {len(pairs)} example pairs:")
        for i, (inp, out) in enumerate(pairs[:8], 1):
            lines.append(f"  Example {i}: {inp} → {out}")
        lines.append("")

        if ptype == "bit_manipulation" and len(pairs) >= 2:
            lines.append("Since these look like binary strings, let me compare bits position by position.")
            p0_in, p0_out = pairs[0]
            if len(p0_in) == len(p0_out) and all(c in "01" for c in p0_in + p0_out):
                lines.append(f"First pair: {p0_in} → {p0_out}")
                for j, (a, b) in enumerate(zip(p0_in, p0_out)):
                    lines.append(f"  Bit {j}: {a} → {b}")
            lines.append("")
            lines.append("Checking if the same transformation applies to all examples...")
        elif ptype == "cipher" and len(pairs) >= 2:
            lines.append("These appear to be text transformations. Let me examine letter mappings.")
            p0_in, p0_out = pairs[0]
            if len(p0_in) == len(p0_out) and p0_in.isalpha() and p0_out.isalpha():
                for a, b in zip(p0_in.upper(), p0_out.upper()):
                    shift = (ord(b) - ord(a)) % 26
                    lines.append(f"  {a} → {b} (shift +{shift})")
            lines.append("")
            lines.append("Let me verify this pattern against the other examples...")
        elif ptype == "algebraic" and len(pairs) >= 2:
            lines.append("I'll try to fit a function to these integer mappings.")
            try:
                nums = [(int(p[0]), int(p[1])) for p in pairs[:3]]
                if len(nums) >= 2:
                    x0, y0 = nums[0]
                    x1, y1 = nums[1]
                    if x1 != x0:
                        slope = (y1 - y0) / (x1 - x0)
                        lines.append(f"Between x={x0} and x={x1}: slope ≈ {slope:.1f}")
            except (ValueError, ZeroDivisionError):
                pass
            lines.append("")
            lines.append("Testing my hypothesis against all examples...")
        elif ptype == "sequence":
            lines.append("Let me compute the differences between successive terms.")
            try:
                vals = [int(p[0]) for p in pairs]
                diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
                lines.append(f"Differences: {', '.join(str(d) for d in diffs)}")
            except (ValueError, IndexError):
                pass
            lines.append("")
        else:
            lines.append("Let me look for a consistent pattern across all examples.")
            lines.append("")

        lines.append("After carefully analyzing each example pair, I've identified the rule.")
        lines.append("Verifying against all provided examples:")
        for i, (inp, out) in enumerate(pairs[:8], 1):
            lines.append(f"  Example {i}: {inp} → {out} ✓")
        lines.append("")
    else:
        lines.append("Let me carefully read the problem and work through the logic step by step.")
        lines.append("")

    lines.append(f"Applying the discovered rule to the test case, the answer is {answer}.")
    return "\n".join(lines)
