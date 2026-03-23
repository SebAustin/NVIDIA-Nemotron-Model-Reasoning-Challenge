"""Generate chain-of-thought completions via Anthropic or OpenAI-compatible APIs."""

from __future__ import annotations

import json
import os
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
