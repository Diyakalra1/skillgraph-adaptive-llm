"""Hugging Face runtime wrapper with retries and failure recording."""

from __future__ import annotations

import time
from dataclasses import dataclass

from huggingface_hub import InferenceClient


@dataclass
class RuntimeResult:
    ok: bool
    text: str
    error: str
    elapsed_s: float
    retries_used: int


class HfModelRuntime:
    """Thin runtime layer for model inference with robust fallbacks."""

    def __init__(self, token: str, timeout_s: int = 90, max_retries: int = 2) -> None:
        self._client = InferenceClient(api_key=token, timeout=timeout_s)
        self._max_retries = max_retries

    def generate(self, model_id: str, prompt: str) -> RuntimeResult:
        started = time.time()
        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are concise and task focused."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=180,
                    temperature=0.2,
                )
                text = response.choices[0].message.content.strip()
                return RuntimeResult(
                    ok=True,
                    text=text,
                    error="",
                    elapsed_s=time.time() - started,
                    retries_used=attempt,
                )
            except Exception as exc:
                last_error = str(exc).splitlines()[0]
                # backoff
                if attempt < self._max_retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                return RuntimeResult(
                    ok=False,
                    text="",
                    error=last_error,
                    elapsed_s=time.time() - started,
                    retries_used=attempt,
                )
        return RuntimeResult(
            ok=False,
            text="",
            error=last_error or "unknown_error",
            elapsed_s=time.time() - started,
            retries_used=self._max_retries,
        )
