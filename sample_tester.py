"""Quick Hugging Face multi-model tester for AMASES."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from huggingface_hub import InferenceClient


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]


def _extract_text(response: Any) -> str:
    """Best-effort extraction for different provider response formats."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    try:
        return response.choices[0].message.content.strip()  # chat completion style
    except Exception:
        pass
    return str(response).strip()


def run_once(token: str, model: str, prompt: str, timeout_s: int) -> tuple[bool, str, float]:
    """Try chat completion first, then text_generation fallback."""
    client = InferenceClient(api_key=token, timeout=timeout_s)
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep answer concise."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=160,
            temperature=0.2,
        )
        return True, _extract_text(response), time.time() - started
    except Exception as chat_exc:
        try:
            response = client.text_generation(
                prompt=prompt,
                model=model,
                max_new_tokens=160,
                temperature=0.2,
                return_full_text=False,
            )
            return True, _extract_text(response), time.time() - started
        except Exception as text_exc:
            return False, f"chat_error={chat_exc}\ntext_error={text_exc}", time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Test 3 Hugging Face models quickly.")
    parser.add_argument("--token", type=str, default="", help="HF token. If omitted, uses HF_TOKEN env var.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Give me 3 bullet points on adaptive curriculum learning.",
        help="Prompt for all models.",
    )
    parser.add_argument("--timeout", type=int, default=90, help="Per-model timeout in seconds.")
    parser.add_argument("--model1", type=str, default=DEFAULT_MODELS[0])
    parser.add_argument("--model2", type=str, default=DEFAULT_MODELS[1])
    parser.add_argument("--model3", type=str, default=DEFAULT_MODELS[2])
    args = parser.parse_args()

    token = args.token.strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing HF token. Pass --token or set HF_TOKEN.")

    models = [args.model1, args.model2, args.model3]
    print("=== Hugging Face 3-Model Sample Tester ===")
    print(f"Prompt: {args.prompt}")
    print("")

    for idx, model in enumerate(models, start=1):
        print(f"[{idx}/3] Testing model: {model}")
        ok, output, elapsed = run_once(token=token, model=model, prompt=args.prompt, timeout_s=args.timeout)
        status = "SUCCESS" if ok else "FAILED"
        print(f"Status: {status} | elapsed={elapsed:.2f}s")
        print("Output:")
        print(output[:1000] if output else "(empty)")
        print("-" * 80)


if __name__ == "__main__":
    main()
