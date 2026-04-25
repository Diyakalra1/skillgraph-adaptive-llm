"""Minimal standalone Hugging Face 3-model test."""

import os

from huggingface_hub import InferenceClient

PROMPT = "hello who are you"

# You can change these if needed.
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def ask_model(client: InferenceClient, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing HF_TOKEN env var.")
    client = InferenceClient(api_key=token, timeout=90)
    for idx, model in enumerate(MODELS, start=1):
        print(f"\n[{idx}] MODEL: {model}")
        try:
            text = ask_model(client, model, PROMPT)
            print("OUTPUT:", text)
        except Exception as exc:
            print("ERROR:", exc)


if __name__ == "__main__":
    main()
