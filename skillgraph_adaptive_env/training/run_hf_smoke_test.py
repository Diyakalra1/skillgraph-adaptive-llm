"""Small HF smoke test for AMASES model connectivity and inference.

This performs only a few low-token requests to validate your setup.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from skillgraph_adaptive_env.server.model_runtime import HfModelRuntime

SMOKE_MODEL_MAP_DEFAULT = {
    "agent_alpha": "meta-llama/Llama-3.2-1B-Instruct",
    "agent_beta": "Qwen/Qwen2.5-1.5B-Instruct",
    "agent_gamma": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def _prompt(agent_id: str) -> str:
    return (
        f"You are {agent_id}.\n"
        "Give a one-line concise response proving model inference works.\n"
        "Mention: plan, trade-off, and evidence."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-cost HF smoke test for three AMASES agents.")
    parser.add_argument("--hf-token", type=str, default="", help="HF token or set HF_TOKEN env var.")
    parser.add_argument("--request-gap-s", type=float, default=2.5, help="Sleep between requests.")
    parser.add_argument("--timeout-s", type=int, default=90, help="HF request timeout.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per request.")
    parser.add_argument("--model-alpha", type=str, default=SMOKE_MODEL_MAP_DEFAULT["agent_alpha"])
    parser.add_argument("--model-beta", type=str, default=SMOKE_MODEL_MAP_DEFAULT["agent_beta"])
    parser.add_argument("--model-gamma", type=str, default=SMOKE_MODEL_MAP_DEFAULT["agent_gamma"])
    args = parser.parse_args()

    token = args.hf_token.strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing HF token. Use --hf-token or set HF_TOKEN.")

    model_map = {
        "agent_alpha": args.model_alpha,
        "agent_beta": args.model_beta,
        "agent_gamma": args.model_gamma,
    }
    runtime = HfModelRuntime(token=token, timeout_s=max(30, args.timeout_s), max_retries=max(0, args.max_retries))

    rows: list[dict] = []
    for agent_id, model_id in model_map.items():
        result = runtime.generate(model_id=model_id, prompt=_prompt(agent_id))
        rows.append(
            {
                "agent_id": agent_id,
                "model_id": model_id,
                "ok": result.ok,
                "elapsed_s": round(result.elapsed_s, 3),
                "retries_used": result.retries_used,
                "error": result.error,
                "sample_text": result.text[:140],
            }
        )
        if args.request_gap_s > 0:
            time.sleep(args.request_gap_s)

    output = {
        "timestamp": datetime.now().isoformat(),
        "request_gap_s": args.request_gap_s,
        "results": rows,
    }
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
