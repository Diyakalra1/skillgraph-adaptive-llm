"""Budget-safe final AMASES run with 3 distinct HF models.

This script:
1) probes a candidate pool,
2) selects three distinct working models,
3) runs a capped training pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import InferenceClient

from skillgraph_adaptive_env.training.run_training_three_models import train


DEFAULT_CANDIDATE_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


def _model_works(client: InferenceClient, model_id: str, max_tokens: int) -> tuple[bool, str]:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply in one short line: ready."}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        return bool(text), ""
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip()
        return False, (msg.splitlines()[0] if msg else exc.__class__.__name__)


def _choose_three_distinct_models(
    token: str,
    timeout_s: int,
    candidate_models: list[str],
    probe_tokens: int,
) -> tuple[list[str], list[dict]]:
    client = InferenceClient(api_key=token, timeout=timeout_s)
    selected: list[str] = []
    probe_report: list[dict] = []
    for model_id in candidate_models:
        ok, err = _model_works(client, model_id, max_tokens=probe_tokens)
        probe_report.append({"model_id": model_id, "ok": ok, "error": err})
        if ok and model_id not in selected:
            selected.append(model_id)
        if len(selected) >= 3:
            break
    if len(selected) < 3:
        raise SystemExit(
            "Could not find 3 distinct working models from candidate list. "
            "Try adding more candidates or run with different account/provider access."
        )
    return selected[:3], probe_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final budget-safe AMASES training.")
    parser.add_argument("--hf-token", type=str, default="", help="HF token or set HF_TOKEN.")
    parser.add_argument("--episodes", type=int, default=8, help="Low-cost final episode count.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--request-gap-s", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--turn-cap", type=int, default=10, help="Budget cap for turns per episode.")
    parser.add_argument("--probe-tokens", type=int, default=24, help="Tokens used for model health probes.")
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--out-dir", type=str, default="training/runs/final_budget_hf_results")
    parser.add_argument(
        "--candidate-models",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_MODELS),
        help="Comma-separated candidate model ids (probe order).",
    )
    args = parser.parse_args()

    token = args.hf_token.strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing HF token. Use --hf-token or set HF_TOKEN.")

    candidates = [m.strip() for m in args.candidate_models.split(",") if m.strip()]
    selected, probe_report = _choose_three_distinct_models(
        token=token,
        timeout_s=max(30, args.timeout_s),
        candidate_models=candidates,
        probe_tokens=max(8, args.probe_tokens),
    )
    model_map = {
        "agent_alpha": selected[0],
        "agent_beta": selected[1],
        "agent_gamma": selected[2],
    }

    summary_path = train(
        episodes=max(1, args.episodes),
        seed=args.seed,
        out_dir=Path(args.out_dir),
        token=token,
        model_map=model_map,
        request_gap_s=max(0.0, args.request_gap_s),
        max_tokens=max(16, args.max_tokens),
        turn_cap=max(3, args.turn_cap),
    )

    payload = {
        "summary_path": str(summary_path),
        "model_map": model_map,
        "probe_report": probe_report,
        "budget_profile": {
            "episodes": max(1, args.episodes),
            "turn_cap": max(3, args.turn_cap),
            "request_gap_s": max(0.0, args.request_gap_s),
            "max_tokens": max(16, args.max_tokens),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
