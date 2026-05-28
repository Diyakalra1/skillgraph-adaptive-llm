"""Multi-agent GRPO training for AMASES (one policy per agent).

This script aligns with AMASES' core idea:
- Run joint multi-agent interaction in the same adaptive environment.
- Collect per-turn rewards and split data by acting agent.
- Train one GRPO policy per agent (alpha/beta/gamma) from its own turns.

Usage (Colab/local):
  python -m skillgraph_adaptive_env.training.run_training_trl_grpo_multi_agent \
    --episodes 30 \
    --seed 7 \
    --max-turns 12 \
    --hf-token "$HF_TOKEN" \
    --out-dir training/runs/trl_grpo_multi_agent_day1
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import (
    SkillgraphAdaptiveEnvironment,
)

AGENT_MODEL_MAP_DEFAULT = {
    "agent_alpha": "meta-llama/Llama-3.2-1B-Instruct",
    "agent_beta": "Qwen/Qwen2.5-1.5B-Instruct",
    "agent_gamma": "google/gemma-3-1b-it",
}


def _build_prompt(obs, agent_id: str) -> str:
    return (
        f"You are {agent_id} in a multi-agent adaptive curriculum task.\n"
        f"Task type: {obs.task_type}\n"
        f"Task: {obs.task_prompt}\n"
        f"Skills: {', '.join(obs.task_skills)}\n"
        f"Turn: {obs.turn_index}/{obs.max_turns}\n"
        "Provide one concise actionable response."
    )


def _generate_response(
    client: InferenceClient,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> str:
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are concise, collaborative, and action-oriented.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    text = (completion.choices[0].message.content or "").strip()
    return text or "I propose a concrete next step with one supporting rationale."


def collect_joint_rollouts(
    episodes: int,
    seed: int,
    max_turns: int,
    client: InferenceClient,
    max_rollout_tokens: int,
) -> list[dict]:
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    rows: list[dict] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        guard = 0
        while not done and guard < max_turns:
            guard += 1
            agent_id = obs.current_agent_id or "agent_alpha"
            model_id = AGENT_MODEL_MAP_DEFAULT.get(agent_id, AGENT_MODEL_MAP_DEFAULT["agent_alpha"])
            prompt = _build_prompt(obs, agent_id)
            response = _generate_response(client, model_id, prompt, max_rollout_tokens)
            action = SkillgraphAdaptiveAction(
                agent_id=agent_id,
                task_id=obs.task_id,
                response_text=response,
                self_rating=0.6,
            )
            obs = env.step(action)
            rows.append(
                {
                    "episode": ep,
                    "turn": int(obs.turn_index),
                    "agent_id": agent_id,
                    "model_id": model_id,
                    "prompt": prompt,
                    "response": response,
                    "env_reward": float(obs.reward if obs.reward is not None else 0.0),
                    "task_type": obs.task_type,
                    "success": bool(obs.success),
                }
            )
            done = bool(obs.done)

    return rows


def _reward_fn(prompts, completions, **kwargs):
    rewards = kwargs.get("env_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(x) for x in rewards]


def train_single_agent_grpo(
    agent_id: str,
    model_id: str,
    rows: list[dict],
    out_dir: Path,
    epochs: int,
    learning_rate: float,
    max_completion_length: int,
    max_samples: int,
) -> dict:
    agent_rows = [r for r in rows if r["agent_id"] == agent_id]
    if not agent_rows:
        return {"agent_id": agent_id, "skipped": True, "reason": "no_rows"}

    clipped = agent_rows[: max(8, min(max_samples, len(agent_rows)))]
    train_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "env_reward": float(r["env_reward"])} for r in clipped]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ckpt_dir = out_dir / "checkpoints"
    cfg = GRPOConfig(
        output_dir=str(ckpt_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_completion_length=max_completion_length,
        num_generations=2,
        logging_steps=5,
        save_steps=100,
        report_to=[],
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=_reward_fn,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=None,
    )
    train_result = trainer.train()
    final_dir = ckpt_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log_history = getattr(train_result, "log_history", None) or getattr(trainer.state, "log_history", [])
    loss_points = [x for x in log_history if isinstance(x, dict) and "loss" in x]
    avg_reward = sum(float(r["env_reward"]) for r in agent_rows) / max(1, len(agent_rows))
    success_rate = sum(1 for r in agent_rows if r.get("success")) / max(1, len(agent_rows))
    return {
        "agent_id": agent_id,
        "model_id": model_id,
        "num_rows": len(agent_rows),
        "num_train_rows": len(clipped),
        "avg_env_reward": round(avg_reward, 4),
        "success_rate": round(success_rate, 4),
        "checkpoint_final": str(final_dir),
        "num_train_log_points": len(loss_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent GRPO training in AMASES.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--hf-token", type=str, required=True)
    parser.add_argument("--rollout-max-tokens", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="training/runs/trl_grpo_multi_agent")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "joint_rollouts.json"
    csv_path = out_dir / "joint_rollouts.csv"

    client = InferenceClient(api_key=args.hf_token.strip(), timeout=120)
    rows = collect_joint_rollouts(
        episodes=args.episodes,
        seed=args.seed,
        max_turns=args.max_turns,
        client=client,
        max_rollout_tokens=args.rollout_max_tokens,
    )
    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "turn",
                    "agent_id",
                    "model_id",
                    "task_type",
                    "env_reward",
                    "success",
                    "prompt",
                    "response",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    per_agent_results = []
    for agent_id, model_id in AGENT_MODEL_MAP_DEFAULT.items():
        agent_out = out_dir / agent_id
        agent_out.mkdir(parents=True, exist_ok=True)
        result = train_single_agent_grpo(
            agent_id=agent_id,
            model_id=model_id,
            rows=rows,
            out_dir=agent_out,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_completion_length=args.max_completion_length,
            max_samples=args.max_samples,
        )
        per_agent_results.append(result)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "trl_grpo_multi_agent",
        "episodes": args.episodes,
        "seed": args.seed,
        "num_joint_rows": len(rows),
        "joint_rows_path": str(rows_path),
        "joint_csv_path": str(csv_path),
        "agent_model_map": AGENT_MODEL_MAP_DEFAULT,
        "per_agent_results": per_agent_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
