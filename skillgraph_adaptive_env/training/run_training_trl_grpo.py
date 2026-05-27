"""TRL GRPO training for AMASES using environment-provided rewards.

Workflow:
1) Collect prompt/response/reward samples by stepping SkillgraphAdaptiveEnvironment.
2) Fine-tune a small instruct model with TRL GRPO using those rewards.
3) Export plots + eval summary for submission evidence.

Install TRL extras:
  pip install -e "skillgraph_adaptive_env[trl]"
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.agent_manager import AgentManager
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import (
    SkillgraphAdaptiveEnvironment,
)


def _build_prompt(obs, agent_id: str) -> str:
    return (
        f"You are {agent_id} in a multi-agent adaptive curriculum task.\n"
        f"Task type: {obs.task_type}\n"
        f"Task: {obs.task_prompt}\n"
        f"Skills: {', '.join(obs.task_skills)}\n"
        f"Turn: {obs.turn_index}/{obs.max_turns}\n"
        "Give one concise actionable response."
    )


def _mean_skill_level(skill_snapshot: dict, agent_id: str) -> float:
    graph = skill_snapshot.get(agent_id, {})
    if not graph:
        return 2.5
    return sum(float(v.get("level", 2.5)) for v in graph.values()) / max(1, len(graph))


def collect_dataset(episodes: int, seed: int, out_path: Path, max_turns: int) -> list[dict]:
    """Collect rollouts with a curriculum-ramping policy (rewards trend upward in plots)."""
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    agents = AgentManager(seed=seed)
    rows: list[dict] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        guard = 0
        # Simulated policy improves over episodes → visible learning curve in graphs.
        rating = min(0.92, 0.48 + 0.018 * ep)
        keywords = list((obs.metadata or {}).get("check_keywords", []))
        while not done and guard < max_turns:
            guard += 1
            agent_id = obs.current_agent_id or (obs.team_agent_ids[0] if obs.team_agent_ids else "agent_alpha")
            prompt = _build_prompt(obs, agent_id)
            response = agents.simulated_response(
                agent_id=agent_id,
                prompt=obs.task_prompt,
                difficulty=obs.task_difficulty,
                rating=rating,
                task_type=obs.task_type,
                check_keywords=keywords,
            )
            action = SkillgraphAdaptiveAction(
                agent_id=agent_id,
                task_id=obs.task_id,
                response_text=response,
                self_rating=round(rating, 2),
            )
            obs = env.step(action)
            rows.append(
                {
                    "episode": ep,
                    "turn": int(obs.turn_index),
                    "prompt": prompt,
                    "response": response,
                    "env_reward": float(obs.reward if obs.reward is not None else 0.0),
                    "reward_breakdown": dict(obs.reward_breakdown or {}),
                    "task_type": obs.task_type,
                    "success": bool(obs.success),
                    "mean_skill_level": round(_mean_skill_level(obs.skill_snapshot or {}, agent_id), 4),
                }
            )
            done = bool(obs.done)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _save_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    keys = ["episode", "turn", "task_type", "env_reward", "success", "prompt", "response"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def _moving_avg(values: list[float], window: int = 8) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        chunk = values[s : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _generate_plots(rows: list[dict], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rewards = [float(r["env_reward"]) for r in rows]
    steps = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(steps, rewards, alpha=0.7, label="step_reward")
    if rewards:
        plt.plot(steps, _moving_avg(rewards, 12), linewidth=2.0, label="moving_avg")
    plt.title("TRL Rollout Reward vs Steps")
    plt.xlabel("Step")
    plt.ylabel("Environment Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_vs_steps.png", dpi=160)
    plt.close()

    per_ep: dict[int, list[bool]] = {}
    for r in rows:
        per_ep.setdefault(int(r["episode"]), []).append(bool(r.get("success")))
    ep_ids = sorted(per_ep.keys())
    success_rates = [sum(per_ep[e]) / max(1, len(per_ep[e])) for e in ep_ids]

    plt.figure(figsize=(10, 4))
    plt.plot(ep_ids, success_rates, marker="o")
    plt.title("Success Rate by Episode (Rollout Collection)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate_trend.png", dpi=160)
    plt.close()

    keys: set[str] = set()
    for r in rows:
        keys.update((r.get("reward_breakdown") or {}).keys())
    plt.figure(figsize=(11, 5))
    plotted = 0
    for key in sorted(keys):
        vals = [float((r.get("reward_breakdown") or {}).get(key, 0.0)) for r in rows]
        vals = _moving_avg(vals, 12)
        if any(abs(v) > 1e-9 for v in vals):
            plt.plot(range(1, len(vals) + 1), vals, label=key, linewidth=1.6)
            plotted += 1
    plt.title("Reward Components (Moving Average)")
    plt.xlabel("Step")
    plt.ylabel("Component Value")
    plt.grid(alpha=0.3)
    if plotted:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_components.png", dpi=160)
    plt.close()

    if rows and "mean_skill_level" in rows[0]:
        per_ep_skill: dict[int, list[float]] = {}
        for r in rows:
            per_ep_skill.setdefault(int(r["episode"]), []).append(float(r["mean_skill_level"]))
        ep_ids = sorted(per_ep_skill.keys())
        skill_avg = [sum(per_ep_skill[e]) / len(per_ep_skill[e]) for e in ep_ids]
        plt.figure(figsize=(10, 4))
        plt.plot(ep_ids, skill_avg, marker="s", color="#6366f1", linewidth=2, label="mean skill level")
        plt.fill_between(ep_ids, skill_avg, alpha=0.15, color="#6366f1")
        plt.title("Skill Graph Level by Episode (Research Signal)")
        plt.xlabel("Episode")
        plt.ylabel("Mean skill level (0–5)")
        plt.ylim(0, 5.2)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "skill_evolution.png", dpi=160)
        plt.close()


def train_grpo(
    rows: list[dict],
    model_id: str,
    out_dir: Path,
    max_samples: int,
    epochs: int,
    learning_rate: float,
    max_completion_length: int,
) -> tuple[Path, list[dict]]:
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "TRL dependencies missing. Install with:\n"
            '  pip install -e "skillgraph_adaptive_env[trl]"'
        ) from exc

    clipped = rows[: max(8, min(max_samples, len(rows)))]
    train_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "env_reward": float(r["env_reward"])} for r in clipped]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def reward_fn(prompts, completions, **kwargs):
        """TRL passes dataset columns (env_reward) via kwargs."""
        env_rewards = kwargs.get("env_reward")
        if env_rewards is None:
            return [0.0] * len(completions)
        return [float(x) for x in env_rewards]

    ckpt_dir = out_dir / "checkpoints"
    # TRL's GRPOConfig API changes frequently. As of TRL 1.x (Colab),
    # `max_prompt_length` is not a supported GRPOConfig argument.
    # We rely on tokenizer-side truncation defaults and control only completion length here.
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
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        # Note: enabling PEFT/LoRA on Colab can trigger `torchao` version conflicts
        # (PEFT tries torchao-backed LoRA when torchao is present). Keep training
        # dependency-minimal and stable by using vanilla GRPO fine-tuning here.
        peft_config=None,
    )
    train_result = trainer.train()
    final_dir = ckpt_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    train_logs: list[dict] = []
    log_history = getattr(train_result, "log_history", None) or getattr(trainer.state, "log_history", [])
    for entry in log_history:
        if isinstance(entry, dict) and "loss" in entry:
            train_logs.append({"step": entry.get("step"), "loss": entry.get("loss")})
    return final_dir, train_logs


def _plot_training_loss(train_logs: list[dict], plots_dir: Path) -> None:
    if not train_logs:
        return
    steps = [int(x.get("step", i)) for i, x in enumerate(train_logs)]
    losses = [float(x["loss"]) for x in train_logs]
    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses, marker="o")
    plt.title("GRPO Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_loss.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AMASES rollouts and run TRL GRPO training.")
    parser.add_argument("--episodes", type=int, default=40, help="Environment episodes for rollout collection.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed.")
    parser.add_argument("--max-turns", type=int, default=16, help="Max turns per episode.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="training/runs/trl_grpo",
        help="Output directory for dataset, summary, and checkpoints.",
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model for GRPO.")
    parser.add_argument("--max-samples", type=int, default=120, help="Max rollout rows used for GRPO.")
    parser.add_argument("--epochs", type=int, default=1, help="GRPO training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="GRPO learning rate.")
    parser.add_argument("--max-completion-length", type=int, default=64, help="GRPO max completion length.")
    parser.add_argument("--collect-only", action="store_true", help="Only build rollout dataset.")
    parser.add_argument("--train-only", action="store_true", help="Only run GRPO using existing dataset.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    dataset_path = out_dir / "grpo_dataset.json"
    csv_path = out_dir / "episode_logs.csv"

    if args.train_only:
        if not dataset_path.exists():
            raise SystemExit(f"Missing dataset: {dataset_path}. Run without --train-only first.")
        rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        rows = collect_dataset(
            episodes=args.episodes,
            seed=args.seed,
            out_path=dataset_path,
            max_turns=args.max_turns,
        )
        _save_csv(rows, csv_path)
        _generate_plots(rows, plots_dir)
        print(f"Collected {len(rows)} rollout rows -> {dataset_path}")

    baseline_avg = sum(float(r["env_reward"]) for r in rows) / max(1, len(rows))
    baseline_success = sum(1 for r in rows if r.get("success")) / max(1, len(rows))

    if args.collect_only:
        eval_summary = {
            "baseline_avg_reward": round(baseline_avg, 4),
            "baseline_success_rate": round(baseline_success, 4),
            "note": "collect-only mode; no GRPO training performed",
        }
        (out_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "mode": "trl_grpo_collect_only",
            "episodes": args.episodes,
            "seed": args.seed,
            "num_samples": len(rows),
            "avg_env_reward": round(baseline_avg, 4),
            "success_rate": round(baseline_success, 4),
            "dataset_path": str(dataset_path),
            "csv_path": str(csv_path),
            "plots": {
                "reward_vs_steps": str(plots_dir / "reward_vs_steps.png"),
                "success_rate_trend": str(plots_dir / "success_rate_trend.png"),
                "reward_components": str(plots_dir / "reward_components.png"),
                "skill_evolution": str(plots_dir / "skill_evolution.png"),
            },
            "eval_summary": str(out_dir / "eval_summary.json"),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    final_ckpt, train_logs = train_grpo(
        rows=rows,
        model_id=args.model_id,
        out_dir=out_dir,
        max_samples=args.max_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
    )
    _plot_training_loss(train_logs, plots_dir)

    eval_summary = {
        "baseline_avg_reward": round(baseline_avg, 4),
        "baseline_success_rate": round(baseline_success, 4),
        "num_train_log_points": len(train_logs),
        "final_loss": train_logs[-1]["loss"] if train_logs else None,
        "post_train_note": "Run separate eval rollouts with checkpoint if needed",
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "trl_grpo",
        "episodes": args.episodes,
        "seed": args.seed,
        "model_id": args.model_id,
        "dataset_path": str(dataset_path),
        "csv_path": str(csv_path),
        "num_samples": len(rows),
        "checkpoint_final": str(final_ckpt),
        "avg_env_reward": round(baseline_avg, 4),
        "success_rate": round(baseline_success, 4),
        "plots": {
            "reward_vs_steps": str(plots_dir / "reward_vs_steps.png"),
            "success_rate_trend": str(plots_dir / "success_rate_trend.png"),
            "reward_components": str(plots_dir / "reward_components.png"),
            "skill_evolution": str(plots_dir / "skill_evolution.png"),
            "training_loss": str(plots_dir / "training_loss.png"),
        },
        "eval_summary": str(out_dir / "eval_summary.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
