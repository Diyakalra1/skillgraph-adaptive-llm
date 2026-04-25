"""Single final TRL entrypoint for real 3-model optimization.

Pipeline:
1) Collect rollout trajectories using three real HF models (alpha/beta/gamma).
2) Export final_run artifacts for UI (summary/csv/jsonl/graphs).
3) Optimize each model separately with TRL (3 independent specialists).

No simulated agent responses are used in rollout collection.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.model_runtime import HfModelRuntime
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import (
    SkillgraphAdaptiveEnvironment,
)


@dataclass
class EpisodeSample:
    agent_id: str
    model_id: str
    prompt: str
    response: str
    reward_scalar: float
    reward_breakdown: dict


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _generate_plots(rows: list[dict], skill_series: dict[str, list[float]], out_dir: Path) -> None:
    episodes = list(range(1, len(rows) + 1))
    rewards = [float(r["reward"]) for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, linewidth=1.6, marker="o", markersize=2.6, alpha=0.9)
    plt.title("Reward vs Training Steps")
    plt.xlabel("Episode-Turn")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_steps.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    for agent_id, series in skill_series.items():
        plt.plot(
            range(1, len(series) + 1),
            series,
            label=agent_id,
            linewidth=2.0,
            marker="o",
            markersize=2.3,
            alpha=0.9,
        )
    plt.title("Agent Mean Skill Evolution")
    plt.xlabel("Episode")
    plt.ylabel("Mean skill level (0-5)")
    plt.ylim(0, 5.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "skill_evolution.png", dpi=160)
    plt.close()

    weakest_agent = min(skill_series.items(), key=lambda item: item[1][0])[0]
    trajectory = skill_series[weakest_agent]
    plt.figure(figsize=(10, 4))
    plt.plot(
        range(1, len(trajectory) + 1),
        trajectory,
        color="tab:red",
        linewidth=2.2,
        marker="o",
        markersize=2.8,
    )
    plt.title(f"Weak to Strong Transition: {weakest_agent}")
    plt.xlabel("Episode")
    plt.ylabel("Mean skill level (0-5)")
    plt.ylim(0, 5.1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "weak_to_strong_transition.png", dpi=160)
    plt.close()


def _build_prompt(obs, agent_id: str) -> str:
    private = obs.private_observation or {}
    public = obs.public_observation or {}
    public_msgs = public.get("recent_public_messages", [])
    public_text = "\n".join(
        f"- T{m.get('turn', '?')} {m.get('agent_id', 'agent')}: {m.get('content', '')}"
        for m in public_msgs[-4:]
    )
    return (
        f"You are {agent_id} in a multi-agent task.\n"
        f"TaskType={obs.task_type}\n"
        f"Task={obs.task_prompt}\n"
        f"Skills={','.join(obs.task_skills)}\n"
        f"Turn={obs.turn_index}/{obs.max_turns}\n"
        f"PrivatePreference={private.get('visible_private_preference', '')}\n"
        f"RecentThread:\n{public_text}\n\n"
        "Respond in exactly 3 bullets:\n"
        "- Proposal: concrete offer/plan\n"
        "- Constraint/Evidence: one reason tied to constraints\n"
        "- Next move: counter-offer, question, or action"
    )


def _policy_self_rating(skill_snapshot: dict, task_skills: list[str], difficulty: float) -> float:
    def _safe_skill(skill_name: str) -> tuple[float, float]:
        node = skill_snapshot.get(skill_name, {})
        return float(node.get("level", 2.5)), float(node.get("confidence", 0.2))

    vals = [_safe_skill(s) for s in task_skills] if task_skills else [(2.5, 0.2)]
    skill_avg = sum(v[0] for v in vals) / max(1, len(vals))
    confidence_avg = sum(v[1] for v in vals) / max(1, len(vals))
    rating = (0.6 * (skill_avg / 5.0)) + (0.4 * confidence_avg) - (0.12 * difficulty) + 0.20
    return max(0.0, min(1.0, rating))


def run_real_rollouts(
    episodes: int,
    seed: int,
    out_dir: Path,
    turn_cap: int,
    dataset_out: Path,
    token: str,
    model_map: dict[str, str],
    request_gap_s: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    runtime = HfModelRuntime(token=token, timeout_s=120, max_retries=2)

    rows: list[dict] = []
    samples: list[EpisodeSample] = []
    skill_series: dict[str, list[float]] = {}
    csv_path = out_dir / "episode_logs.csv"
    jsonl_path = out_dir / "episode_logs.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "task_id",
                "task_type",
                "agent_id",
                "model_id",
                "turn",
                "skills",
                "difficulty",
                "curriculum_bucket",
                "self_rating",
                "success",
                "reward",
                "task_score",
                "skill_improvement",
                "consistency",
                "skill_drop",
                "response_text",
            ],
        )
        writer.writeheader()

        for ep in range(1, episodes + 1):
            obs = env.reset()
            done = False
            guard = 0
            last_obs = obs
            episode_turn_cap = max(3, int(obs.max_turns or 0), turn_cap)
            while not done and guard < episode_turn_cap:
                guard += 1
                current_agent = obs.current_agent_id or (
                    obs.team_agent_ids[0] if obs.team_agent_ids else "agent_alpha"
                )
                model_id = model_map[current_agent]
                rating = _policy_self_rating(
                    obs.skill_snapshot[current_agent], obs.task_skills, obs.task_difficulty
                )
                prompt = _build_prompt(obs, current_agent)
                generation = runtime.generate(model_id=model_id, prompt=prompt)
                if not generation.ok:
                    raise RuntimeError(
                        f"strict_no_fallback:model_generation_failed agent={current_agent} "
                        f"model={model_id} error={generation.error}"
                    )
                response = generation.text.strip()
                if request_gap_s > 0:
                    time.sleep(request_gap_s)
                action = SkillgraphAdaptiveAction(
                    agent_id=current_agent,
                    task_id=obs.task_id,
                    response_text=response,
                    self_rating=rating,
                )
                obs = env.step(action)
                rb = obs.reward_breakdown or {}
                row = {
                    "episode": ep,
                    "task_id": obs.task_id,
                    "task_type": obs.task_type,
                    "agent_id": current_agent,
                    "model_id": model_id,
                    "turn": obs.turn_index,
                    "skills": ",".join(obs.task_skills),
                    "difficulty": obs.task_difficulty,
                    "curriculum_bucket": obs.curriculum_bucket,
                    "self_rating": round(rating, 4),
                    "success": obs.success,
                    "reward": obs.reward,
                    "task_score": rb.get("task_success", 0.0),
                    "skill_improvement": rb.get("skill_demonstration", 0.0),
                    "consistency": rb.get("learning_evidence", 0.0),
                    "skill_drop": rb.get("skill_drop_penalty", 0.0),
                    "response_text": response,
                }
                rows.append(row)
                writer.writerow(row)
                jsonl_file.write(json.dumps(row) + "\n")

                samples.append(
                    EpisodeSample(
                        agent_id=current_agent,
                        model_id=model_id,
                        prompt=prompt,
                        response=response,
                        reward_scalar=float(obs.reward),
                        reward_breakdown=dict(obs.reward_breakdown or {}),
                    )
                )
                done = obs.done
                last_obs = obs

            for agent_id, graph in last_obs.skill_snapshot.items():
                mean_level = sum(node["level"] for node in graph.values()) / max(1, len(graph))
                skill_series.setdefault(agent_id, []).append(mean_level)

    dataset_payload = [
        {
            "agent_id": s.agent_id,
            "model_id": s.model_id,
            "prompt": s.prompt,
            "response": s.response,
            "reward_scalar": s.reward_scalar,
            "reward_breakdown": s.reward_breakdown,
        }
        for s in samples
    ]
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset_out.write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")

    _generate_plots(rows, skill_series, out_dir)

    avg_reward = sum(float(r["reward"]) for r in rows) / max(1, len(rows))
    success_rate = sum(1 for r in rows if str(r["success"]).lower() == "true") / max(1, len(rows))
    summary = {
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "seed": seed,
        "mode": "trl_real_three_model_rollout",
        "model_map": model_map,
        "turn_cap": turn_cap,
        "avg_reward": round(avg_reward, 4),
        "success_rate": round(success_rate, 4),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "trl_dataset_path": str(dataset_out),
    }
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


def optimize_three_models_with_trl(
    dataset_path: Path,
    model_map: dict[str, str],
    out_dir: Path,
    train_epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    top_fraction: float,
) -> dict[str, str]:
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            "Missing TRL training dependencies. Install with: pip install trl datasets transformers torch"
        ) from exc

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not payload:
        raise RuntimeError(f"No rollout samples found at {dataset_path}")

    output_paths: dict[str, str] = {}
    for agent_id, base_model in model_map.items():
        agent_rows = [r for r in payload if r.get("agent_id") == agent_id]
        if not agent_rows:
            continue
        agent_rows.sort(key=lambda r: float(r.get("reward_scalar", 0.0)), reverse=True)
        keep_n = max(8, int(len(agent_rows) * max(0.1, min(1.0, top_fraction))))
        selected = agent_rows[:keep_n]
        texts = [
            {
                "text": (
                    f"{row.get('prompt', '').strip()}\n\n"
                    f"Response:\n{row.get('response', '').strip()}"
                )
            }
            for row in selected
        ]
        train_ds = Dataset.from_list(texts)

        model = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        agent_out = out_dir / "trl_models" / agent_id
        args = TrainingArguments(
            output_dir=str(agent_out),
            per_device_train_batch_size=max(1, batch_size),
            gradient_accumulation_steps=max(1, gradient_accumulation_steps),
            num_train_epochs=max(1, train_epochs),
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
        )
        trainer.train()
        trainer.save_model(str(agent_out))
        tokenizer.save_pretrained(str(agent_out))
        output_paths[agent_id] = str(agent_out)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Final TRL training runner for real 3-model optimization.")
    parser.add_argument("--episodes", type=int, default=120, help="Rollout episodes to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed.")
    parser.add_argument("--hf-token", type=str, required=True, help="HF token for real model inference.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="training/runs/final_run",
        help="Final run artifact directory.",
    )
    parser.add_argument(
        "--turn-cap",
        type=int,
        default=24,
        help="Upper bound on turns per episode.",
    )
    parser.add_argument(
        "--dataset-out",
        type=str,
        default="training/runs/final_run/grpo_dataset.json",
        help="GRPO-ready dataset output path.",
    )
    parser.add_argument("--model-alpha", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model-beta", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--model-gamma", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--request-gap-s", type=float, default=0.8, help="Sleep between model calls.")
    parser.add_argument("--skip-optimization", action="store_true", help="Only collect real rollouts.")
    parser.add_argument("--train-epochs", type=int, default=1, help="TRL train epochs per agent model.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="TRL learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="TRL per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="TRL gradient accumulation.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="TRL max sequence length.")
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.6,
        help="Top fraction of reward-ranked samples used for each agent optimization.",
    )
    parser.add_argument(
        "--print-trl-template",
        action="store_true",
        help="Print TRL GRPO config template for manual copy/paste.",
    )
    args = parser.parse_args()

    model_map = {
        "agent_alpha": args.model_alpha,
        "agent_beta": args.model_beta,
        "agent_gamma": args.model_gamma,
    }
    summary_path = run_real_rollouts(
        episodes=max(1, args.episodes),
        seed=args.seed,
        out_dir=Path(args.out_dir),
        turn_cap=max(3, args.turn_cap),
        dataset_out=Path(args.dataset_out),
        token=args.hf_token.strip(),
        model_map=model_map,
        request_gap_s=max(0.0, args.request_gap_s),
    )
    print(f"Real rollout complete. Summary: {summary_path}")

    optimized_paths: dict[str, str] = {}
    if not args.skip_optimization:
        optimized_paths = optimize_three_models_with_trl(
            dataset_path=Path(args.dataset_out),
            model_map=model_map,
            out_dir=Path(args.out_dir),
            train_epochs=args.train_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.max_seq_length,
            top_fraction=args.top_fraction,
        )
        print("TRL optimization complete:")
        print(json.dumps(optimized_paths, indent=2))
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        summary["optimized_model_paths"] = optimized_paths
        summary["optimization_mode"] = "trl_sft_per_agent"
        _write_json(Path(summary_path), summary)

    if args.print_trl_template:
        template = {
            "library": "trl",
            "algorithm": "GRPO",
            "model_map": model_map,
            "dataset_path": args.dataset_out,
            "max_prompt_length": 512,
            "max_completion_length": 96,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "num_train_epochs": args.train_epochs,
            "learning_rate": args.learning_rate,
            "save_steps": 200,
            "logging_steps": 20,
            "optimization_mode": "three_separate_agent_models",
            "cost_notes": "This run uses real model inference + TRL optimization (no simulated responses).",
        }
        print(json.dumps(template, indent=2))


if __name__ == "__main__":
    main()
