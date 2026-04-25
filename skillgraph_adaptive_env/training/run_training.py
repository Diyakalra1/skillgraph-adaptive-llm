"""Simulation-driven training loop for SkillGraph adaptive curriculum environment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.agent_manager import AgentManager
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import (
    SkillgraphAdaptiveEnvironment,
)


def _policy_self_rating(skill_snapshot: dict, task_skills: list[str], difficulty: float) -> float:
    def _safe_skill(skill_name: str) -> tuple[float, float]:
        node = skill_snapshot.get(skill_name, {})
        return float(node.get("level", 2.5)), float(node.get("confidence", 0.2))

    vals = [_safe_skill(s) for s in task_skills] if task_skills else [(2.5, 0.2)]
    skill_avg = sum(v[0] for v in vals) / max(1, len(vals))
    confidence_avg = sum(v[1] for v in vals) / max(1, len(vals))
    rating = (0.6 * skill_avg) + (0.4 * confidence_avg) - (0.35 * difficulty) + 0.25
    return max(0.0, min(1.0, rating))


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train(episodes: int, seed: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    agents = AgentManager(seed=seed)

    csv_path = out_dir / "episode_logs.csv"
    jsonl_path = out_dir / "episode_logs.jsonl"
    summary_path = out_dir / "summary.json"

    rows: list[dict] = []
    skill_series: dict[str, list[float]] = {}

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "task_id",
                "task_type",
                "target_skill",
                "is_diagnostic",
                "is_verification",
                "agent_id",
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
                "public_observation",
                "private_observation",
            ],
        )
        writer.writeheader()

        for ep in range(1, episodes + 1):
            obs = env.reset()
            done = False
            turn_guard = 0
            last_obs = obs
            while not done and turn_guard < 24:
                turn_guard += 1
                current_agent = obs.current_agent_id or (obs.team_agent_ids[0] if obs.team_agent_ids else "agent_alpha")
                rating = _policy_self_rating(
                    obs.skill_snapshot[current_agent], obs.task_skills, obs.task_difficulty
                )
                response_text = agents.simulated_response(
                    agent_id=current_agent,
                    prompt=obs.task_prompt,
                    difficulty=obs.task_difficulty,
                    rating=rating + (rng.random() * 0.02),
                )
                private_notes = obs.private_observation.get("private_notes", [])
                if private_notes:
                    response_text = f"{response_text} | note:{private_notes[-1][:60]}"
                action = SkillgraphAdaptiveAction(
                    agent_id=current_agent,
                    task_id=obs.task_id,
                    response_text=response_text,
                    self_rating=rating,
                )
                obs = env.step(action)
                reward_breakdown = obs.reward_breakdown
                row = {
                    "episode": ep,
                    "task_id": obs.task_id,
                    "task_type": obs.task_type,
                    "target_skill": (obs.metadata or {}).get("target_skill", ""),
                    "is_diagnostic": (obs.metadata or {}).get("is_diagnostic", False),
                    "is_verification": (obs.metadata or {}).get("is_verification", False),
                    "agent_id": current_agent,
                    "turn": obs.turn_index,
                    "skills": ",".join(obs.task_skills),
                    "difficulty": obs.task_difficulty,
                    "curriculum_bucket": obs.curriculum_bucket,
                    "self_rating": round(rating, 4),
                    "success": obs.success,
                    "reward": obs.reward,
                    "task_score": reward_breakdown.get("task_success", 0.0),
                    "skill_improvement": reward_breakdown.get("skill_demonstration", 0.0),
                    "consistency": reward_breakdown.get("learning_evidence", 0.0),
                    "skill_drop": reward_breakdown.get("skill_drop_penalty", 0.0),
                    "response_text": response_text,
                    "public_observation": json.dumps(obs.public_observation, ensure_ascii=True),
                    "private_observation": json.dumps(obs.private_observation, ensure_ascii=True),
                }
                rows.append(row)
                writer.writerow(row)
                jsonl_file.write(json.dumps(row) + "\n")
                done = obs.done
                last_obs = obs

            for agent_id, graph in last_obs.skill_snapshot.items():
                mean_level = sum(node["level"] for node in graph.values()) / max(1, len(graph))
                skill_series.setdefault(agent_id, []).append(mean_level)

            print(
                f"[EP {ep:03d}] task={last_obs.task_id:24s} type={last_obs.task_type:13s} "
                f"team={','.join(last_obs.team_agent_ids)} success={str(last_obs.success):5s} "
                f"turns={last_obs.turn_index:>2d} reward={last_obs.reward:>6.3f} "
                f"bucket={last_obs.curriculum_bucket}"
            )

    _generate_plots(rows, skill_series, out_dir)

    avg_reward = sum(r["reward"] for r in rows) / max(1, len(rows))
    success_rate = sum(1 for r in rows if r["success"]) / max(1, len(rows))
    summary = {
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "seed": seed,
        "mode": "simulation",
        "avg_reward": round(avg_reward, 4),
        "success_rate": round(success_rate, 4),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    _write_json(summary_path, summary)
    return summary_path


def _generate_plots(rows: list[dict], skill_series: dict[str, list[float]], out_dir: Path) -> None:
    episodes = [r["episode"] for r in rows]
    rewards = [r["reward"] for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, linewidth=1.6, marker="o", markersize=2.6, alpha=0.9)
    plt.title("Reward vs Training Steps")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    if rewards:
        ymin, ymax = min(rewards), max(rewards)
        if abs(ymax - ymin) < 0.15:
            pad = 0.08
            plt.ylim(ymin - pad, ymax + pad)
        low_i, high_i = rewards.index(ymin), rewards.index(ymax)
        plt.annotate(f"min={ymin:.3f}", (episodes[low_i], ymin), xytext=(4, -12), textcoords="offset points", fontsize=8)
        plt.annotate(f"max={ymax:.3f}", (episodes[high_i], ymax), xytext=(4, 8), textcoords="offset points", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_steps.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    all_values = []
    for skill, series in skill_series.items():
        if series:
            all_values.extend(series)
        plt.plot(
            range(1, len(series) + 1),
            series,
            label=skill,
            linewidth=2.0,
            marker="o",
            markersize=2.3,
            alpha=0.9,
        )
    plt.title("Skill Graph Evolution (Skill Levels)")
    plt.xlabel("Episode")
    plt.ylabel("Skill level")
    if all_values:
        vmin, vmax = min(all_values), max(all_values)
        if abs(vmax - vmin) < 0.12:
            pad = 0.06
            plt.ylim(max(0.0, vmin - pad), min(1.02, vmax + pad))
        else:
            plt.ylim(0, 1.02)
    else:
        plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "skill_evolution.png", dpi=160)
    plt.close()

    weakest_skill = min(skill_series.items(), key=lambda item: item[1][0])[0]
    transition = skill_series[weakest_skill]
    plt.figure(figsize=(10, 4))
    x = list(range(1, len(transition) + 1))
    plt.plot(x, transition, color="tab:red", linewidth=2.2, marker="o", markersize=2.8)
    plt.title(f"Weak to Strong Transition: {weakest_skill}")
    plt.xlabel("Episode")
    plt.ylabel("Skill level")
    if transition:
        tmin, tmax = min(transition), max(transition)
        if abs(tmax - tmin) < 0.10:
            pad = 0.05
            plt.ylim(max(0.0, tmin - pad), min(1.02, tmax + pad))
        else:
            plt.ylim(0, 1.02)
        plt.annotate(f"start={transition[0]:.3f}", (x[0], transition[0]), xytext=(4, 8), textcoords="offset points", fontsize=8)
        plt.annotate(f"end={transition[-1]:.3f}", (x[-1], transition[-1]), xytext=(4, -12), textcoords="offset points", fontsize=8)
    else:
        plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "weak_to_strong_transition.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline SkillGraph adaptive curriculum loop.")
    parser.add_argument("--episodes", type=int, default=120, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic runs.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="training/runs/latest",
        help="Output directory for logs and graphs.",
    )
    args = parser.parse_args()
    output_summary = train(
        episodes=args.episodes,
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )
    print(f"\nTraining complete. Summary: {output_summary}")


if __name__ == "__main__":
    main()
