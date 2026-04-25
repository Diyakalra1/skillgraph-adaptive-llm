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
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import (
    SkillgraphAdaptiveEnvironment,
)


def _policy_self_rating(skill_snapshot: dict, task_skills: list[str], difficulty: float) -> float:
    skill_avg = sum(skill_snapshot[s]["level"] for s in task_skills) / max(1, len(task_skills))
    confidence_avg = sum(skill_snapshot[s]["confidence"] for s in task_skills) / max(1, len(task_skills))
    rating = (0.6 * skill_avg) + (0.4 * confidence_avg) - (0.35 * difficulty) + 0.25
    return max(0.0, min(1.0, rating))


def _simulated_response(task_id: str, prompt: str, rating: float, rng: random.Random) -> str:
    """
    Produce a deterministic-ish simulated answer with controllable quality.
    Higher self-rating means more likely to include target-like keywords.
    """
    if "math_easy_1" in task_id:
        return "43" if rating > 0.45 else "44"
    if "math_mid_1" in task_id:
        return (
            "0.75 is greater than 0.66, so 3/4 is greater."
            if rating > 0.5
            else "I think they are close."
        )
    if "reason_easy_1" in task_id:
        return "apple is the odd one because others are numbers." if rating > 0.45 else "apple is odd."
    if "reason_hard_1" in task_id:
        return "We cannot conclude some robots are smart with certainty." if rating > 0.65 else "some robots are smart."
    if "code_easy_1" in task_id:
        return "sum_val=0\nfor x in nums:\n    sum_val += x" if rating > 0.45 else "sum(nums)"
    if "debug_mid_1" in task_id:
        return (
            "Use range(len(arr)) to avoid IndexError."
            if rating > 0.55
            else "Change the loop a bit."
        )
    if "opt_hard_1" in task_id:
        return "Use a hash set for O(n) lookups instead of O(n^2)." if rating > 0.7 else "Try optimization."
    if "comm_easy_1" in task_id:
        return "Please consider this reminder. Thank you." if rating > 0.4 else "Reminder."
    if "persuasion_mid_1" in task_id:
        return "A trial can improve productivity and has clear benefit." if rating > 0.55 else "Please allow Fridays."
    if "strategy_mid_1" in task_id:
        return "Choose A because budget is tight and risk is lower." if rating > 0.55 else "Choose A."
    return prompt[:60] + ("..." if rng.random() > rating else "")


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train(episodes: int, seed: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    env = SkillgraphAdaptiveEnvironment(seed=seed)

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
            ],
        )
        writer.writeheader()

        for ep in range(1, episodes + 1):
            reset_obs = env.reset()
            rating = _policy_self_rating(
                reset_obs.skill_snapshot, reset_obs.task_skills, reset_obs.task_difficulty
            )
            action = SkillgraphAdaptiveAction(
                task_id=reset_obs.task_id,
                response_text=_simulated_response(
                    task_id=reset_obs.task_id,
                    prompt=reset_obs.task_prompt,
                    rating=rating,
                    rng=rng,
                ),
                self_rating=rating,
            )
            result_obs = env.step(action)
            reward_breakdown = result_obs.reward_breakdown

            row = {
                "episode": ep,
                "task_id": result_obs.task_id,
                "skills": ",".join(result_obs.task_skills),
                "difficulty": result_obs.task_difficulty,
                "curriculum_bucket": reset_obs.curriculum_bucket,
                "self_rating": round(rating, 4),
                "success": result_obs.success,
                "reward": result_obs.reward,
                "task_score": reward_breakdown.get("task_score", 0.0),
                "skill_improvement": reward_breakdown.get("skill_improvement", 0.0),
                "consistency": reward_breakdown.get("consistency", 0.0),
                "skill_drop": reward_breakdown.get("skill_drop", 0.0),
            }
            rows.append(row)
            writer.writerow(row)
            jsonl_file.write(json.dumps(row) + "\n")

            for skill, state in result_obs.skill_snapshot.items():
                skill_series.setdefault(skill, []).append(state["level"])

            print(
                f"[EP {ep:03d}] task={result_obs.task_id:16s} bucket={reset_obs.curriculum_bucket:6s} "
                f"success={str(result_obs.success):5s} reward={result_obs.reward:>6.3f} "
                f"rating={rating:>5.2f} skills={','.join(result_obs.task_skills)} "
                f"resp='{(result_obs.metadata.get('response_text', '') or '')[:80]}'"
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
    plt.plot(episodes, rewards, linewidth=1.4)
    plt.title("Reward vs Training Steps")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_steps.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    for skill, series in skill_series.items():
        plt.plot(range(1, len(series) + 1), series, label=skill, linewidth=1.2)
    plt.title("Skill Graph Evolution (Skill Levels)")
    plt.xlabel("Episode")
    plt.ylabel("Skill level")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "skill_evolution.png", dpi=160)
    plt.close()

    weakest_skill = min(skill_series.items(), key=lambda item: item[1][0])[0]
    transition = skill_series[weakest_skill]
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(transition) + 1), transition, color="tab:red", linewidth=2.0)
    plt.title(f"Weak to Strong Transition: {weakest_skill}")
    plt.xlabel("Episode")
    plt.ylabel("Skill level")
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
