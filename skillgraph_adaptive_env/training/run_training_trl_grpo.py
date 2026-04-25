"""TRL-based GRPO training entrypoint for AMASES.

This script is intentionally manual-run only. It never auto-launches training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.agent_manager import AgentManager
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import SkillgraphAdaptiveEnvironment


@dataclass
class EpisodeSample:
    prompt: str
    response: str
    reward_scalar: float
    reward_breakdown: dict


def build_dataset(episodes: int, seed: int) -> list[EpisodeSample]:
    """Create prompt/response/reward samples from environment rollouts.

    This is lightweight and deterministic. It does not call remote models.
    """
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    agents = AgentManager(seed=seed)
    samples: list[EpisodeSample] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        guard = 0
        while not done and guard < 24:
            guard += 1
            current_agent = obs.current_agent_id or (obs.team_agent_ids[0] if obs.team_agent_ids else "agent_alpha")
            rating = 0.5
            response = agents.simulated_response(
                agent_id=current_agent,
                prompt=obs.task_prompt,
                difficulty=obs.task_difficulty,
                rating=rating,
            )
            action = SkillgraphAdaptiveAction(
                agent_id=current_agent,
                task_id=obs.task_id,
                response_text=response,
                self_rating=rating,
            )
            obs = env.step(action)
            prompt = (
                f"TaskType={obs.task_type}\n"
                f"Task={obs.task_prompt}\n"
                f"Skills={','.join(obs.task_skills)}\n"
                f"Turn={obs.turn_index}/{obs.max_turns}\n"
            )
            samples.append(
                EpisodeSample(
                    prompt=prompt,
                    response=response,
                    reward_scalar=float(obs.reward),
                    reward_breakdown=dict(obs.reward_breakdown),
                )
            )
            done = obs.done
    return samples


def save_dataset(samples: list[EpisodeSample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "prompt": s.prompt,
            "response": s.response,
            "reward_scalar": s.reward_scalar,
            "reward_breakdown": s.reward_breakdown,
        }
        for s in samples
    ]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AMASES GRPO-ready dataset for TRL.")
    parser.add_argument("--episodes", type=int, default=40, help="Rollout episodes to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed.")
    parser.add_argument(
        "--out",
        type=str,
        default="training/runs/grpo/grpo_dataset.json",
        help="Dataset output path.",
    )
    parser.add_argument(
        "--print-trl-template",
        action="store_true",
        help="Print TRL GRPO config template for manual copy/paste.",
    )
    args = parser.parse_args()

    samples = build_dataset(episodes=args.episodes, seed=args.seed)
    save_dataset(samples, Path(args.out))
    print(f"Generated {len(samples)} samples at {args.out}")

    if args.print_trl_template:
        template = {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "algorithm": "GRPO",
            "max_prompt_length": 512,
            "max_completion_length": 96,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 1,
            "learning_rate": 2e-5,
            "save_steps": 200,
            "logging_steps": 20,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "cost_notes": "Start with <=40 episodes and short completions for low HF credit burn.",
        }
        print(json.dumps(template, indent=2))


if __name__ == "__main__":
    main()
