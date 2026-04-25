"""Run 7-iteration AMASES training with free HF models + dual scoring."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.model_runtime import HfModelRuntime
from skillgraph_adaptive_env.server.role_classifier import RoleClassifier
from skillgraph_adaptive_env.server.scoring import llm_judge_score, merge_scores, rubric_score
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import SkillgraphAdaptiveEnvironment

MODEL_MAP_DEFAULT = {
    "agent_alpha": "meta-llama/Llama-3.2-1B-Instruct",
    "agent_beta": "Qwen/Qwen2.5-1.5B-Instruct",
    "agent_gamma": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def _prompt(obs, agent_id: str, iteration: int) -> str:
    return (
        f"Iteration {iteration}/7.\n"
        f"You are {agent_id}. Task type: {obs.task_type}. Task: {obs.task_prompt}\n"
        f"Required skills: {', '.join(obs.task_skills)}\n"
        "Respond in 2 concise lines with actionable reasoning."
    )


def run(
    out_dir: Path,
    token: str,
    seed: int,
    model_map: dict[str, str],
    judge_model: str,
    request_gap_s: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = HfModelRuntime(token=token, timeout_s=120, max_retries=2)
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    classifier = RoleClassifier()

    iteration_path = out_dir / "iteration_report.jsonl"
    reliability: dict[str, dict[str, int]] = {
        aid: {"ok": 0, "failed": 0} for aid in model_map.keys()
    }

    prev_scores: dict[str, dict[str, float]] | None = None
    with iteration_path.open("w", encoding="utf-8") as f:
        for iteration in range(1, 8):
            obs = env.reset()
            # Iteration-aware curriculum override.
            anchor = min(
                env._agent_manager.agent_ids,  # noqa: SLF001
                key=lambda aid: min(s["level"] for s in env._skill_graph.snapshot()[aid].values()),  # noqa: SLF001
            )
            weakest = env._skill_graph.weakest_skills(anchor, top_n=3)  # noqa: SLF001
            weakest_level = min(env._skill_graph.snapshot()[anchor][s]["level"] for s in weakest)  # noqa: SLF001
            task, bucket = env._curriculum.choose_task_for_iteration(  # noqa: SLF001
                iteration_idx=iteration,
                weakest_skills=weakest,
                weakest_level=weakest_level,
                previous_role_scores=prev_scores,
            )
            env._current_task = dict(task)  # noqa: SLF001
            env._current_task["bucket"] = bucket  # noqa: SLF001
            env._current_team = env._agent_manager.agent_ids  # noqa: SLF001
            env._turn_index = 0  # noqa: SLF001
            obs = env._build_observation(0.0, False, {}, {a: 0.0 for a in env._current_team}, False)  # noqa: SLF001

            per_agent_scores: dict[str, dict[str, float]] = {}
            for agent_id in env._current_team:  # noqa: SLF001
                prompt = _prompt(obs, agent_id, iteration)
                model_id = model_map[agent_id]
                generated = runtime.generate(model_id, prompt)
                if request_gap_s > 0:
                    time.sleep(request_gap_s)
                if generated.ok:
                    reliability[agent_id]["ok"] += 1
                    response_text = generated.text
                else:
                    reliability[agent_id]["failed"] += 1
                    response_text = f"[DEGRADED] {generated.error}"

                rub = rubric_score(response_text)
                judge, judge_err = llm_judge_score(runtime, judge_model, obs.task_prompt, response_text)
                if request_gap_s > 0:
                    time.sleep(request_gap_s)
                merged = merge_scores(rub, judge)
                per_agent_scores[agent_id] = merged.merged

                action = SkillgraphAdaptiveAction(
                    agent_id=agent_id,
                    task_id=obs.task_id,
                    response_text=response_text,
                    self_rating=0.5,
                    merged_reward_override=merged.merged_reward,
                )
                obs = env.step(action)
                rec = {
                    "iteration": iteration,
                    "agent_id": agent_id,
                    "model_id": model_id,
                    "task_id": obs.task_id,
                    "task_type": obs.task_type,
                    "bucket": bucket,
                    "prompt": prompt,
                    "output": response_text,
                    "runtime_ok": generated.ok,
                    "runtime_error": generated.error,
                    "rubric_score": rub,
                    "judge_score": judge,
                    "judge_error": judge_err,
                    "merged_role_score": merged.merged,
                    "merged_reward": merged.merged_reward,
                    "env_reward": obs.reward,
                }
                f.write(json.dumps(rec) + "\n")

            winners = classifier.classify_iteration(per_agent_scores)
            f.write(json.dumps({"iteration": iteration, "role_winners": winners}) + "\n")
            prev_scores = per_agent_scores
            print(
                f"[ITER {iteration}] task={obs.task_id} type={obs.task_type} "
                f"winners={winners}"
            )

    final = classifier.final_classification(iterations=7)
    final["reliability"] = reliability
    final["timestamp"] = datetime.now().isoformat()
    final["judge_model"] = judge_model
    final["model_map"] = model_map
    final_path = out_dir / "final_classification.json"
    final_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return {"iteration_report": str(iteration_path), "final_classification": str(final_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 7-iteration free-HF AMASES loop.")
    parser.add_argument("--out-dir", type=str, default="training/runs/hf_7iter")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--request-gap-s", type=float, default=2.0, help="Sleep between HF requests.")
    parser.add_argument("--model-alpha", type=str, default=MODEL_MAP_DEFAULT["agent_alpha"])
    parser.add_argument("--model-beta", type=str, default=MODEL_MAP_DEFAULT["agent_beta"])
    parser.add_argument("--model-gamma", type=str, default=MODEL_MAP_DEFAULT["agent_gamma"])
    args = parser.parse_args()

    token = args.hf_token.strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing HF token. Use --hf-token or set HF_TOKEN.")

    model_map = {
        "agent_alpha": args.model_alpha,
        "agent_beta": args.model_beta,
        "agent_gamma": args.model_gamma,
    }
    results = run(
        out_dir=Path(args.out_dir),
        token=token,
        seed=args.seed,
        model_map=model_map,
        judge_model=args.judge_model,
        request_gap_s=max(0.0, args.request_gap_s),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
