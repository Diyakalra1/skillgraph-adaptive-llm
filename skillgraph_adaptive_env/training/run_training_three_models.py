"""Train AMASES with three real HF models mapped to three agents."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient

from skillgraph_adaptive_env import SkillgraphAdaptiveAction
from skillgraph_adaptive_env.server.skillgraph_adaptive_env_environment import (
    SkillgraphAdaptiveEnvironment,
)


AGENT_MODEL_MAP_DEFAULT = {
    "agent_alpha": "meta-llama/Llama-3.2-1B-Instruct",
    "agent_beta": "Qwen/Qwen2.5-1.5B-Instruct",
    "agent_gamma": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _policy_self_rating(skill_snapshot: dict, task_skills: list[str], difficulty: float) -> float:
    def _safe_skill(skill_name: str) -> tuple[float, float]:
        node = skill_snapshot.get(skill_name, {})
        return float(node.get("level", 2.5)), float(node.get("confidence", 0.2))

    vals = [_safe_skill(s) for s in task_skills] if task_skills else [(2.5, 0.2)]
    skill_avg = sum(v[0] for v in vals) / max(1, len(vals))
    confidence_avg = sum(v[1] for v in vals) / max(1, len(vals))
    rating = (0.6 * (skill_avg / 5.0)) + (0.4 * confidence_avg) - (0.12 * difficulty) + 0.20
    return max(0.0, min(1.0, rating))


def _make_prompt(obs, agent_id: str) -> str:
    private = obs.private_observation or {}
    public = obs.public_observation or {}
    private_notes = private.get("private_notes", [])
    public_msgs = public.get("recent_public_messages", [])
    public_text = "\n".join(
        f"- T{m.get('turn', '?')} {m.get('agent_id', 'agent')}: {m.get('content', '')}"
        for m in public_msgs[-4:]
    )
    return (
        f"You are {agent_id} in a multi-agent task.\n"
        f"Task type: {obs.task_type}\n"
        f"Task: {obs.task_prompt}\n"
        f"Skills tested: {', '.join(obs.task_skills)}\n"
        f"Current turn: {obs.turn_index}/{obs.max_turns}\n"
        f"Masking rule: {private.get('task_masking_rule', 'n/a')}\n"
        f"Private preference: {private.get('visible_private_preference', '')}\n"
        f"Private notes: {' | '.join(private_notes[-2:])}\n"
        f"Recent public thread:\n{public_text}\n\n"
        "Respond in exactly 3 short bullet lines:\n"
        "- Proposal: one concrete offer/plan.\n"
        "- Constraint/Evidence: reference one constraint or prior message evidence.\n"
        "- Next move: one counter-offer, question, or action step with rationale."
    )


def _generate_with_model(client: InferenceClient, model_id: str, prompt: str, max_tokens: int) -> tuple[str, bool, str]:
    def _first_error_line(exc: Exception) -> str:
        text = str(exc).strip()
        return text.splitlines()[0] if text else exc.__class__.__name__

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip(), True, ""
    except Exception as chat_exc:
        try:
            # Fallback for providers that only expose text generation.
            text = client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=max_tokens,
                temperature=0.2,
                return_full_text=False,
            )
            return str(text).strip(), True, ""
        except Exception as text_exc:
            return (
                "",
                False,
                f"model={model_id} chat_error={_first_error_line(chat_exc)} text_error={_first_error_line(text_exc)}",
            )


def _generate_plots(rows: list[dict], skill_series: dict[str, list[float]], out_dir: Path) -> None:
    episodes = [r["episode"] for r in rows]
    rewards = [r["reward"] for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, linewidth=1.6, marker="o", markersize=2.6, alpha=0.9)
    plt.title("Reward vs Training Steps")
    plt.xlabel("Episode-Turn")
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
    for agent_id, series in skill_series.items():
        if series:
            all_values.extend(series)
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
    if all_values:
        vmin, vmax = min(all_values), max(all_values)
        if abs(vmax - vmin) < 0.20:
            pad = 0.12
            plt.ylim(max(0.0, vmin - pad), min(5.1, vmax + pad))
        else:
            plt.ylim(0, 5.1)
    else:
        plt.ylim(0, 5.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "skill_evolution.png", dpi=160)
    plt.close()

    weakest_agent = min(skill_series.items(), key=lambda item: item[1][0])[0]
    trajectory = skill_series[weakest_agent]
    plt.figure(figsize=(10, 4))
    x = list(range(1, len(trajectory) + 1))
    plt.plot(x, trajectory, color="tab:red", linewidth=2.2, marker="o", markersize=2.8)
    plt.title(f"Weak to Strong Transition: {weakest_agent}")
    plt.xlabel("Episode")
    plt.ylabel("Mean skill level (0-5)")
    if trajectory:
        tmin, tmax = min(trajectory), max(trajectory)
        if abs(tmax - tmin) < 0.20:
            pad = 0.12
            plt.ylim(max(0.0, tmin - pad), min(5.1, tmax + pad))
        else:
            plt.ylim(0, 5.1)
        plt.annotate(f"start={trajectory[0]:.3f}", (x[0], trajectory[0]), xytext=(4, 8), textcoords="offset points", fontsize=8)
        plt.annotate(f"end={trajectory[-1]:.3f}", (x[-1], trajectory[-1]), xytext=(4, -12), textcoords="offset points", fontsize=8)
    else:
        plt.ylim(0, 5.1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "weak_to_strong_transition.png", dpi=160)
    plt.close()


def train(
    episodes: int,
    seed: int,
    out_dir: Path,
    token: str,
    model_map: dict[str, str],
    request_gap_s: float,
    max_tokens: int,
    turn_cap: int = 24,
    max_api_calls: int = 700,
    max_fallback_rate: float = 0.35,
    min_calls_before_abort: int = 18,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = SkillgraphAdaptiveEnvironment(seed=seed)
    client = InferenceClient(api_key=token, timeout=120)

    csv_path = out_dir / "episode_logs.csv"
    jsonl_path = out_dir / "episode_logs.jsonl"
    summary_path = out_dir / "summary.json"
    rows: list[dict] = []
    skill_series: dict[str, list[float]] = {}
    api_calls = 0
    fallback_calls = 0
    aborted_early = False
    abort_reason = ""

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
            turn_guard = 0
            last_obs = obs
            while not done and turn_guard < max(3, turn_cap):
                turn_guard += 1
                current_agent = obs.current_agent_id or (obs.team_agent_ids[0] if obs.team_agent_ids else "agent_alpha")
                model_id = model_map[current_agent]
                rating = _policy_self_rating(
                    obs.skill_snapshot[current_agent], obs.task_skills, obs.task_difficulty
                )
                prompt = _make_prompt(obs, current_agent)
                response_text, ok, model_error = _generate_with_model(
                    client, model_id, prompt, max_tokens=max_tokens
                )
                api_calls += 1
                if not ok:
                    fallback_calls += 1
                    aborted_early = True
                    abort_reason = (
                        "strict_no_fallback:model_generation_failed "
                        f"({model_error})"
                    )
                    break
                if request_gap_s > 0:
                    time.sleep(request_gap_s)
                action = SkillgraphAdaptiveAction(
                    agent_id=current_agent,
                    task_id=obs.task_id,
                    response_text=response_text,
                    self_rating=rating,
                )
                obs = env.step(action)
                rb = obs.reward_breakdown
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
                    "response_text": response_text,
                }
                rows.append(row)
                writer.writerow(row)
                jsonl_file.write(json.dumps(row) + "\n")
                done = obs.done
                last_obs = obs

                if api_calls >= max(1, max_api_calls):
                    aborted_early = True
                    abort_reason = (
                        f"budget_guard:max_api_calls reached ({api_calls}/{max_api_calls})"
                    )
                    done = True
                    break
                if api_calls >= max(1, min_calls_before_abort):
                    fallback_rate = fallback_calls / max(1, api_calls)
                    if fallback_rate > max_fallback_rate:
                        aborted_early = True
                        abort_reason = (
                            "quality_guard:fallback_rate "
                            f"{fallback_rate:.3f} exceeded {max_fallback_rate:.3f}"
                        )
                        done = True
                        break

            for agent_id, graph in last_obs.skill_snapshot.items():
                mean_level = sum(node["level"] for node in graph.values()) / max(1, len(graph))
                skill_series.setdefault(agent_id, []).append(mean_level)

            print(
                f"[EP {ep:03d}] task={last_obs.task_id:24s} type={last_obs.task_type:13s} "
                f"team={','.join(last_obs.team_agent_ids)} success={str(last_obs.success):5s} "
                f"turns={last_obs.turn_index:>2d} reward={last_obs.reward:>6.3f}"
            )
            if aborted_early:
                print(f"[STOP EARLY] {abort_reason}")
                break

    _generate_plots(rows, skill_series, out_dir)

    avg_reward = sum(r["reward"] for r in rows) / max(1, len(rows))
    success_rate = sum(1 for r in rows if r["success"]) / max(1, len(rows))
    summary = {
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "seed": seed,
        "mode": "hf_three_models",
        "model_map": model_map,
        "turn_cap": max(3, turn_cap),
        "api_calls": api_calls,
        "fallback_calls": fallback_calls,
        "fallback_rate": round(fallback_calls / max(1, api_calls), 4),
        "max_api_calls": max(1, max_api_calls),
        "max_fallback_rate": max(0.0, min(1.0, max_fallback_rate)),
        "min_calls_before_abort": max(1, min_calls_before_abort),
        "aborted_early": aborted_early,
        "abort_reason": abort_reason,
        "avg_reward": round(avg_reward, 4),
        "success_rate": round(success_rate, 4),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    _write_json(summary_path, summary)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AMASES with three Hugging Face models.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default="training/runs/hf_three_models", help="Output directory.")
    parser.add_argument("--hf-token", type=str, default="", help="HF token (or set HF_TOKEN env var).")
    parser.add_argument("--model-alpha", type=str, default=AGENT_MODEL_MAP_DEFAULT["agent_alpha"])
    parser.add_argument("--model-beta", type=str, default=AGENT_MODEL_MAP_DEFAULT["agent_beta"])
    parser.add_argument("--model-gamma", type=str, default=AGENT_MODEL_MAP_DEFAULT["agent_gamma"])
    parser.add_argument("--request-gap-s", type=float, default=2.0, help="Sleep between HF requests.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max tokens per model response.")
    parser.add_argument("--turn-cap", type=int, default=24, help="Max turns per episode for budget control.")
    parser.add_argument("--max-api-calls", type=int, default=700, help="Hard cap on provider calls.")
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=0.35,
        help="Abort if fallback response share exceeds this threshold after warmup.",
    )
    parser.add_argument(
        "--min-calls-before-abort",
        type=int,
        default=18,
        help="Minimum calls before fallback-rate early-stop can trigger.",
    )
    args = parser.parse_args()

    token = args.hf_token.strip() or os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing HF token. Use --hf-token or set HF_TOKEN.")

    model_map = {
        "agent_alpha": args.model_alpha,
        "agent_beta": args.model_beta,
        "agent_gamma": args.model_gamma,
    }
    summary_path = train(
        episodes=args.episodes,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        token=token,
        model_map=model_map,
        request_gap_s=max(0.0, args.request_gap_s),
        max_tokens=max(16, args.max_tokens),
        turn_cap=max(3, args.turn_cap),
        max_api_calls=max(1, args.max_api_calls),
        max_fallback_rate=max(0.0, min(1.0, args.max_fallback_rate)),
        min_calls_before_abort=max(1, args.min_calls_before_abort),
    )
    print(f"\nTraining complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
