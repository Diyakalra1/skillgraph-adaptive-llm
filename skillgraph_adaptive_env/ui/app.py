"""Simple Streamlit UI to inspect SkillGraph training runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_rows(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "episode_logs.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _safe_json_load(text: str) -> dict:
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _build_skill_heatmap(rows: list[dict]) -> pd.DataFrame:
    """
    Build agent x skill matrix from logs.
    Cell value = success rate (0-1) for that skill when agent attempted it.
    """
    stat: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        agent = row.get("agent_id", "").strip() or "unknown_agent"
        success = 1.0 if str(row.get("success", "")).lower() == "true" else 0.0
        skills = [s.strip() for s in str(row.get("skills", "")).split(",") if s.strip()]
        for skill in skills:
            key = (agent, skill)
            slot = stat.setdefault(key, {"success_sum": 0.0, "count": 0.0})
            slot["success_sum"] += success
            slot["count"] += 1.0

    rows_out = []
    for (agent, skill), vals in stat.items():
        rate = vals["success_sum"] / max(1.0, vals["count"])
        rows_out.append({"agent": agent, "skill": skill, "success_rate": round(rate, 4)})
    return pd.DataFrame(rows_out)


def _build_interaction_matrix(rows: list[dict], last_n: int = 120) -> pd.DataFrame:
    """Build directed interaction counts from adjacent turns in an episode."""
    clipped = rows[-last_n:] if len(rows) > last_n else rows
    rows_by_ep: dict[int, list[dict]] = {}
    for r in clipped:
        ep = int(r.get("episode", 0))
        rows_by_ep.setdefault(ep, []).append(r)
    counts: dict[tuple[str, str], int] = {}
    for ep_rows in rows_by_ep.values():
        ordered = sorted(ep_rows, key=lambda x: int(x.get("turn", 0)))
        for i in range(len(ordered) - 1):
            a = ordered[i].get("agent_id", "unknown")
            b = ordered[i + 1].get("agent_id", "unknown")
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1
    return pd.DataFrame(
        [{"from_agent": k[0], "to_agent": k[1], "count": v} for k, v in counts.items()]
    )


def _build_curriculum_plan(rows: list[dict], target_agent: str) -> list[dict]:
    """Simple next-task suggestions from weakest recent skills."""
    agent_rows = [r for r in rows if r.get("agent_id") == target_agent]
    if not agent_rows:
        return []
    skill_stat: dict[str, dict[str, float]] = {}
    for r in agent_rows[-150:]:
        success = 1.0 if str(r.get("success", "")).lower() == "true" else 0.0
        for skill in [s.strip() for s in str(r.get("skills", "")).split(",") if s.strip()]:
            slot = skill_stat.setdefault(skill, {"sum": 0.0, "count": 0.0})
            slot["sum"] += success
            slot["count"] += 1.0
    weak = sorted(
        skill_stat.items(),
        key=lambda kv: kv[1]["sum"] / max(1.0, kv[1]["count"]),
    )[:5]
    templates = [
        ("collaborative", "Collaborative decomposition challenge"),
        ("competitive", "Budget negotiation game"),
        ("peer_teaching", "Teach-and-transfer mini session"),
        ("debate", "Evidence-backed policy debate"),
        ("mixed_motive", "Startup coalition scenario"),
    ]
    plan = []
    for idx, (skill, vals) in enumerate(weak, start=1):
        rate = vals["sum"] / max(1.0, vals["count"])
        t_type, title = templates[(idx - 1) % len(templates)]
        plan.append(
            {
                "slot": idx,
                "task_type": t_type,
                "task_name": title,
                "target_skill": skill,
                "current_level_proxy": round(rate * 5.0, 2),
                "target_level_proxy": round(min(5.0, (rate * 5.0) + 0.6), 2),
            }
        )
    return plan


def main() -> None:
    st.set_page_config(page_title="SkillGraph Baseline UI", layout="wide")
    st.title("SkillGraph: Adaptive Curriculum Baseline")
    st.caption("Understand training behavior with clear logs + skill graphs.")

    default_dir = Path("training/runs/latest")
    run_dir_input = st.text_input("Run directory", value=str(default_dir))
    run_dir = Path(run_dir_input)

    summary = _load_summary(run_dir)
    rows = _load_rows(run_dir)
    iter_rows = _load_jsonl(run_dir / "iteration_report.jsonl")
    final_cls = _safe_json_load((run_dir / "final_classification.json").read_text(encoding="utf-8")) if (run_dir / "final_classification.json").exists() else {}

    if not summary:
        st.warning("No run found yet. Run training first, then refresh this page.")
        st.code("python -m skillgraph_adaptive_env.training.run_training --episodes 120")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", summary.get("episodes", 0))
    c2.metric("Avg Reward", summary.get("avg_reward", 0))
    c3.metric("Success Rate", summary.get("success_rate", 0))

    st.subheader("Skill Evolution Heatmap")
    heatmap_df = _build_skill_heatmap(rows)
    if heatmap_df.empty:
        st.info("Not enough data to render heatmap yet.")
    else:
        chart = (
            alt.Chart(heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("skill:N", title="Skill"),
                y=alt.Y("agent:N", title="Agent"),
                color=alt.Color(
                    "success_rate:Q",
                    title="Success rate",
                    scale=alt.Scale(domain=[0, 1], scheme="yellowgreenblue"),
                ),
                tooltip=["agent:N", "skill:N", "success_rate:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Recent Episode Logs")
    show_n = st.slider("Rows to show", min_value=10, max_value=100, value=25, step=5)
    st.dataframe(rows[-show_n:], use_container_width=True)

    st.subheader("Live Task Arena (Replay)")
    if rows:
        episode_ids = sorted({int(r["episode"]) for r in rows})
        selected_ep = st.selectbox("Select episode", options=episode_ids, index=max(0, len(episode_ids) - 1))
        ep_rows = [r for r in rows if int(r["episode"]) == int(selected_ep)]
        ep_rows = sorted(ep_rows, key=lambda r: int(r.get("turn", 0)))
        if ep_rows:
            head = ep_rows[-1]
            st.markdown(
                f"**Task**: `{head.get('task_id','')}` | **Type**: `{head.get('task_type','')}` | "
                f"**Turns**: `{head.get('turn','0')}` | **Success**: `{head.get('success','')}`"
            )
            turn_df = pd.DataFrame(
                [
                    {
                        "turn": int(r.get("turn", 0)),
                        "agent": r.get("agent_id", ""),
                        "reward": float(r.get("reward", 0)),
                        "response": r.get("response_text", "")[:120],
                    }
                    for r in ep_rows
                ]
            )
            st.dataframe(turn_df, use_container_width=True)

            latest_public = _safe_json_load(ep_rows[-1].get("public_observation", ""))
            latest_private = _safe_json_load(ep_rows[-1].get("private_observation", ""))
            c_pub, c_priv = st.columns(2)
            with c_pub:
                st.markdown("**Public Thread**")
                for msg in latest_public.get("recent_public_messages", []):
                    st.write(f"- T{msg.get('turn', '?')} [{msg.get('agent_id', 'agent')}]: {msg.get('content', '')}")
            with c_priv:
                st.markdown("**Private Thread (Current Agent View)**")
                st.write(f"- Masking rule: {latest_private.get('task_masking_rule', 'n/a')}")
                pref = latest_private.get("visible_private_preference", "")
                if pref:
                    st.write(f"- Visible private preference: {pref}")
                for note in latest_private.get("private_notes", []):
                    st.write(f"- {note}")

    st.subheader("Curriculum Planner")
    all_agents = sorted({r.get("agent_id", "") for r in rows if r.get("agent_id", "")})
    if all_agents:
        selected_agent = st.selectbox("Plan for agent", options=all_agents, index=0)
        plan = _build_curriculum_plan(rows, selected_agent)
        if plan:
            st.dataframe(pd.DataFrame(plan), use_container_width=True)
        else:
            st.info("Not enough rows yet for curriculum suggestions.")

    st.subheader("Agent Interaction Network (Matrix View)")
    edge_df = _build_interaction_matrix(rows, last_n=200)
    if edge_df.empty:
        st.info("Not enough interaction data yet.")
    else:
        matrix_chart = (
            alt.Chart(edge_df)
            .mark_rect()
            .encode(
                x=alt.X("to_agent:N", title="To Agent"),
                y=alt.Y("from_agent:N", title="From Agent"),
                color=alt.Color("count:Q", title="Interaction Count", scale=alt.Scale(scheme="blues")),
                tooltip=["from_agent:N", "to_agent:N", "count:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(matrix_chart, use_container_width=True)

    st.subheader("Saved Graphs")
    for img_name in [
        "reward_vs_steps.png",
        "skill_evolution.png",
        "weak_to_strong_transition.png",
    ]:
        image_path = run_dir / img_name
        if image_path.exists():
            st.image(str(image_path), caption=img_name, use_container_width=True)
        else:
            st.info(f"Missing image: {img_name}")

    st.subheader("HF 7-Iteration Role Winners")
    winner_rows = [r for r in iter_rows if "role_winners" in r]
    if winner_rows:
        winner_df = pd.DataFrame(
            [
                {
                    "iteration": r.get("iteration"),
                    "planner": r.get("role_winners", {}).get("planner", ""),
                    "negotiator": r.get("role_winners", {}).get("negotiator", ""),
                    "teacher": r.get("role_winners", {}).get("teacher", ""),
                }
                for r in winner_rows
            ]
        )
        st.dataframe(winner_df, use_container_width=True)
    else:
        st.info("No iteration_report.jsonl role winner entries found.")

    st.subheader("Final Role Classification")
    if final_cls:
        winners = final_cls.get("winners", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Planner", winners.get("planner", "n/a"))
        c2.metric("Best Negotiator", winners.get("negotiator", "n/a"))
        c3.metric("Best Teacher", winners.get("teacher", "n/a"))
        st.json(final_cls)
    else:
        st.info("No final_classification.json found.")

    st.subheader("Judge vs Rubric Comparison")
    score_rows = [r for r in iter_rows if "rubric_score" in r]
    if score_rows:
        flat: list[dict] = []
        for r in score_rows:
            for role in ("planner", "negotiator", "teacher"):
                flat.append(
                    {
                        "iteration": r.get("iteration"),
                        "agent": r.get("agent_id", ""),
                        "role": role,
                        "rubric": float(r.get("rubric_score", {}).get(role, 0.0)),
                        "judge": float(r.get("judge_score", {}).get(role, 0.0)),
                    }
                )
        score_df = pd.DataFrame(flat)
        melted = score_df.melt(
            id_vars=["iteration", "agent", "role"],
            value_vars=["rubric", "judge"],
            var_name="scorer",
            value_name="score",
        )
        cmp_chart = (
            alt.Chart(melted)
            .mark_line(point=True)
            .encode(
                x=alt.X("iteration:Q"),
                y=alt.Y("mean(score):Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("scorer:N"),
                strokeDash="role:N",
                tooltip=["scorer:N", "role:N", "mean(score):Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(cmp_chart, use_container_width=True)
    else:
        st.info("No rubric/judge rows found in iteration report.")


if __name__ == "__main__":
    main()
