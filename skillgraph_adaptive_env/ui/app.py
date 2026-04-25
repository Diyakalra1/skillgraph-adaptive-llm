"""Simple Streamlit UI to inspect SkillGraph training runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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


def main() -> None:
    st.set_page_config(page_title="SkillGraph Baseline UI", layout="wide")
    st.title("SkillGraph: Adaptive Curriculum Baseline")
    st.caption("Understand training behavior with clear logs + skill graphs.")

    default_dir = Path("training/runs/latest")
    run_dir_input = st.text_input("Run directory", value=str(default_dir))
    run_dir = Path(run_dir_input)

    summary = _load_summary(run_dir)
    rows = _load_rows(run_dir)

    if not summary:
        st.warning("No run found yet. Run training first, then refresh this page.")
        st.code("python -m skillgraph_adaptive_env.training.run_training --episodes 120")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", summary.get("episodes", 0))
    c2.metric("Avg Reward", summary.get("avg_reward", 0))
    c3.metric("Success Rate", summary.get("success_rate", 0))

    st.subheader("Recent Episode Logs")
    show_n = st.slider("Rows to show", min_value=10, max_value=100, value=25, step=5)
    st.dataframe(rows[-show_n:], use_container_width=True)

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


if __name__ == "__main__":
    main()
