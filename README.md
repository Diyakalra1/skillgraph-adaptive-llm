# skillgraph-adaptive-llm

Baseline implementation of **SkillGraph: Self-Aware LLM with Adaptive Curriculum**.

Main entrypoints:
- `skillgraph_adaptive_env/training/run_training.py` for adaptive training simulation + graph outputs
- `skillgraph_adaptive_env/ui/app.py` for a simple Streamlit UI
- `skillgraph_adaptive_env/server/skillgraph_adaptive_env_environment.py` for OpenEnv-compatible environment logic
