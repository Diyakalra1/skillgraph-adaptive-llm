---
title: SkillGraph Adaptive Env
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
base_path: /web
python_version: "3.11"
pinned: false
tags:
  - openenv
short_description: Adaptive multi-agent skill evolution OpenEnv environment
---

# AMASES: Adaptive Multi-Agent Skill Evolution System

AMASES is a multi-agent OpenEnv environment focused on adaptive skill learning across three agents, five task families, and deterministic rubric-based rewards.

Deployed environment:
- [https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env](https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env)

## Hugging Face Space playground

When the Space is **Running**, the **App** tab shows the OpenEnv web UI:

1. Click **Reset** to start a new episode (task + agents).
2. Fill action fields (`agent_id`, `response_text`, `self_rating`, etc.) and click **Step**.
3. Use **Get state** to inspect the current environment.

The left panel has Python snippets to connect from Colab or a local client via `SkillgraphAdaptiveEnv.from_env("jeeya-ahuja05/skill-graph-adaptive-env")`.

If the Space stays on **Starting**, confirm `app_port: 8000` in this README and that the container listens on `0.0.0.0:8000` (not Gradio’s default `7860`).

## Agent Setup

- `agent_alpha` (planner)
- `agent_beta` (debater)
- `agent_gamma` (integrator)

## Task Inventory

AMASES uses a fixed 15-task curriculum:

- Collaborative: easy, medium, hard
- Competitive: easy, medium, hard
- Mixed motive: easy, medium, hard
- Peer teaching: easy, medium, hard
- Debate: easy, medium, hard

Each task has fixed logic and schema fields: `type`, `agents_needed`, `skills_tested`, `difficulty`, `max_turns`, `reward_mode`, rules, and rubric settings. Runtime variation changes scenario surface text only.

## Reward Model

Per-turn score is decomposed into weighted components:

- `task_success` (30%)
- `skill_demo` (25%)
- `collab_quality` (20%)
- `learning_evidence` (15%)
- `meta_cognition` (10%)

Penalty hooks include:

- instant-agreement exploitation
- repeated/progressless proposals
- ignoring context/history
- timeout/empty/incoherent responses
- inflated self-assessment behavior

The environment computes both:
- merged scalar reward (for logging and comparison),
- structured breakdown values (for diagnostics and skill updates).

## Curriculum Engine Behavior

`server/curriculum_engine.py` drives adaptive progression:

- cold-start diagnostics identify weak initial skills,
- weak skills are assigned easier targeted tasks,
- improving skills move to harder tasks,
- verification checks run periodically to validate transfer and stability.

This yields per-agent divergence and measurable long-horizon development.

## Main Architecture

- `server/skillgraph_adaptive_env_environment.py`: environment state machine and turn orchestration
- `server/task_library.py`: task templates and constraints
- `server/skill_graph.py`: persistent per-agent skill state
- `server/scoring.py`: deterministic reward decomposition and penalties
- `server/curriculum_engine.py`: adaptive task selection
- `training/run_training_three_models.py`: three-model training/evaluation pipeline
- `ui/app.py`: Streamlit log and graph viewer

## TRL GRPO Training (RL Fine-Tuning)

This is the reinforcement-learning training path (TRL GRPO) using rewards from your environment:

```bash
pip install -e ".[trl]"
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 40 \
  --seed 7 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --out-dir training/runs/trl_grpo
```

Artifacts:
- `grpo_dataset.json` (prompt + `env_reward` from environment)
- `summary.json`
- `checkpoints/final/` (fine-tuned model)

Colab-friendly pins: `training/requirements-trl.txt`

## Three-Model Training (HF Inference Evaluation)

Run from repository root:

```bash
pip install -e skillgraph_adaptive_env
python -m skillgraph_adaptive_env.training.run_training_three_models \
  --episodes 3 \
  --seed 7 \
  --hf-token <YOUR_HF_TOKEN> \
  --out-dir training/runs/hf_three_models
```

## Output Artifacts

Generated under `training/runs/hf_three_models/`:

- `episode_logs.csv`
- `episode_logs.jsonl`
- `summary.json`
- `reward_vs_steps.png`
- `skill_evolution.png`
- `weak_to_strong_transition.png`

## Log Schema Reference (Three-Model Run)

Each row in `episode_logs.csv` (and each JSON object in `episode_logs.jsonl`) includes:

- run context: `episode`, `turn`, `task_id`, `task_type`, `curriculum_bucket`
- model context: `agent_id`, `model_id`, `response_text`
- task context: `skills`, `difficulty`, `self_rating`
- outcomes: `success`, `reward`
- reward decomposition: `task_score`, `skill_improvement`, `consistency`, `skill_drop`

Example JSONL row shape:

```json
{
  "episode": 1,
  "task_id": "collaborative_medium",
  "task_type": "collaborative",
  "agent_id": "agent_alpha",
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "turn": 2,
  "skills": "collaboration,planning",
  "difficulty": 3.0,
  "curriculum_bucket": "cold_start_diagnostic",
  "self_rating": 0.57,
  "success": false,
  "reward": 0.0915,
  "task_score": 0.0,
  "skill_improvement": 0.12,
  "consistency": 0.06,
  "skill_drop": 0.0,
  "response_text": "..."
}
```

## Graph Reference

- `reward_vs_steps.png`: reward trend by episode-turn; useful for stability and variance checks.
- `skill_evolution.png`: mean skill trajectory per agent; shows relative learning divergence.
- `weak_to_strong_transition.png`: tracks weakest starting agent through training progression.

## Demo Artifacts (Committed Run)

Repository includes example outputs at `training/runs/final_run/` (logs, summary, and the three plots above).

## UI Dashboard

Path: `ui/app.py`

```bash
pip install -e .
streamlit run skillgraph_adaptive_env/ui/app.py
```

Use the run-directory selector in the UI to load `training/runs/final_run` or any output folder from `run_training_three_models.py`.

## Blog Post

Draft placeholder: `blogpost.md` (edit in this folder).

## Integration Notes

- Core typed interfaces: `models.py`
- Environment entrypoint: `server/skillgraph_adaptive_env_environment.py`
- Server app wiring: `server/app.py`
- Client adapter: `client.py`
- OpenEnv integration guide: `INTEGRATION_GUIDE.md`
