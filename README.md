# skillgraph-adaptive-llm

AMASES (Adaptive Multi-Agent Skill Evolution System) is an OpenEnv-based multi-agent training environment with persistent skill graphs, adaptive curriculum selection, and deterministic rubric scoring. The primary workflow runs three Hugging Face models through the environment and logs rewards, skills, and plots.

## Deployed Environment

- Hugging Face Space: [https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env](https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env)

## Core Components

- `skillgraph_adaptive_env/server/skillgraph_adaptive_env_environment.py`: turn loop, task selection, rewards, state transitions.
- `skillgraph_adaptive_env/server/task_library.py`: fixed task inventory across collaborative, competitive, mixed_motive, peer_teaching, debate.
- `skillgraph_adaptive_env/server/skill_graph.py`: per-agent skill state, confidence, and updates over time.
- `skillgraph_adaptive_env/server/curriculum_engine.py`: weak-skill targeting, diagnostics, and verification checks.
- `skillgraph_adaptive_env/server/scoring.py`: deterministic reward decomposition and penalty logic.
- `skillgraph_adaptive_env/training/run_training_trl_grpo.py`: TRL GRPO fine-tuning using environment rewards.
- `skillgraph_adaptive_env/training/run_training_three_models.py`: model-backed evaluation run with three HF models and artifact generation.
- `skillgraph_adaptive_env/ui/app.py`: Streamlit inspection UI for logs and plots.

## Supported Training Workflows

### TRL GRPO Training (Model Fine-Tuning)

Collect environment rollouts (prompt + env reward), then fine-tune with TRL GRPO:

```bash
pip install -e "skillgraph_adaptive_env[trl]"
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 40 \
  --seed 7 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --out-dir training/runs/trl_grpo
```

Outputs:
- `training/runs/trl_grpo/grpo_dataset.json`
- `training/runs/trl_grpo/summary.json`
- `training/runs/trl_grpo/checkpoints/final/`

### Three-Model Training (HF Inference Evaluation)

This workflow maps three agents to three real HF models and runs them through the environment:

- `agent_alpha` -> `meta-llama/Llama-3.2-1B-Instruct`
- `agent_beta` -> `Qwen/Qwen2.5-1.5B-Instruct`
- `agent_gamma` -> `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Run from repository root:

```bash
pip install -e skillgraph_adaptive_env
python -m skillgraph_adaptive_env.training.run_training_three_models \
  --episodes 3 \
  --seed 7 \
  --hf-token <YOUR_HF_TOKEN> \
  --out-dir training/runs/hf_three_models
```

## Training Outputs and Artifacts

The three-model run writes:

- `training/runs/hf_three_models/episode_logs.csv`
- `training/runs/hf_three_models/episode_logs.jsonl`
- `training/runs/hf_three_models/summary.json`
- `training/runs/hf_three_models/reward_vs_steps.png`
- `training/runs/hf_three_models/skill_evolution.png`
- `training/runs/hf_three_models/weak_to_strong_transition.png`

## Output Log Structure (Three-Model Run)

`episode_logs.csv` / JSONL rows include:

- episode metadata: `episode`, `task_id`, `task_type`, `turn`, `curriculum_bucket`
- model execution metadata: `agent_id`, `model_id`, `response_text`
- task/skill metadata: `skills`, `difficulty`, `self_rating`, `success`
- reward metrics: `reward`, `task_score`, `skill_improvement`, `consistency`, `skill_drop`

## Demo Artifacts (Committed Run)

Example outputs from a completed run are included at `training/runs/final_run/`:

- `episode_logs.csv`, `episode_logs.jsonl`, `summary.json`
- `reward_vs_steps.png`, `skill_evolution.png`, `weak_to_strong_transition.png`

## UI Dashboard

Inspect logs and plots with Streamlit:

```bash
pip install -e skillgraph_adaptive_env
streamlit run skillgraph_adaptive_env/ui/app.py
```

The UI reads run directories under `training/runs/` (for example `training/runs/final_run` or your latest `--out-dir`).

## Graph Interpretation

- `reward_vs_steps.png`: scalar reward trend across episode-turn steps.
- `skill_evolution.png`: mean skill evolution per agent over episodes.
- `weak_to_strong_transition.png`: trajectory of the weakest initial agent over time.

## Day 1 Sprint (quick reproduction)

Local rollout collection (no GPU, no HF token):

```bash
pip install -e skillgraph_adaptive_env
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 12 --seed 7 --collect-only \
  --out-dir training/runs/trl_grpo_day1
```

Colab notebook: `skillgraph_adaptive_env/training/colab_day1_sprint.ipynb`

Expected artifacts under `training/runs/trl_grpo_day1/`:
- `grpo_dataset.json`, `episode_logs.csv`, `summary.json`, `eval_summary.json`
- `plots/reward_vs_steps.png`, `plots/success_rate_trend.png`, `plots/reward_components.png`

## Notes

- `run_training_three_models.py` is model-backed environment interaction and evaluation with logged rewards.
- It does not fine-tune model weights; it evaluates/model-rolls through the environment and records learning signals.
