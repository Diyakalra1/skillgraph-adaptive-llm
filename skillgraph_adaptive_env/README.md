# SkillGraph: Self-Aware LLM with Adaptive Curriculum (Baseline)

This baseline turns the default OpenEnv template into a **skill-aware training world**:

- Tasks are tagged with skills (`reasoning`, `coding`, `debugging`, `communication`, etc.)
- A live **skill graph** tracks each node's:
  - level (`0 -> 1`)
  - confidence (`0 -> 1`)
  - streak (consistency signal)
- Curriculum selection adapts by weakness:
  - weak skill -> easier tasks
  - improving skill -> medium/harder tasks
  - strong skill -> mixed challenges

## Reward Design

The environment uses decomposed reward:

`Reward = TaskScore + alpha*SkillImprovement + beta*Consistency - gamma*SkillDrop`

Where:
- `TaskScore`: +1.0 for success, -0.4 for failure
- `SkillImprovement`: positive skill delta this episode
- `Consistency`: streak-based stability reward
- `SkillDrop`: penalty for regressions

## Project Additions

- `server/skillgraph_adaptive_env_environment.py`  
  SkillGraph environment with adaptive task selection and reward decomposition.
- `models.py`  
  Action/Observation models for tasks, skill snapshots, and reward breakdown.
- `training/run_training.py`  
  Baseline training loop with clear episode logs + graph generation.
- `ui/app.py`  
  Simple Streamlit UI to inspect logs and charts quickly.

## How To Run

From the `skillgraph_adaptive_env` directory:

1) Install dependencies

```bash
pip install -e .
```

2) Run simulation training (no model/API required)

```bash
python -m skillgraph_adaptive_env.training.run_training --episodes 120 --seed 7
```

3) Open simple UI

```bash
streamlit run ui/app.py
```

## What You Will See

Training writes outputs to `training/runs/latest/`:

- `episode_logs.csv` -> per-episode clear logs
- `episode_logs.jsonl` -> line-by-line machine readable logs
- `summary.json` -> avg reward, success rate, metadata
- `reward_vs_steps.png`
- `skill_evolution.png`
- `weak_to_strong_transition.png`

Example console log:

```text
[EP 001] task=reason_easy_1    bucket=easy   success=False reward=-0.431 rating=0.42 skills=reasoning
[EP 002] task=code_easy_1      bucket=easy   success=True  reward= 1.099 rating=0.49 skills=coding
```

## OpenEnv Server (Optional)

You can still run the environment as an OpenEnv HTTP/WebSocket server:

```bash
uvicorn server.app:app --reload
```

Then use the `SkillgraphAdaptiveEnv` client to call `reset()`/`step()`.
