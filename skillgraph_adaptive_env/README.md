# AMASES: Adaptive Multi-Agent Skill Evolution System

This project now follows a multi-agent architecture with **3 model-backed agents**:

- `agent_alpha` (planner)
- `agent_beta` (debater)
- `agent_gamma` (integrator)

The baseline environment still runs without external dependencies, and a
dedicated 7-iteration Hugging Face runner is included for free-tier experiments.

## Fixed Task Inventory

AMASES uses a fixed 15-task adaptive curriculum:

- Collaborative: easy/medium/hard
- Competitive: easy/medium/hard
- Mixed motive: easy/medium/hard
- Peer teaching: easy/medium/hard
- Debate: easy/medium/hard

Each task is hardcoded with the same structure fields:
`type`, `agents_needed`, `skills_tested`, `difficulty (0-5)`, `max_turns`, `reward_mode`,
`rules`, and rubric settings.

Runtime variation only changes surface content (topic, budget, scenario labels); task logic does not change.

## Architecture

- `server/skillgraph_adaptive_env_environment.py`
  Main OpenEnv environment, turn loop, rewards, and episode state.
- `server/skill_graph.py`
  Per-agent skill graph tracking, trajectories, confidence, and updates.
- `server/task_library.py`
  Multi-agent task types: collaborative, competitive, mixed motive, peer teaching, debate.
- `server/agent_manager.py`
  Agent registry and team matching.
- `server/curriculum_engine.py`
  Confidence-aware weakest-skill selection, cold-start diagnostics, and every-20-episode verification checks.
- `server/model_runtime.py`
  HF runtime wrapper with retries/backoff and error capture.
- `server/scoring.py`
  Deterministic rubric scorer + penalties + reward vectors + optional debate judge scoring.
- `server/role_classifier.py`
  Per-iteration and final role classification.
- `models.py`
  Action/Observation schema for multi-agent turns and arena state.
- `training/run_training.py`
  Training driver for multi-agent simulation and output artifacts.
- `ui/app.py`
  Simple Streamlit inspector for logs and generated plots.

## Reward Design

Per-turn reward categories:

- `task_success` (30%)
- `skill_demo` (25%)
- `collab_quality` (20%)
- `learning_evidence` (15%)
- `meta_cognition` (10%)

Weighted scalar reward is used by trainer loops, while per-skill vectors are used for skill-graph updates.
Penalty hooks include instant agreement hacks, repeated proposals, context-ignoring turns, timeout failures, incoherent output, and self-assessment inflation.

## Key Variables (Meaning)

- `difficulty`: task hardness on 0-5 scale.
- `skills_tested`: skills directly updated by this task.
- `reward_mode`: `shared`, `zero_sum`, `partial`, `linked`, or `judge_scored`.
- `curriculum_bucket`: selected difficulty band (`easy`, `medium`, `hard`, diagnostics, verification).
- `confidence`: how reliable each skill estimate is; rises with more task evidence.
- `learning_velocity`: improvement trend over recent history.
- `plateau`: true when a skill barely changes over recent tasks.

## How To Run

From repo root:

1) Install package

```bash
pip install -e skillgraph_adaptive_env
```

2) Run baseline simulation training

```bash
python -m skillgraph_adaptive_env.training.run_training --episodes 120 --seed 7
```

3) Build GRPO-ready rollout dataset (manual, no model serving required)

```bash
python -m skillgraph_adaptive_env.training.run_training_trl_grpo --episodes 40 --seed 7 --print-trl-template
```

4) Open UI

```bash
streamlit run skillgraph_adaptive_env/ui/app.py
```

5) Run the 7-iteration HF plan (exactly 7 iterations)

```bash
python -m skillgraph_adaptive_env.training.run_training_hf_7iter --out-dir training/runs/hf_7iter --hf-token <YOUR_HF_TOKEN>
```

## Output Artifacts

Outputs are written to `training/runs/latest/`:

- `episode_logs.csv`
- `episode_logs.jsonl`
- `summary.json`
- `reward_vs_steps.png`
- `skill_evolution.png`
- `weak_to_strong_transition.png`

For the 7-iteration HF run (`training/runs/hf_7iter/`), the key artifacts are:

- `iteration_report.jsonl`
- `final_classification.json`

## Story (README/Blog Ready)

AMASES starts with three agents that all look similar at first.  
They are not labeled as "good" or "bad" manually. Instead, every task measures
how each agent performs on skill rubrics (collaboration, strategy, synthesis, negotiation).

After each task:
- the environment updates each agent's skill graph,
- identifies weak and improving areas,
- and chooses the next task difficulty accordingly.

This means weak skills get easier practice first, improving skills get harder
challenges, and strong skills get mixed-motive scenarios where transfer is needed.
Over many tasks, agents diverge into specialties naturally through interaction.

The core value is persistent long-horizon learning:
- not random task jumping,
- not one scalar reward only,
- but measurable per-skill progression with adaptive curriculum.

## Clarity: What AMASES Is / Is Not

**AMASES is:**
- a multi-agent training environment
- persistent per-agent skill tracking
- adaptive curriculum based on weak/improving skills
- collaborative + competitive + mixed + teaching + debate interactions

**AMASES is not (in current scope):**
- a production user-profiling system
- a full TRL/Unsloth run pipeline yet
- a final deployment product

## Requirement Coverage Snapshot

- Multi-agent environment with agent pool: ✅
- Persistent skill graph tracking: ✅
- Curriculum engine (weak -> easy, improving -> harder, strong -> mixed): ✅
- Competitive/collaborative/mixed/teaching/debate tasks: ✅
- Multi-dimensional reward decomposition: ✅
- Public/private memory with task-type masking: ✅
- UI heatmap: ✅
- UI live task arena with message threads: ✅
- UI curriculum planner: ✅
- UI interaction network (matrix view): ✅
- Training outputs and plots: ✅
- TRL + GRPO dataset entrypoint: ✅
- Unsloth integration: ⚠️ optional future extension

## Part 10: Create Your Own Integration

Use the project-aligned 5-step integration guide here:

- `INTEGRATION_GUIDE.md`

It covers:
- action/observation types
- environment implementation
- client parsing/payload
- OpenEnv app wiring
- Docker packaging + validation checklist

This AMASES environment now follows that pattern directly:
- Types in `models.py` (`SkillgraphAdaptiveAction`, `SkillgraphAdaptiveObservation`, `SkillgraphAdaptiveState`)
- Environment entry in `server/skillgraph_adaptive_env_environment.py`
- Client adapter in `client.py`
- Server wiring in `server/app.py`

## HF Free-Tier Note

Hugging Face free/serverless models can fail intermittently due to cold starts,
rate limits, or credits. The 7-iteration runner captures call failures in
`iteration_report.jsonl` and includes model reliability stats in
`final_classification.json`.
