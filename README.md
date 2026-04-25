# skillgraph-adaptive-llm

Implementation of **AMASES (Adaptive Multi-Agent Skill Evolution System)** with:

- persistent per-agent skill graphs
- adaptive curriculum selection
- multi-agent task arena (collaborative/competitive/mixed)
- simulation-first training loop that is ready to swap to real LLMs

Current curriculum design in this repo:

- fixed **15 adaptive tasks** (5 interaction types x 3 difficulty tiers)
- fixed **medium diagnostic variants** (1 per type) used for cold start and verification only
- deterministic task structure with light runtime surface randomization
- rubric-first reward vectors plus weighted scalar reward for GRPO-compatible training

Main entrypoints:
- `skillgraph_adaptive_env/training/run_training.py`
- `skillgraph_adaptive_env/training/run_training_trl_grpo.py`
- `skillgraph_adaptive_env/ui/app.py`
- `skillgraph_adaptive_env/server/skillgraph_adaptive_env_environment.py`
