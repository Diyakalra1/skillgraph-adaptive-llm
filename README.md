# skillgraph-adaptive-llm

AMASES (Adaptive Multi-Agent Skill Evolution System) is an OpenEnv-based multi-agent environment with:

- adaptive curriculum selection,
- deterministic reward decomposition,
- per-agent skill graph updates,
- GRPO training from real rollout responses.

## Deployed environment

- Hugging Face Space: [https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env](https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env)

## Project structure

- `skillgraph_adaptive_env/server/skillgraph_adaptive_env_environment.py` - environment reset/step loop
- `skillgraph_adaptive_env/server/curriculum_engine.py` - diagnostic + adaptive task selection
- `skillgraph_adaptive_env/server/scoring.py` - merged reward + component penalties
- `skillgraph_adaptive_env/server/skill_graph.py` - skill/confidence tracking
- `skillgraph_adaptive_env/training/run_training_trl_grpo.py` - rollout + GRPO training pipeline
- `skillgraph_adaptive_env/training/colab_day1_sprint.ipynb` - Colab workflow
- `skillgraph_adaptive_env/training/requirements-colab.txt` - single source requirements

## Training workflow (current)

Install:

```bash
pip install -e skillgraph_adaptive_env
pip install -r skillgraph_adaptive_env/training/requirements-colab.txt
```

Run GRPO with real rollout responses:

```bash
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 10 \
  --seed 7 \
  --max-turns 6 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --rollout-model-id meta-llama/Llama-3.2-1B-Instruct \
  --hf-token <YOUR_HF_TOKEN> \
  --rollout-max-tokens 64 \
  --max-samples 70 \
  --epochs 1 \
  --max-completion-length 28 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-generations 2 \
  --grpo-temperature 0.9 \
  --out-dir training/runs/trl_grpo_tuned_10ep
```

| Task Category | Difficulty Tier | Task Name (ID) | Skills Tested | Difficulty Score |
|---|---|---|---|---:|
| Collaborative | Easy | `collaborative_easy` | collaboration, problem_decomposition, communication | 2.0 |
| Collaborative | Medium | `collaborative_medium` | information_synthesis, collaboration, communication | 3.0 |
| Collaborative | Hard | `collaborative_hard` | information_synthesis, strategic_reasoning | 4.4 |
| Competitive | Easy | `competitive_easy` | negotiation, competitive_strategy, opponent_modeling | 2.2 |
| Competitive | Medium | `competitive_medium` | competitive_strategy, risk_assessment, opponent_modeling | 3.1 |
| Competitive | Hard | `competitive_hard` | competitive_strategy, strategic_reasoning, negotiation | 4.5 |
| Mixed Motive | Easy | `mixed_motive_easy` | collaboration, strategic_reasoning, negotiation | 2.4 |
| Mixed Motive | Medium | `mixed_motive_medium` | strategic_reasoning, risk_assessment, collaboration | 3.3 |
| Mixed Motive | Hard | `mixed_motive_hard` | strategic_reasoning, collaboration, long_term_planning | 4.6 |
| Peer Teaching | Easy | `peer_teaching_easy` | communication, meta_learning, knowledge_transfer | 2.1 |
| Peer Teaching | Medium | `peer_teaching_medium` | communication, meta_learning | 3.0 |
| Peer Teaching | Hard | `peer_teaching_hard` | meta_learning, communication | 4.2 |
| Debate | Easy | `debate_easy` | information_synthesis, communication, argumentation | 2.6 |
| Debate | Medium | `debate_medium` | argumentation, information_synthesis | 3.4 |
| Debate | Hard | `debate_hard` | argumentation, strategic_reasoning | 4.8 |
Train again on same dataset (no new rollouts):

```bash
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --train-only \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --max-samples 70 \
  --epochs 5 \
  --max-completion-length 28 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-generations 2 \
  --grpo-temperature 0.9 \
  --out-dir training/runs/trl_grpo_tuned_10ep
```

## Output artifacts

Typical run output directory:

- `grpo_dataset.json`
- `episode_logs.csv`
- `summary.json`
- `eval_summary.json`
- `checkpoints/final/`
- `plots/reward_vs_steps.png`
- `plots/success_rate_trend.png`
- `plots/reward_components.png`
- `plots/skill_evolution.png`
- `plots/training_loss.png`

## Notes

- Early episodes are fixed diagnostics (`collaborative`, `competitive`, `mixed_motive`, `peer_teaching`, `debate`), then curriculum switches to weak-skill targeting.
- Success can remain sparse under strict tasks; monitor reward components and skill evolution jointly.
- For the latest reproducible run recipe, prefer the notebook at `skillgraph_adaptive_env/training/colab_day1_sprint.ipynb`.
