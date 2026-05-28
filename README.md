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
