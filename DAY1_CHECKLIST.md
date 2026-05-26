# Day-1 hackathon checklist

## Done in repo

- [x] HF Space running with AMASES UI
- [x] TRL collect script + plots (`run_training_trl_grpo.py`)
- [x] Reward logic doc (`skillgraph_adaptive_env/REWARD_LOGIC.md`)
- [x] Reward tuning for visible curves (keyword-aware simulator, episode bonus, skill plot)
- [x] Colab notebook (`skillgraph_adaptive_env/training/colab_day1_sprint.ipynb`)
- [x] Blogpost draft (`skillgraph_adaptive_env/blogpost.md`)

## You run (with HF token / Colab GPU)

### 1. Refresh rollout plots (~2 min, local)

```powershell
cd C:\Users\diyak\skillgraph-adaptive-llm
pip install -e skillgraph_adaptive_env
python -m skillgraph_adaptive_env.training.run_training_trl_grpo `
  --episodes 30 --seed 7 --collect-only `
  --out-dir training/runs/trl_grpo_day1
```

Check: `training/runs/trl_grpo_day1/plots/` — reward trend up, skill_evolution up.

### 2. Three-model HF eval (~$3–5)

```powershell
$env:HF_TOKEN="your_token"
python -m skillgraph_adaptive_env.training.run_training_three_models `
  --episodes 3 --seed 7 --hf-token $env:HF_TOKEN `
  --max-tokens 64 --max-api-calls 40 `
  --out-dir training/runs/hf_three_models_day1
```

### 3. GRPO on Colab T4 (~30–60 min)

Open `colab_day1_sprint.ipynb` or:

```bash
pip install -r skillgraph_adaptive_env/training/requirements-trl.txt
pip install -e skillgraph_adaptive_env
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 30 --seed 7 --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --out-dir training/runs/trl_grpo_day1
```

### 4. Package submission

- [ ] Paste plot paths + metrics into `blogpost.md`
- [ ] Screenshot Space + one plot
- [ ] `git push origin main` (optional: commit artifacts)

## Links

- Space: https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env
- Repo: https://github.com/Diyakalra1/skillgraph-adaptive-llm
