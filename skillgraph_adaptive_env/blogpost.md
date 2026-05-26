# AMASES Blog Post (Draft)

## Problem

Multi-agent LLM systems often lack measurable, persistent skill growth. AMASES addresses this with an adaptive curriculum environment where agents interact across collaborative, competitive, teaching, and debate tasks while receiving structured rewards.

## Environment Design

- **Agents:** `agent_alpha` (planner), `agent_beta` (debater), `agent_gamma` (integrator)
- **Tasks:** 15 fixed tasks (5 families × 3 difficulty tiers)
- **Curriculum:** weak-skill targeting, diagnostics (episodes 1–5), verification checks
- **Rewards:** weighted rubric + penalties (see `REWARD_LOGIC.md`); keywords gate **success** but do not dominate total reward

**Deployed Space:** https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env

## Training Methods

### A) Three-model HF evaluation (real inference)

Three distinct small instruct models via `huggingface_hub.InferenceClient`:

| Agent | Role | Model |
|--------|------|--------|
| `agent_alpha` | Planner | `meta-llama/Llama-3.2-1B-Instruct` |
| `agent_beta` | Debater | `Qwen/Qwen2.5-1.5B-Instruct:featherless-ai` |
| `agent_gamma` | Integrator | `google/gemma-3-1b-it:featherless-ai` |

```bash
python -m skillgraph_adaptive_env.training.run_training_three_models \
  --episodes 3 --seed 7 --hf-token $HF_TOKEN \
  --max-tokens 64 --max-api-calls 40 --turn-cap 10 \
  --out-dir training/runs/hf_three_models_day1
```

### B) TRL GRPO rollout collection (baseline dataset)

```bash
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 30 --seed 7 --collect-only \
  --out-dir training/runs/trl_grpo_day1
```

Full GRPO training (policy update) runs on **Colab T4** with the same script without `--collect-only`.

---

## Results summary

| Metric | TRL collect (30 ep, sim policy) | Three-model HF (3 ep, real LLMs) | After GRPO |
|--------|--------------------------------:|---------------------------------:|-----------:|
| Avg step reward | 0.309 | 0.412 | TBD (Colab) |
| Per-episode success (any turn) | ~35% of steps | 0% steps marked success | TBD |
| API / rollout rows | 74 steps | 30 steps (30 API calls) | — |
| Fallback rate | N/A | 0% | — |

---

## What the logs and graphs mean

### 1. Three-model run — `reward_vs_steps.png`

**File:** `training/runs/hf_three_models_day1/reward_vs_steps.png`

**What it shows:** Per-turn environment reward across 3 episodes (each episode = one vertical cluster of points).

**How to read it:**
- **Episode 1 (collaborative_medium):** Rewards cluster high (~0.47–0.55). Models produce structured bullets with plan/evidence language; rubric scores are strong on cold-start diagnostic task.
- **Episode 2 (competitive_medium):** Sharp drop; minimum reward **~0.15**. Competitive/auction tasks need different vocabulary (offer, counter, bid). Llama/Qwen/Gemma replies are more generic here → lower `skill_demo` and keyword alignment.
- **Episode 3 (mixed_motive_medium):** Recovery to ~0.38–0.54. Agents adapt phrasing toward coalition/trade-off language.

**Significance:** Real models **do affect reward** through task-appropriate language. Short 3-episode run is not training—it is **evaluation** that the env + rubric respond sensibly to live generations.

### 2. Three-model run — `skill_evolution.png`

**File:** `training/runs/hf_three_models_day1/skill_evolution.png`

**What it shows:** Mean skill-graph level (0–5) per agent per episode.

**How to read it:**
- All agents start near **~2.4–2.5** (default init).
- **agent_gamma (Gemma)** stays highest (~2.5); **agent_alpha (Llama)** drifts down slightly (~2.5 → ~2.2); **agent_beta (Qwen)** is flat (~2.4).

**Significance:** With only **3 episodes** and no weight updates, skill levels barely move. This plot is evidence the **skill graph is wired**, not that multi-day learning occurred. Longer runs or GRPO are needed for clear upward trends.

### 3. Three-model run — `weak_to_strong_transition.png`

**File:** `training/runs/hf_three_models_day1/weak_to_strong_transition.png`

**What it shows:** Zoomed skill trajectory for the initially weakest agent (`agent_beta`).

**How to read it:** Nearly flat (2.399 → 2.397 over 3 episodes).

**Significance:** “Weak-to-strong” transition needs **more episodes** or training; this run is too short to show curriculum-driven skill growth.

### 4. Three-model logs — `episode_logs.csv`

**Per-agent average reward (30 turns):**

| Agent | Model | Avg reward | Max turn reward |
|--------|--------|------------|-----------------|
| agent_beta | Qwen 1.5B | **0.507** | 0.547 |
| agent_gamma | Gemma 3 1B | 0.419 | 0.535 |
| agent_alpha | Llama 1B | 0.364 | 0.542 |

**Significance:** Qwen (debater role) scored highest on average in this seed—often assigned more turns in collaborative episode and produced rubric-friendly bullet structure. `success=False` on rows means the strict **keyword solve threshold** was not met in a single turn (common with 10-turn cap); rewards can still be **moderate/high** from partial rubric credit.

### 5. TRL collect — `reward_vs_steps.png`

**File:** `training/runs/trl_grpo_day1/plots/reward_vs_steps.png`

**What it shows:** 74 rollout steps; blue = per-step reward, orange = moving average.

**How to read it:**
- Steps **0–49:** Low rewards (~0–0.2) with noise.
- **~Step 50:** Jump to ~0.7+ as the collection policy improves (simulator uses **episode-ramping** self-rating and keyword-aware responses).
- **Steps 50–74:** Moving average plateaus near **~0.75**.

**Significance:** This is the **GRPO training dataset quality curve**, not post-training model improvement. It shows we can generate high-reward trajectories for later fine-tuning. The spike is expected from the scripted ramp policy used during `collect-only`.

### 6. TRL collect — `success_rate_trend.png`

**File:** `training/runs/trl_grpo_day1/plots/success_rate_trend.png`

**What it shows:** Fraction of turns marked `success` (keyword solve) per episode.

**How to read it:**
- Episodes **1–3:** 0% (cold-start diagnostics, weak responses).
- Episode **4:** 100% (first strong keyword-aligned episode).
- Episode **5:** dip to 0%.
- Episodes **6–30:** 100% success each episode.

**Significance:** Aligns with curriculum: early diagnostic episodes are hard; later episodes use the improving simulator policy. Good signal for **GRPO positives**, but success metric is **stricter** than average reward.

### 7. TRL collect — `skill_evolution.png`

**File:** `training/runs/trl_grpo_day1/plots/skill_evolution.png`

**What it shows:** Mean skill level across episodes (aggregated over agents).

**How to read it:** Stable band **~2.0–2.4** with small wobble; no large climb over 30 episodes.

**Significance:** Skill graph α=0.1 updates slowly; reward can rise while level stays flat. Use this plot with reward plots together—**reward is the primary training signal for GRPO**.

### 8. TRL collect — `reward_components.png`

**File:** `training/runs/trl_grpo_day1/plots/reward_components.png`

**What it shows:** Moving average of rubric breakdown terms over steps.

**Significance:** After step ~50, `task_success`, `collab_quality`, and related terms rise together—confirms gains are from **structured multi-component rubric**, not a single hacked feature.

---

## Screenshots for submission

Insert these in the blog / slides:

1. Space playground — Reset + Step with reward breakdown  
2. `training/runs/hf_three_models_day1/reward_vs_steps.png`  
3. `training/runs/hf_three_models_day1/skill_evolution.png`  
4. `training/runs/trl_grpo_day1/plots/reward_vs_steps.png`  
5. `training/runs/trl_grpo_day1/plots/success_rate_trend.png`  

---

## Interpretation (research narrative)

1. **Environment works:** Live HF models, rubric rewards, and skill graph updates all produce measurable, interpretable signals.  
2. **Task type matters:** Collaborative episodes score higher than competitive with the same prompt template—future work: task-specific prompt hints per model role.  
3. **GRPO dataset is viable:** Rollout collection produces a clear high-reward tail (avg episode 30 ≈ 0.83 per-step mean) for fine-tuning.  
4. **Next proof point:** Run full GRPO on Colab and compare **post-train** rollouts vs baseline 0.309 avg reward.

---

## Limitations

- Very short HF eval (3 episodes, 10 turns each); `success_rate` 0% on HF run despite decent rewards.  
- TRL collect uses **simulated** responses (not live HF) unless extended.  
- LLM-as-judge not enabled in Space loop (optional future upgrade).  
- Skill-level plots move slowly; reward plots are the main learning indicator for day-1.

---

## Reproduce in 10 minutes

```bash
pip install -e skillgraph_adaptive_env
pip install openai  # only if using OpenAI router variant

# Local GRPO dataset + plots
python -m skillgraph_adaptive_env.training.run_training_trl_grpo \
  --episodes 30 --seed 7 --collect-only \
  --out-dir training/runs/trl_grpo_day1

# Three real models (HF token required)
python -m skillgraph_adaptive_env.training.run_training_three_models \
  --episodes 3 --seed 7 --hf-token $HF_TOKEN \
  --out-dir training/runs/hf_three_models_day1
```

**Repo:** https://github.com/Diyakalra1/skillgraph-adaptive-llm  
**Space:** https://huggingface.co/spaces/jeeya-ahuja05/skill-graph-adaptive-env
