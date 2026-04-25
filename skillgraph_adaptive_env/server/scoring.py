"""Rubric and LLM-judge scoring utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import mean

from .model_runtime import HfModelRuntime


ROLE_KEYS = ("planner", "negotiator", "teacher")
RUBRIC_WEIGHTS = {
    "task_success": 0.24,
    "skill_demo": 0.31,
    "collab_quality": 0.21,
    "learning_evidence": 0.16,
    "meta_cognition": 0.08,
}
SKILL_TOKEN_MAP: dict[str, list[str]] = {
    "negotiation": ["offer", "counter", "concede", "deal", "split", "bid", "proposal", "terms", "tradeoff", "trade-off"],
    "collaboration": ["together", "shared", "joint", "align", "support", "coordinate", "team", "cooperate"],
    "strategic_reasoning": ["strategy", "risk", "trade-off", "scenario", "long-term", "contingency", "fallback", "priority"],
    "information_synthesis": ["evidence", "synthesize", "integrate", "source", "summary", "combine", "insight", "finding"],
    "communication": ["explain", "clarify", "because", "example", "question", "rationale", "therefore"],
    "meta_learning": ["reflect", "revise", "improve", "feedback", "adapt", "iteration", "update", "learned"],
    "communication_clarity": ["explain", "clarify", "because", "example", "question", "rationale"],
    "competitive_strategy": ["offer", "counter", "reserve", "utility", "win", "maximize", "leverage"],
    "opponent_modeling": ["you value", "your priority", "opponent", "likely", "your constraint", "your objective"],
    "risk_assessment": ["risk", "failure", "contingency", "uncertain", "downside", "mitigate"],
    "problem_decomposition": ["step", "milestone", "breakdown", "plan", "phase", "sequence"],
    "argumentation": ["claim", "premise", "rebuttal", "logic", "counterpoint", "evidence"],
}


@dataclass
class ScoreBundle:
    rubric: dict[str, float]
    judge: dict[str, float]
    merged: dict[str, float]
    merged_reward: float


@dataclass
class RewardResult:
    scalar: float
    rubric: dict[str, float]
    skill_vector: dict[str, float]
    penalties: dict[str, float]


def rubric_score(response_text: str) -> dict[str, float]:
    text = (response_text or "").lower()
    planner = 0.2 + 0.2 * sum(k in text for k in ["plan", "step", "risk", "timeline"])
    negotiator = 0.2 + 0.2 * sum(k in text for k in ["trade-off", "constraint", "budget", "deal"])
    teacher = 0.2 + 0.2 * sum(k in text for k in ["example", "explain", "because", "lesson"])
    return {
        "planner": round(min(1.0, planner), 4),
        "negotiator": round(min(1.0, negotiator), 4),
        "teacher": round(min(1.0, teacher), 4),
    }


def _extract_json_scores(text: str) -> dict[str, float]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for key in ROLE_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            out[key] = round(max(0.0, min(1.0, float(value))), 4)
    return out


def llm_judge_score(
    runtime: HfModelRuntime,
    judge_model: str,
    task_prompt: str,
    response_text: str,
) -> tuple[dict[str, float], str]:
    prompt = (
        "Score the response for planner, negotiator, teacher in [0,1]. "
        "Return only JSON: "
        '{"planner": <float>, "negotiator": <float>, "teacher": <float>}.\n\n'
        f"Task: {task_prompt}\nResponse: {response_text}"
    )
    result = runtime.generate(judge_model, prompt)
    if not result.ok:
        return {}, result.error
    parsed = _extract_json_scores(result.text)
    if not parsed:
        return {}, "judge_parse_failed"
    return parsed, ""


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _task_success_score(outcome: dict) -> float:
    agreed = 1.0 if outcome.get("agreement_reached", False) else 0.0
    quality = float(outcome.get("quality", 0.0))
    turns_used = float(outcome.get("turns_used", 0))
    max_turns = float(max(1, int(outcome.get("max_turns", 1))))
    efficiency = max(0.0, 1.0 - (turns_used / max_turns))
    if not agreed:
        return round(min(0.45, 0.35 * quality + 0.10 * efficiency), 4)
    return min(1.0, 0.6 * agreed + 0.25 * efficiency + 0.15 * quality)


def _skill_demo_score(task_type: str, text: str) -> float:
    if task_type == "competitive":
        score = 0.0
        score += 0.28 if _contains_any(text, ["counter", "counter-offer", "revised offer", "alternative offer"]) else 0.0
        score += 0.20 if _contains_any(text, ["concede", "flex", "non-priority", "adjust"]) else 0.0
        score += 0.26 if _contains_any(text, ["must-have", "non-negotiable", "required", "priority"]) else 0.0
        score += 0.16 if _contains_any(text, ["build on", "your idea", "as you suggested", "based on your"]) else 0.0
        score += 0.10 if _contains_any(text, ["because", "therefore", "rationale"]) else 0.0
        return min(1.0, score)
    score = 0.0
    score += 0.26 if _contains_any(text, ["?", "clarify", "can you explain", "quick check"]) else 0.0
    score += 0.28 if _contains_any(text, ["constraint", "your requirement", "your goal", "limitation", "must"]) else 0.0
    score += 0.28 if _contains_any(text, ["proposal", "because", "therefore", "so that", "plan"]) else 0.0
    score += 0.18 if _contains_any(text, ["next step", "timeline", "action item", "milestone"]) else 0.0
    return min(1.0, score)


def _collab_quality(turn_texts: list[str], current_text: str, context_refs: list[str]) -> float:
    if not turn_texts:
        return 0.5
    repetition = max(0.0, sum(1 for t in turn_texts if t.strip().lower() == current_text.strip().lower()) / max(1, len(turn_texts)))
    has_context = 1.0 if _contains_any(current_text.lower(), context_refs) else 0.0
    lengths = [len(t.strip().split()) for t in turn_texts if t.strip()]
    balance = 1.0
    if lengths:
        avg = max(1.0, mean(lengths))
        span = max(lengths) - min(lengths)
        balance = max(0.0, 1.0 - (span / (avg * 4)))
    score = 0.35 * balance + 0.35 * has_context + 0.30 * (1.0 - repetition)
    return round(max(0.0, min(1.0, score)), 4)


def _learning_evidence(turn_idx: int, turn_texts: list[str], current_text: str) -> float:
    if turn_idx < 5:
        return 0.5
    if not turn_texts:
        return 0.2
    novelty = 0.0
    if all(current_text.strip().lower() != prev.strip().lower() for prev in turn_texts[-3:]):
        novelty += 0.6
    if _contains_any(current_text.lower(), ["revise", "update", "new strategy", "alternative", "adjust", "improve"]):
        novelty += 0.4
    return min(1.0, novelty)


def _meta_cognition(self_rating: float, weighted_without_meta: float) -> float:
    diff = abs(float(self_rating) - weighted_without_meta)
    return round(max(0.0, min(1.0, 1.0 - diff)), 4)


def _penalties(
    task_type: str,
    turn_idx: int,
    max_turns: int,
    agreement_reached: bool,
    current_text: str,
    turn_texts: list[str],
    self_rating: float,
    predicted_score: float,
    context_refs: list[str],
) -> dict[str, float]:
    text = (current_text or "").lower()
    penalties = {
        "instant_agreement_hack": 0.0,
        "proposal_repetition": 0.0,
        "context_ignoring": 0.0,
        "timeout_failure": 0.0,
        "incoherent_output": 0.0,
        "self_assessment_inflation": 0.0,
    }
    if agreement_reached and turn_idx <= 2 and not _contains_any(text, ["counter", "evaluate", "trade-off", "constraint"]):
        penalties["instant_agreement_hack"] = 0.18
    repeat_hits = sum(1 for prev in turn_texts[-3:] if prev.strip().lower() == text and text)
    if repeat_hits >= 2:
        penalties["proposal_repetition"] = 0.08
    if context_refs and not _contains_any(text, context_refs):
        penalties["context_ignoring"] = 0.05
    if (turn_idx >= max_turns) and not agreement_reached and task_type in {"collaborative", "mixed_motive"}:
        penalties["timeout_failure"] = 0.12
    if len(text.strip()) < 4:
        penalties["incoherent_output"] = 0.18
    if (self_rating - predicted_score) > 0.35:
        penalties["self_assessment_inflation"] = 0.08
    return penalties


def compute_reward(
    *,
    task_type: str,
    task_skills: list[str],
    turn_idx: int,
    max_turns: int,
    current_text: str,
    turn_texts: list[str],
    context_refs: list[str],
    self_rating: float,
    outcome: dict,
) -> RewardResult:
    text = (current_text or "").lower()
    task_success = _task_success_score(outcome)
    skill_demo = _skill_demo_score(task_type, text)
    collab_quality = _collab_quality(turn_texts=turn_texts, current_text=text, context_refs=context_refs)
    learning_evidence = _learning_evidence(turn_idx=turn_idx, turn_texts=turn_texts, current_text=text)

    pre_meta = (
        RUBRIC_WEIGHTS["task_success"] * task_success
        + RUBRIC_WEIGHTS["skill_demo"] * skill_demo
        + RUBRIC_WEIGHTS["collab_quality"] * collab_quality
        + RUBRIC_WEIGHTS["learning_evidence"] * learning_evidence
    )
    meta_cognition = _meta_cognition(self_rating=self_rating, weighted_without_meta=pre_meta)
    rubric = {
        "task_success": round(task_success, 4),
        "skill_demo": round(skill_demo, 4),
        "collab_quality": round(collab_quality, 4),
        "learning_evidence": round(learning_evidence, 4),
        "meta_cognition": round(meta_cognition, 4),
    }
    scalar = sum(RUBRIC_WEIGHTS[k] * rubric[k] for k in RUBRIC_WEIGHTS)
    penalties = _penalties(
        task_type=task_type,
        turn_idx=turn_idx,
        max_turns=max_turns,
        agreement_reached=bool(outcome.get("agreement_reached", False)),
        current_text=text,
        turn_texts=turn_texts,
        self_rating=self_rating,
        predicted_score=scalar,
        context_refs=context_refs,
    )
    scalar = max(0.01, scalar - sum(penalties.values()))
    base_skill_value = round(max(0.0, min(1.0, 0.45 * skill_demo + 0.25 * learning_evidence + 0.30 * collab_quality)), 4)
    derived_matches: dict[str, float] = {}
    for skill in task_skills:
        tokens = SKILL_TOKEN_MAP.get(skill, [])
        if not tokens:
            continue
        hit_ratio = sum(1 for token in tokens if token in text) / len(tokens)
        if hit_ratio > 0:
            derived_matches[skill] = round(min(1.0, 0.5 * base_skill_value + 0.5 * hit_ratio), 4)
    if derived_matches:
        skill_vector = derived_matches
    else:
        # Fallback when no explicit signal is found in text.
        skill_vector = {skill: round(base_skill_value * 0.6, 4) for skill in task_skills}
    return RewardResult(
        scalar=round(scalar, 4),
        rubric=rubric,
        skill_vector=skill_vector,
        penalties=penalties,
    )


def merge_scores(rubric: dict[str, float], judge: dict[str, float], judge_weight: float = 0.4) -> ScoreBundle:
    merged: dict[str, float] = {}
    for key in ROLE_KEYS:
        r = rubric.get(key, 0.0)
        j = judge.get(key, r)
        merged[key] = round((1 - judge_weight) * r + judge_weight * j, 4)
    merged_reward = round(sum(merged.values()) / len(ROLE_KEYS), 4)
    return ScoreBundle(rubric=rubric, judge=judge, merged=merged, merged_reward=merged_reward)
