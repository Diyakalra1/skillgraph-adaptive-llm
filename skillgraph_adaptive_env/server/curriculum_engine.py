"""Adaptive curriculum and task matching for AMASES."""

from __future__ import annotations

import random

from .task_library import TaskLibrary


class CurriculumEngine:
    """Picks the next task based on weakest skills and adaptive constraints."""

    def __init__(self, task_library: TaskLibrary, seed: int = 7) -> None:
        self._task_library = task_library
        self._rng = random.Random(seed)
        self._diagnostic_order = [
            "collaborative",
            "competitive",
            "mixed_motive",
            "peer_teaching",
            "debate",
        ]
        self._verification_order = [
            "peer_teaching",
            "competitive",
            "collaborative",
            "debate",
            "mixed_motive",
        ]

    def _difficulty_bucket(self, level_0_to_5: float) -> str:
        if level_0_to_5 < 2.5:
            return "easy"
        if level_0_to_5 < 3.5:
            return "medium"
        if level_0_to_5 <= 5.0:
            return "hard"
        return "balanced"

    def _confidence_weighted_weak_skill(self, skill_scores: dict[str, dict[str, float]]) -> tuple[str, float]:
        ranked = []
        for skill, values in skill_scores.items():
            level = float(values.get("level", 2.5))
            conf = float(values.get("confidence", 0.2))
            ranked.append((skill, level * max(conf, 0.15), level))
        ranked.sort(key=lambda item: item[1])
        if not ranked:
            return "collaboration", 2.5
        return ranked[0][0], ranked[0][2]

    def _difficulty_match(self, task: dict, weakest_level: float, bucket: str) -> bool:
        diff = float(task.get("difficulty", 3.0))
        if bucket == "easy":
            return diff <= 2.6
        if bucket == "medium":
            return 2.4 <= diff <= 3.6
        if bucket == "hard":
            return diff >= 3.4
        return abs(diff - weakest_level) <= 1.0

    def _select_regular_task(self, weak_skill: str, weakest_level: float) -> tuple[dict, str]:
        bucket = self._difficulty_bucket(weakest_level)
        candidates = []
        for task in self._task_library.all_tasks():
            skills = task.get("skills_tested", [])
            if weak_skill in skills and self._difficulty_match(task, weakest_level, bucket):
                candidates.append(task)
        if not candidates:
            candidates = [task for task in self._task_library.all_tasks() if self._difficulty_match(task, weakest_level, bucket)]
        if not candidates:
            candidates = self._task_library.all_tasks()
        return self._rng.choice(candidates), bucket

    def choose_task(self, weakest_skills: list[str], weakest_level: float) -> tuple[dict, str]:
        weak_skill = weakest_skills[0] if weakest_skills else "collaboration"
        task, bucket = self._select_regular_task(weak_skill, weakest_level)
        return self._task_library.instantiate_task(task["id"]), bucket

    def choose_task_for_iteration(
        self,
        iteration_idx: int,
        weakest_skills: list[str],
        weakest_level: float,
        previous_role_scores: dict[str, dict[str, float]] | None = None,
    ) -> tuple[dict, str]:
        del previous_role_scores
        if iteration_idx <= len(self._diagnostic_order):
            return self._task_library.fixed_diagnostic_variant(self._diagnostic_order[iteration_idx - 1]), "cold_start_diagnostic"
        return self.choose_task(weakest_skills, weakest_level)

    def choose_task_for_episode(
        self,
        episode_idx: int,
        agent_skill_scores: dict[str, dict[str, dict[str, float]]],
        anchor_agent_id: str,
    ) -> tuple[dict, str]:
        # Cold start diagnostics: fixed medium tasks, no randomization.
        if episode_idx <= len(self._diagnostic_order):
            task_type = self._diagnostic_order[episode_idx - 1]
            return self._task_library.fixed_diagnostic_variant(task_type), "cold_start_diagnostic"

        # Surprise fixed solo verification every 20 episodes.
        if episode_idx % 20 == 0:
            order_idx = (episode_idx // 20 - 1) % len(self._verification_order)
            task_type = self._verification_order[order_idx]
            task = self._task_library.fixed_diagnostic_variant(task_type)
            task["is_verification"] = True
            return task, "verification_check"

        anchor = agent_skill_scores.get(anchor_agent_id, {})
        weak_skill, weak_level = self._confidence_weighted_weak_skill(anchor)
        task, bucket = self._select_regular_task(weak_skill, weak_level)
        task = self._task_library.instantiate_task(task["id"], episode_idx=episode_idx)
        task["target_skill"] = weak_skill
        return task, f"weak_skill_{bucket}"

    def choose_task_for_iteration(
        self,
        iteration_idx: int,
        weakest_skills: list[str],
        weakest_level: float,
        previous_role_scores: dict[str, dict[str, float]] | None = None,
    ) -> tuple[dict, str]:
        """
        Iteration-aware task picker:
        - Iteration 1: fixed baseline collaborative task for all
        - Iteration 2..N: adapt using weak role dimensions from previous scores
        """
        tasks = self._task_library.all_tasks()
        if iteration_idx <= 1:
            baseline = next((t for t in tasks if t["type"] == "collaborative"), tasks[0])
            return dict(baseline), "baseline"

        role_focus = None
        if previous_role_scores:
            # Find weakest average role dimension (planner/negotiator/teacher).
            role_avg: dict[str, float] = {}
            for scores in previous_role_scores.values():
                for role_name, score in scores.items():
                    role_avg.setdefault(role_name, 0.0)
                    role_avg[role_name] += score
            if role_avg:
                role_focus = min(role_avg.items(), key=lambda kv: kv[1])[0]

        if role_focus == "negotiator":
            candidates = [t for t in tasks if t["type"] in ("competitive", "mixed_motive")]
            return dict(self._rng.choice(candidates or tasks)), "role_adapt_negotiator"
        if role_focus == "teacher":
            candidates = [t for t in tasks if t["type"] in ("peer_teaching", "collaborative")]
            return dict(self._rng.choice(candidates or tasks)), "role_adapt_teacher"
        if role_focus == "planner":
            candidates = [t for t in tasks if t["type"] in ("collaborative", "mixed_motive")]
            return dict(self._rng.choice(candidates or tasks)), "role_adapt_planner"

        # fallback to existing weak-skill difficulty policy
        task, bucket = self.choose_task(weakest_skills, weakest_level)
        return dict(task), f"weak_skill_{bucket}"
