# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SkillGraph adaptive curriculum environment implementation."""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SkillgraphAdaptiveAction, SkillgraphAdaptiveObservation
except ImportError:
    from models import SkillgraphAdaptiveAction, SkillgraphAdaptiveObservation


class SkillgraphAdaptiveEnvironment(Environment):
    """Task world with a live skill graph and adaptive curriculum."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._pending_task: dict | None = None
        self._task_history: list[dict] = []
        self._skill_graph = self._build_skill_graph()
        self._task_bank = self._build_task_bank()
        self._reward_weights = {"alpha": 0.7, "beta": 0.2, "gamma": 0.9}

    def _build_skill_graph(self) -> dict[str, dict]:
        return {
            "reasoning": {"level": 0.32, "confidence": 0.45, "streak": 0, "parents": []},
            "advanced_reasoning": {"level": 0.14, "confidence": 0.22, "streak": 0, "parents": ["reasoning"]},
            "arithmetic": {"level": 0.30, "confidence": 0.42, "streak": 0, "parents": []},
            "coding": {"level": 0.27, "confidence": 0.38, "streak": 0, "parents": []},
            "debugging": {"level": 0.16, "confidence": 0.26, "streak": 0, "parents": ["coding"]},
            "optimization": {"level": 0.11, "confidence": 0.18, "streak": 0, "parents": ["debugging"]},
            "communication": {"level": 0.36, "confidence": 0.50, "streak": 0, "parents": []},
            "persuasion": {"level": 0.20, "confidence": 0.30, "streak": 0, "parents": ["communication"]},
            "strategy": {"level": 0.24, "confidence": 0.33, "streak": 0, "parents": []},
            "decision_making": {"level": 0.21, "confidence": 0.31, "streak": 0, "parents": ["strategy"]},
        }

    def _build_task_bank(self) -> list[dict]:
        return [
            {
                "id": "math_easy_1",
                "prompt": "Compute 27 + 16. Return only the number.",
                "skills": ["arithmetic"],
                "difficulty": 0.2,
                "check": {"type": "exact", "target": "43"},
            },
            {
                "id": "math_mid_1",
                "prompt": "Explain briefly why 3/4 is greater than 2/3.",
                "skills": ["reasoning", "arithmetic"],
                "difficulty": 0.45,
                "check": {"type": "keywords", "target": ["0.75", "0.66", "greater"]},
            },
            {
                "id": "reason_easy_1",
                "prompt": "In [2, 4, 8, apple], identify the odd item and explain in one sentence.",
                "skills": ["reasoning"],
                "difficulty": 0.25,
                "check": {"type": "keywords", "target": ["apple", "number"]},
            },
            {
                "id": "reason_hard_1",
                "prompt": "If all robots are machines and some machines are smart, what follows with certainty?",
                "skills": ["advanced_reasoning"],
                "difficulty": 0.72,
                "check": {"type": "keywords", "target": ["cannot conclude", "some robots are smart"]},
            },
            {
                "id": "code_easy_1",
                "prompt": "Write Python code to sum a list named nums.",
                "skills": ["coding"],
                "difficulty": 0.28,
                "check": {"type": "keywords", "target": ["for", "nums", "sum"]},
            },
            {
                "id": "debug_mid_1",
                "prompt": "Fix this bug: for i in range(len(arr)+1): print(arr[i])",
                "skills": ["coding", "debugging"],
                "difficulty": 0.58,
                "check": {"type": "keywords", "target": ["range(len(arr))", "IndexError"]},
            },
            {
                "id": "opt_hard_1",
                "prompt": "How would you optimize a nested loop O(n^2) search for repeated lookups?",
                "skills": ["optimization"],
                "difficulty": 0.82,
                "check": {"type": "keywords", "target": ["hash", "set", "O(n)"]},
            },
            {
                "id": "comm_easy_1",
                "prompt": "Write a polite reminder email for a pending invoice.",
                "skills": ["communication"],
                "difficulty": 0.22,
                "check": {"type": "keywords", "target": ["please", "thank", "reminder"]},
            },
            {
                "id": "persuasion_mid_1",
                "prompt": "Draft a concise proposal convincing a manager to allow remote Fridays.",
                "skills": ["communication", "persuasion"],
                "difficulty": 0.55,
                "check": {"type": "keywords", "target": ["benefit", "productivity", "trial"]},
            },
            {
                "id": "strategy_mid_1",
                "prompt": "Choose launch strategy A or B with limited budget and justify.",
                "skills": ["strategy", "decision_making"],
                "difficulty": 0.62,
                "check": {"type": "keywords", "target": ["because", "risk", "budget"]},
            },
        ]

    def _weakest_skill(self) -> str:
        return min(self._skill_graph.items(), key=lambda item: item[1]["level"])[0]

    def _curriculum_bucket(self, weakest_level: float) -> str:
        if weakest_level < 0.25:
            return "easy"
        if weakest_level < 0.50:
            return "medium"
        if weakest_level < 0.75:
            return "hard"
        return "mixed"

    def _select_task(self) -> tuple[dict, str]:
        weakest = self._weakest_skill()
        weakest_level = self._skill_graph[weakest]["level"]
        bucket = self._curriculum_bucket(weakest_level)

        def by_bucket(task: dict) -> bool:
            if bucket == "easy":
                return task["difficulty"] <= 0.4 and weakest in task["skills"]
            if bucket == "medium":
                return 0.3 <= task["difficulty"] <= 0.7 and weakest in task["skills"]
            if bucket == "hard":
                return task["difficulty"] >= 0.6 and weakest in task["skills"]
            return True

        candidates = [task for task in self._task_bank if by_bucket(task)]
        if not candidates:
            candidates = list(self._task_bank)
        return self._rng.choice(candidates), bucket

    def _evaluate_response(self, task: dict, response_text: str, self_rating: float) -> tuple[bool, float]:
        response = (response_text or "").strip().lower()
        check = task.get("check", {"type": "keywords", "target": []})

        solved = False
        quality = 0.0
        if check["type"] == "exact":
            target = str(check["target"]).strip().lower()
            solved = target == response
            quality = 1.0 if solved else 0.0
        elif check["type"] == "keywords":
            targets = [str(item).lower() for item in check["target"]]
            hits = sum(1 for token in targets if token in response)
            quality = hits / max(1, len(targets))
            solved = quality >= 0.67

        confidence = sum(self._skill_graph[s]["confidence"] for s in task["skills"]) / len(task["skills"])
        solved_score = (0.7 * quality) + (0.15 * confidence) + (0.15 * self_rating)
        return solved, max(0.0, min(1.0, solved_score))

    def _update_skill(self, skill: str, solved: bool, solved_score: float) -> tuple[float, float, float]:
        node = self._skill_graph[skill]
        before = node["level"]
        if solved:
            delta = 0.05 + (0.05 * solved_score)
            node["streak"] += 1
            node["confidence"] = min(1.0, node["confidence"] + 0.04)
        else:
            delta = -0.03
            node["streak"] = 0
            node["confidence"] = max(0.0, node["confidence"] - 0.03)

        node["level"] = max(0.0, min(1.0, node["level"] + delta))
        drop_penalty = max(0.0, before - node["level"])
        return before, node["level"], drop_penalty

    def _snapshot(self) -> dict[str, dict[str, float]]:
        return {
            skill: {
                "level": round(data["level"], 4),
                "confidence": round(data["confidence"], 4),
                "streak": float(data["streak"]),
            }
            for skill, data in self._skill_graph.items()
        }

    def reset(self) -> SkillgraphAdaptiveObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._pending_task, bucket = self._select_task()
        return SkillgraphAdaptiveObservation(
            task_id=self._pending_task["id"],
            task_prompt=self._pending_task["prompt"],
            task_skills=self._pending_task["skills"],
            task_difficulty=self._pending_task["difficulty"],
            curriculum_bucket=bucket,
            success=False,
            reward=0.0,
            done=False,
            reward_breakdown={},
            skill_snapshot=self._snapshot(),
            metadata={"phase": "task_assigned", "history_size": len(self._task_history)},
        )

    def step(self, action: SkillgraphAdaptiveAction) -> SkillgraphAdaptiveObservation:  # type: ignore[override]
        self._state.step_count += 1
        task = self._pending_task or {"id": action.task_id, "prompt": "", "skills": ["reasoning"], "difficulty": 0.5}
        solved, solved_score = self._evaluate_response(task, action.response_text, action.self_rating)

        total_improvement = 0.0
        total_consistency = 0.0
        total_drop = 0.0
        for skill in task["skills"]:
            before, after, drop = self._update_skill(skill, solved, solved_score)
            total_improvement += max(0.0, after - before)
            total_drop += drop
            if solved:
                total_consistency += min(1.0, self._skill_graph[skill]["streak"] / 5.0)

        base_reward = 1.0 if solved else -0.4
        reward = (
            base_reward
            + (self._reward_weights["alpha"] * total_improvement)
            + (self._reward_weights["beta"] * total_consistency)
            - (self._reward_weights["gamma"] * total_drop)
        )
        reward = round(reward, 4)

        self._task_history.append(
            {
                "task_id": task["id"],
                "skills": task["skills"],
                "difficulty": task["difficulty"],
                "solved": solved,
                "reward": reward,
            }
        )

        return SkillgraphAdaptiveObservation(
            task_id=task["id"],
            task_prompt=task["prompt"],
            task_skills=task["skills"],
            task_difficulty=task["difficulty"],
            curriculum_bucket=self._curriculum_bucket(self._skill_graph[self._weakest_skill()]["level"]),
            success=solved,
            reward=reward,
            done=True,
            reward_breakdown={
                "task_score": round(base_reward, 4),
                "skill_improvement": round(total_improvement, 4),
                "consistency": round(total_consistency, 4),
                "skill_drop": round(total_drop, 4),
                "alpha": self._reward_weights["alpha"],
                "beta": self._reward_weights["beta"],
                "gamma": self._reward_weights["gamma"],
            },
            skill_snapshot=self._snapshot(),
            metadata={
                "step": self._state.step_count,
                "self_rating": action.self_rating,
                "response_text": action.response_text,
                "solved_score": round(solved_score, 4),
            },
        )

    @property
    def state(self) -> State:
        return self._state
