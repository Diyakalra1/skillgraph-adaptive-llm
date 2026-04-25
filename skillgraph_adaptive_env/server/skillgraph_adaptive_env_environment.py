"""AMASES environment: adaptive multi-agent skill evolution system."""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SkillgraphAdaptiveAction, SkillgraphAdaptiveObservation
    from .agent_manager import AgentManager
    from .curriculum_engine import CurriculumEngine
    from .interaction_memory import InteractionMemory
    from .scoring import compute_reward
    from .skill_graph import AgentSkillGraphManager
    from .task_library import TaskLibrary
except ImportError:
    from models import SkillgraphAdaptiveAction, SkillgraphAdaptiveObservation
    from server.agent_manager import AgentManager
    from server.curriculum_engine import CurriculumEngine
    from server.interaction_memory import InteractionMemory
    from server.scoring import compute_reward
    from server.skill_graph import AgentSkillGraphManager
    from server.task_library import TaskLibrary


class SkillgraphAdaptiveEnvironment(Environment):
    """Persistent multi-agent environment with dynamic skill graph curriculum."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_idx = 0
        self._agent_manager = AgentManager(seed=seed)
        self._task_library = TaskLibrary(seed=seed)
        self._curriculum = CurriculumEngine(self._task_library, seed=seed)
        self._skill_graph = AgentSkillGraphManager(self._agent_manager.agent_ids)
        self._memory = InteractionMemory(self._agent_manager.agent_ids)
        self._current_task: dict | None = None
        self._current_team: list[str] = []
        self._turn_index = 0
        self._task_history: list[dict] = []
        self._turn_texts: list[str] = []
        self._verification_alerts: list[dict] = []

    def _anchor_agent(self) -> str:
        """Select one agent to drive next curriculum decision."""
        return min(
            self._agent_manager.agent_ids,
            key=lambda aid: min(s["level"] for s in self._skill_graph.snapshot()[aid].values()),
        )

    def _evaluate_turn(self, task: dict, response: str) -> tuple[bool, float]:
        text = (response or "").lower()
        keys = task.get("check_keywords", [])
        hit_count = sum(1 for token in keys if token in text)
        quality = hit_count / max(1, len(keys))
        tier = str(task.get("difficulty_tier", "medium"))
        threshold = {"easy": 0.45, "medium": 0.55, "hard": 0.62}.get(tier, 0.55)
        solved = quality >= threshold
        return solved, quality

    def _build_observation(
        self,
        reward: float,
        solved: bool,
        reward_breakdown: dict[str, float],
        per_agent_reward: dict[str, float],
        done: bool,
    ) -> SkillgraphAdaptiveObservation:
        task = self._current_task or {}
        current_agent = self._current_team[min(self._turn_index, max(0, len(self._current_team) - 1))] if self._current_team else ""
        return SkillgraphAdaptiveObservation(
            task_id=task.get("id", ""),
            task_type=task.get("type", ""),
            task_prompt=task.get("prompt", ""),
            task_skills=task.get("skills_tested", []),
            task_difficulty=float(task.get("difficulty", 0.0)),
            curriculum_bucket=task.get("bucket", ""),
            current_agent_id=current_agent,
            turn_index=self._turn_index,
            max_turns=int(task.get("max_turns", 0)),
            team_agent_ids=list(self._current_team),
            success=solved,
            reward=reward,
            done=done,
            per_agent_reward=per_agent_reward,
            reward_breakdown=reward_breakdown,
            public_observation=self._memory.public_view(),
            private_observation=self._memory.private_view(current_agent) if current_agent else {},
            skill_snapshot=self._skill_graph.snapshot(),
            metadata={
                "history_size": len(self._task_history),
                "step": self._state.step_count,
                "episode": self._episode_idx,
                "target_skill": task.get("target_skill", ""),
                "is_diagnostic": bool(task.get("is_diagnostic", False)),
                "is_verification": bool(task.get("is_verification", False)),
                "verification_alerts": list(self._verification_alerts[-3:]),
            },
        )

    def reset(self) -> SkillgraphAdaptiveObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_idx += 1
        anchor = self._anchor_agent()
        task, bucket = self._curriculum.choose_task_for_episode(
            episode_idx=self._episode_idx,
            agent_skill_scores=self._skill_graph.snapshot(),
            anchor_agent_id=anchor,
        )
        task = dict(task)
        task["bucket"] = bucket
        task["max_turns"] = int(task.get("max_turns", task.get("agent_count", 3) * 3))
        self._current_task = task
        self._current_team = self._agent_manager.form_team(task)
        if task.get("is_verification"):
            self._current_team = [anchor]
        self._memory.reset(
            task_prompt=task["prompt"],
            task_type=task.get("type", ""),
            team_agent_ids=self._current_team,
        )
        self._turn_index = 0
        self._turn_texts = []
        return self._build_observation(
            reward=0.0,
            solved=False,
            reward_breakdown={},
            per_agent_reward={agent: 0.0 for agent in self._current_team},
            done=False,
        )

    def step(self, action: SkillgraphAdaptiveAction) -> SkillgraphAdaptiveObservation:  # type: ignore[override]
        self._state.step_count += 1
        if not self._current_task:
            self.reset()
        assert self._current_task is not None

        expected_agent = self._current_team[self._turn_index % len(self._current_team)]
        actor = action.agent_id or expected_agent
        if action.response_text:
            self._memory.add_public(
                turn=self._turn_index + 1,
                agent_id=actor,
                content=action.response_text[:220],
            )
        self._turn_texts.append(action.response_text or "")
        solved, quality = self._evaluate_turn(self._current_task, action.response_text)
        tested_skills = list(self._current_task.get("skills_tested", []))
        max_turns = int(self._current_task.get("max_turns", 9))
        outcome = {
            "agreement_reached": solved,
            "turns_used": self._turn_index + 1,
            "max_turns": max_turns,
            "quality": quality,
        }
        reward_result = compute_reward(
            task_type=self._current_task.get("type", ""),
            task_skills=tested_skills,
            turn_idx=self._turn_index + 1,
            max_turns=max_turns,
            current_text=action.response_text or "",
            turn_texts=self._turn_texts,
            context_refs=self._current_task.get("check_keywords", []),
            self_rating=float(action.self_rating),
            outcome=outcome,
        )
        reward = reward_result.scalar
        derived_skills = list(reward_result.skill_vector.keys()) or tested_skills
        improvement, consistency, drop = self._skill_graph.update(
            actor, derived_skills, reward_result.skill_vector, solved
        )
        if action.merged_reward_override is not None:
            # Allow external rubric+judge reward to drive training runs.
            reward = float(action.merged_reward_override)
        reward = round(reward, 4)
        self._memory.add_private(
            turn=self._turn_index + 1,
            agent_id="system",
            target_agent_id=actor,
            content=f"Turn quality={quality:.2f}, solved={solved}, reward={reward:.3f}",
        )
        self._turn_index += 1
        done = solved or self._turn_index >= max_turns

        per_agent_reward = {agent: 0.0 for agent in self._current_team}
        per_agent_reward[actor] = reward
        reward_breakdown = {
            "task_success": round(reward_result.rubric["task_success"], 4),
            "skill_demonstration": round(improvement, 4),
            "learning_evidence": round(reward_result.rubric["learning_evidence"], 4),
            "collab_quality": round(reward_result.rubric["collab_quality"], 4),
            "meta_cognition": round(reward_result.rubric["meta_cognition"], 4),
            "skill_drop_penalty": round(drop, 4),
            "confidence_gain": round(consistency, 4),
            "penalties_total": round(sum(reward_result.penalties.values()), 4),
        }
        reward_breakdown.update({f"penalty_{k}": round(v, 4) for k, v in reward_result.penalties.items()})

        if self._current_task.get("is_verification"):
            if reward < 0.35:
                self._verification_alerts.append(
                    {
                        "episode": self._episode_idx,
                        "agent": actor,
                        "task_id": self._current_task["id"],
                        "reward": reward,
                        "signal": "possible_reward_hacking_drop",
                    }
                )

        self._task_history.append(
            {
                "task_id": self._current_task["id"],
                "agent_id": actor,
                "turn": self._turn_index,
                "done": done,
                "reward": reward,
                "solved": solved,
                "bucket": self._current_task.get("bucket", ""),
                "verification": bool(self._current_task.get("is_verification", False)),
            }
        )
        return self._build_observation(
            reward=reward,
            solved=solved,
            reward_breakdown=reward_breakdown,
            per_agent_reward=per_agent_reward,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state
