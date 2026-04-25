"""Per-agent skill graph tracking for AMASES."""

from __future__ import annotations

from dataclasses import dataclass, field


BASE_SKILLS: tuple[str, ...] = (
    "negotiation",
    "collaboration",
    "strategic_reasoning",
    "information_synthesis",
    "communication",
    "meta_learning",
)


@dataclass
class SkillNode:
    level: float = 2.5
    confidence: float = 0.1
    streak: int = 0
    history: list[float] = field(default_factory=lambda: [2.5])
    learning_velocity: float = 0.0
    plateau: bool = False


class AgentSkillGraphManager:
    """Tracks and updates skill states for each agent."""

    def __init__(self, agent_ids: list[str]) -> None:
        self._graphs: dict[str, dict[str, SkillNode]] = {
            agent_id: self._init_graph(agent_id) for agent_id in agent_ids
        }

    def _init_graph(self, agent_id: str) -> dict[str, SkillNode]:
        offset = {"agent_alpha": 0.00, "agent_beta": -0.05, "agent_gamma": 0.05}.get(agent_id, 0.0)
        return {skill: SkillNode(level=max(0.0, min(5.0, 2.5 + offset)), history=[2.5 + offset]) for skill in BASE_SKILLS}

    def _ensure_skill(self, agent_id: str, skill: str) -> None:
        if skill not in self._graphs[agent_id]:
            self._graphs[agent_id][skill] = SkillNode()

    def weakest_skills(self, agent_id: str, top_n: int = 3) -> list[str]:
        items = sorted(
            self._graphs[agent_id].items(),
            key=lambda kv: kv[1].level * max(kv[1].confidence, 0.15),
        )
        return [skill for skill, _ in items[:top_n]]

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            agent_id: {
                skill: {
                    "level": round(node.level, 4),
                    "confidence": round(node.confidence, 4),
                    "streak": float(node.streak),
                    "learning_velocity": round(node.learning_velocity, 4),
                    "plateau": float(1.0 if node.plateau else 0.0),
                }
                for skill, node in graph.items()
            }
            for agent_id, graph in self._graphs.items()
        }

    def update(
        self,
        agent_id: str,
        tested_skills: list[str],
        skill_rewards: dict[str, float],
        solved: bool,
    ) -> tuple[float, float, float]:
        alpha = 0.1
        improvement = 0.0
        confidence_gain = 0.0
        drop = 0.0
        for skill in tested_skills:
            self._ensure_skill(agent_id, skill)
            node = self._graphs[agent_id][skill]
            before = node.level
            reward_score = max(0.0, min(1.0, float(skill_rewards.get(skill, 0.0))))
            target_level = reward_score * 5.0
            node.level = max(0.0, min(5.0, (1 - alpha) * node.level + alpha * target_level))
            if solved and reward_score >= 0.5:
                node.streak += 1
            else:
                node.streak = 0
            node.history.append(node.level)
            n = len(node.history)
            node.confidence = min(1.0, n / 20.0)
            confidence_gain += 0.05 if n <= 20 else 0.0
            if n >= 20:
                node.learning_velocity = (node.history[-1] - node.history[-20]) / 20.0
            elif n >= 2:
                node.learning_velocity = node.history[-1] - node.history[-2]
            if n >= 10:
                window_delta = abs(node.history[-1] - node.history[-10])
                node.plateau = window_delta < 0.05
            improvement += max(0.0, node.level - before)
            drop += max(0.0, before - node.level)
        return improvement, confidence_gain, drop
