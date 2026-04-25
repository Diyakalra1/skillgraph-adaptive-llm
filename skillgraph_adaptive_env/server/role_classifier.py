"""Role classification across iterations."""

from __future__ import annotations

from dataclasses import dataclass, field


ROLES = ("planner", "negotiator", "teacher")


@dataclass
class RoleClassifier:
    cumulative: dict[str, dict[str, float]] = field(default_factory=dict)

    def classify_iteration(self, per_agent_role_scores: dict[str, dict[str, float]]) -> dict[str, str]:
        winners: dict[str, str] = {}
        for role in ROLES:
            best_agent = max(
                per_agent_role_scores.keys(),
                key=lambda aid: per_agent_role_scores.get(aid, {}).get(role, 0.0),
            )
            winners[role] = best_agent

        for agent_id, role_map in per_agent_role_scores.items():
            self.cumulative.setdefault(agent_id, {r: 0.0 for r in ROLES})
            for role in ROLES:
                self.cumulative[agent_id][role] += role_map.get(role, 0.0)
        return winners

    def final_classification(self, iterations: int) -> dict[str, dict]:
        final_winners: dict[str, str] = {}
        confidences: dict[str, float] = {}
        for role in ROLES:
            best_agent = max(self.cumulative.keys(), key=lambda aid: self.cumulative[aid][role])
            final_winners[role] = best_agent
            top = self.cumulative[best_agent][role]
            total = sum(scores[role] for scores in self.cumulative.values()) or 1.0
            confidences[role] = round(top / total, 4)
        normalized = {
            aid: {role: round(score / max(1, iterations), 4) for role, score in roles.items()}
            for aid, roles in self.cumulative.items()
        }
        return {
            "iterations": iterations,
            "winners": final_winners,
            "confidence": confidences,
            "average_role_scores": normalized,
        }
