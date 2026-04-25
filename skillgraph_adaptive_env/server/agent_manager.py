"""Agent registry and team formation for AMASES."""

from __future__ import annotations

import random


class AgentManager:
    """Manages available agents and role-aware team matching."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._agents = {
            "agent_alpha": {"role": "planner", "collaboration_level": 0.7},
            "agent_beta": {"role": "debater", "collaboration_level": 0.6},
            "agent_gamma": {"role": "integrator", "collaboration_level": 0.8},
        }

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    def form_team(self, task: dict) -> list[str]:
        count = max(2, min(task.get("agent_count", 3), len(self._agents)))
        ids = self.agent_ids
        self._rng.shuffle(ids)
        return ids[:count]

    def simulated_response(self, agent_id: str, prompt: str, difficulty: float, rating: float) -> str:
        """Deterministic non-LLM simulator used by offline training scripts."""
        role = self._agents[agent_id]["role"]
        good = rating >= (difficulty / 5.0)
        prompt_l = prompt.lower()
        if "negotiate" in prompt_l or "budget" in prompt_l or "auction" in prompt_l:
            return (
                "Counter-offer: keep must-haves, concede on non-priority, proposal because constraints changed."
                if good
                else "I accept. proposal."
            )
        if "teach" in prompt_l or "transfer" in prompt_l:
            return (
                "Explain with example, then guided check: why this works and how to transfer."
                if good
                else "Example only."
            )
        if "debate" in prompt_l:
            return (
                "Claim with evidence, rebuttal to your premise, and conclusion with trade-off."
                if good
                else "I disagree."
            )
        if role == "planner":
            return "Plan with steps, constraints, risk timeline, and rationale." if good else "A short plan."
        return "Proposal with rationale that builds on prior context." if good else "Generic response."
