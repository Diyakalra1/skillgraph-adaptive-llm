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

    def simulated_response(
        self,
        agent_id: str,
        prompt: str,
        difficulty: float,
        rating: float,
        *,
        task_type: str = "collaborative",
        check_keywords: list[str] | None = None,
    ) -> str:
        """Deterministic policy for offline rollouts; injects task keywords when rating is high."""
        role = self._agents[agent_id]["role"]
        good = rating >= max(0.42, difficulty / 5.5)
        prompt_l = prompt.lower()
        keywords = check_keywords or []
        kw = " ".join(keywords[:6]) if keywords else "evidence synthesis trade-off summary"

        if good and keywords:
            if role == "planner":
                return (
                    f"Plan: gather {kw} from sources. Step 1: extract evidence. "
                    f"Step 2: note one trade-off. Step 3: shared summary outline."
                )
            if role == "debater":
                return (
                    f"I challenge the plan: {kw} shows risk, but trade-off on budget. "
                    "Synthesis should combine reports before recommendation."
                )
            return (
                f"Final synthesis: {kw} supports one shared recommendation. "
                "Summary: phased rollout with evidence and clear trade-off rationale."
            )

        if "negotiate" in prompt_l or "budget" in prompt_l or "auction" in prompt_l or task_type == "competitive":
            return (
                "Counter-offer: keep must-haves, concede on non-priority, proposal because constraints changed. "
                "Evidence and trade-off noted."
                if good
                else "I accept. proposal."
            )
        if "teach" in prompt_l or "transfer" in prompt_l or task_type == "peer_teaching":
            return (
                "Explain with example, clarify because, lesson transfer with evidence."
                if good
                else "Example only."
            )
        if "debate" in prompt_l or task_type == "debate":
            return (
                "Claim with evidence, rebuttal to your premise, and conclusion with trade-off."
                if good
                else "I disagree."
            )
        if role == "planner":
            return (
                "Plan with steps, constraints, risk timeline, and rationale with evidence."
                if good
                else "A short plan."
            )
        return (
            "Proposal with rationale that builds on prior context and synthesis summary."
            if good
            else "Generic response."
        )
