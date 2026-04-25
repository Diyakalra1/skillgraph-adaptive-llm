"""Task definitions for AMASES multi-agent interactions."""

from __future__ import annotations

import random
from copy import deepcopy


TASK_TYPES = (
    "collaborative",
    "competitive",
    "mixed_motive",
    "peer_teaching",
    "debate",
)


class TaskLibrary:
    """Fixed 15-task library with deterministic structure and randomized surface."""

    def __init__(self, seed: int = 7) -> None:
        self._seed = seed
        self._tasks: list[dict] = self._build_tasks()
        self._by_id = {task["id"]: task for task in self._tasks}

    def all_tasks(self) -> list[dict]:
        return [deepcopy(task) for task in self._tasks]

    def get_task(self, task_id: str) -> dict:
        return deepcopy(self._by_id[task_id])

    def instantiate_task(self, task_id: str, episode_idx: int = 0) -> dict:
        """Instantiate one task by randomizing only surface slots."""
        template = self.get_task(task_id)
        rng = random.Random(self._seed + (episode_idx * 7919) + abs(hash(task_id)) % 100_000)
        slots = template.get("surface_pools", {})
        filled: dict[str, str] = {}
        for key, values in slots.items():
            filled[key] = values[rng.randrange(len(values))]
        template["surface"] = filled
        template["prompt"] = template["prompt_template"].format(**filled)
        private_ctx = template.get("private_context_rules", {})
        if template["type"] == "competitive":
            total = int(filled.get("budget", "1200"))
            floor = rng.randint(300, max(301, total // 2))
            private_ctx = {
                **private_ctx,
                "secret_floor_price": floor,
                "hidden_valuation": total - floor,
            }
        template["private_context"] = private_ctx
        return template

    def fixed_diagnostic_variant(self, task_type: str) -> dict:
        """Return fixed medium variant used only in cold-start checks."""
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type={task_type}")
        task_id = f"{task_type}_medium"
        template = self.get_task(task_id)
        default_surface = {k: values[0] for k, values in template.get("surface_pools", {}).items()}
        template["surface"] = default_surface
        template["prompt"] = template["prompt_template"].format(**default_surface)
        template["is_diagnostic"] = True
        template["diagnostic_id"] = f"D_{task_type}"
        return template

    def _build_tasks(self) -> list[dict]:
        return [
            # Collaborative: Task 1-3
            {
                "id": "collaborative_easy",
                "name": "Task1_EasyJointPlanning",
                "type": "collaborative",
                "difficulty_tier": "easy",
                "difficulty": 2.0,
                "agents_needed": 2,
                "agent_count": 2,
                "max_turns": 8,
                "reward_mode": "shared",
                "skills_tested": ["collaboration", "problem_decomposition", "communication_clarity"],
                "rules": ["joint_plan_required", "explicit_constraints", "turn_balanced"],
                "rubric_spec": "collaboration_v1",
                "check_keywords": ["plan", "step", "constraint", "timeline"],
                "prompt_template": "Create a joint plan for {project} in {city} with budget {budget}.",
                "surface_pools": {
                    "project": ["school wifi rollout", "community solar pilot", "flood alert setup"],
                    "city": ["Pune", "Indore", "Nagpur"],
                    "budget": ["900", "1100", "1300"],
                },
                "private_context_rules": {"mode": "shared_goals"},
            },
            {
                "id": "collaborative_medium",
                "name": "Task2_MediumResearchSynthesis",
                "type": "collaborative",
                "difficulty_tier": "medium",
                "difficulty": 3.0,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 12,
                "reward_mode": "shared",
                "skills_tested": ["information_synthesis", "collaboration", "communication_clarity"],
                "rules": ["partial_info", "tradeoff_required", "joint_summary"],
                "rubric_spec": "collaboration_v2",
                "check_keywords": ["evidence", "synthesis", "trade-off", "summary"],
                "prompt_template": "Synthesize findings from {sources} about {topic} and propose one shared recommendation for {city}.",
                "surface_pools": {
                    "sources": ["3 reports", "4 memos", "5 interview notes"],
                    "topic": ["public transport safety", "water leakage reduction", "clinic queue optimization"],
                    "city": ["Kochi", "Jaipur", "Surat"],
                },
                "private_context_rules": {"mode": "shared_goals"},
            },
            {
                "id": "collaborative_hard",
                "name": "Task3_HardMultiDocIntegration",
                "type": "collaborative",
                "difficulty_tier": "hard",
                "difficulty": 4.4,
                "agents_needed": 4,
                "agent_count": 3,
                "max_turns": 15,
                "reward_mode": "shared",
                "skills_tested": ["information_synthesis", "contradiction_resolution", "strategic_reasoning"],
                "rules": ["hidden_info", "coalition_dynamics", "near_pareto_solution"],
                "rubric_spec": "collaboration_v3",
                "check_keywords": ["contradiction", "evidence", "integrate", "consensus"],
                "prompt_template": "Integrate contradictory documents on {topic} for {region} and finalize a near-optimal execution plan.",
                "surface_pools": {
                    "topic": ["disaster logistics", "renewable grid stability", "district education reform"],
                    "region": ["Zone-A", "Zone-B", "Zone-C"],
                },
                "private_context_rules": {"mode": "partial_private_constraints"},
            },
            # Competitive: Task 4-6
            {
                "id": "competitive_easy",
                "name": "Task4_EasyBudgetSplit",
                "type": "competitive",
                "difficulty_tier": "easy",
                "difficulty": 2.2,
                "agents_needed": 2,
                "agent_count": 2,
                "max_turns": 8,
                "reward_mode": "zero_sum",
                "skills_tested": ["competitive_strategy", "opponent_modeling", "negotiation"],
                "rules": ["private_utility", "counter_offer_expected"],
                "rubric_spec": "negotiation_v1",
                "check_keywords": ["offer", "counter", "constraint", "split"],
                "prompt_template": "Negotiate a compute budget split of {budget} credits for {project}.",
                "surface_pools": {
                    "budget": ["1000", "1200", "1400"],
                    "project": ["evaluation pipeline", "vision benchmark", "search indexing"],
                },
                "private_context_rules": {"mode": "private_valuations"},
            },
            {
                "id": "competitive_medium",
                "name": "Task5_MediumResourceAuction",
                "type": "competitive",
                "difficulty_tier": "medium",
                "difficulty": 3.1,
                "agents_needed": 2,
                "agent_count": 2,
                "max_turns": 12,
                "reward_mode": "zero_sum",
                "skills_tested": ["competitive_strategy", "risk_assessment", "opponent_modeling"],
                "rules": ["partial_info", "auction_style_negotiation", "concession_tracking"],
                "rubric_spec": "negotiation_v2",
                "check_keywords": ["bid", "value", "counter", "trade-off"],
                "prompt_template": "Run a two-agent resource auction for {resource_units} units under total cap {budget}.",
                "surface_pools": {
                    "resource_units": ["8", "10", "12"],
                    "budget": ["900", "1100", "1300"],
                },
                "private_context_rules": {"mode": "private_valuations"},
            },
            {
                "id": "competitive_hard",
                "name": "Task6_HardZeroSumNegotiation",
                "type": "competitive",
                "difficulty_tier": "hard",
                "difficulty": 4.5,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 15,
                "reward_mode": "zero_sum",
                "skills_tested": ["competitive_strategy", "nash_reasoning", "deception_detection"],
                "rules": ["hidden_info", "coalition_pressure", "strict_win_loss"],
                "rubric_spec": "negotiation_v3",
                "check_keywords": ["reserve", "counter", "credible", "utility"],
                "prompt_template": "Zero-sum negotiation over {contract} with capped budget {budget} and hidden reservation values.",
                "surface_pools": {
                    "contract": ["GPU time", "data labeling quota", "edge compute slots"],
                    "budget": ["1300", "1500", "1700"],
                },
                "private_context_rules": {"mode": "private_valuations"},
            },
            # Mixed motive: Task 7-9
            {
                "id": "mixed_motive_easy",
                "name": "Task7_EasyPartialCoop",
                "type": "mixed_motive",
                "difficulty_tier": "easy",
                "difficulty": 2.4,
                "agents_needed": 2,
                "agent_count": 2,
                "max_turns": 8,
                "reward_mode": "partial",
                "skills_tested": ["collaboration", "strategic_reasoning", "negotiation"],
                "rules": ["shared_goal_plus_private_bonus"],
                "rubric_spec": "mixed_v1",
                "check_keywords": ["shared", "bonus", "plan", "agreement"],
                "prompt_template": "Partial-cooperation task: deliver {milestone} while optimizing your private share of bonus {bonus}.",
                "surface_pools": {
                    "milestone": ["prototype demo", "customer report", "security checklist"],
                    "bonus": ["200", "250", "300"],
                },
                "private_context_rules": {"mode": "shared_plus_private_bonus"},
            },
            {
                "id": "mixed_motive_medium",
                "name": "Task8_MediumStartupSim",
                "type": "mixed_motive",
                "difficulty_tier": "medium",
                "difficulty": 3.3,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 12,
                "reward_mode": "partial",
                "skills_tested": ["strategic_reasoning", "coalition_building", "risk_assessment"],
                "rules": ["partial_info", "equity_tradeoffs", "ship_deadline"],
                "rubric_spec": "mixed_v2",
                "check_keywords": ["equity", "milestone", "risk", "coalition"],
                "prompt_template": "Startup simulation in {market}: negotiate equity split and launch plan before {deadline}.",
                "surface_pools": {
                    "market": ["health-tech", "agri-tech", "fintech"],
                    "deadline": ["6 weeks", "8 weeks", "10 weeks"],
                },
                "private_context_rules": {"mode": "shared_plus_private_bonus"},
            },
            {
                "id": "mixed_motive_hard",
                "name": "Task9_HardCoalitionGame",
                "type": "mixed_motive",
                "difficulty_tier": "hard",
                "difficulty": 4.6,
                "agents_needed": 4,
                "agent_count": 3,
                "max_turns": 15,
                "reward_mode": "partial",
                "skills_tested": ["coalition_building", "strategic_reasoning", "long_term_planning"],
                "rules": ["hidden_info", "coalition_dynamics", "near_pareto_solution"],
                "rubric_spec": "mixed_v3",
                "check_keywords": ["coalition", "contingency", "trade-off", "allocation"],
                "prompt_template": "Coalition game: align {teams} factions on allocation of {budget} while preserving long-term alliance stability.",
                "surface_pools": {
                    "teams": ["3", "4", "5"],
                    "budget": ["1800", "2100", "2400"],
                },
                "private_context_rules": {"mode": "shared_plus_private_bonus"},
            },
            # Peer teaching: Task 10-12
            {
                "id": "peer_teaching_easy",
                "name": "Task10_EasyConceptExplain",
                "type": "peer_teaching",
                "difficulty_tier": "easy",
                "difficulty": 2.1,
                "agents_needed": 2,
                "agent_count": 2,
                "max_turns": 8,
                "reward_mode": "linked",
                "skills_tested": ["communication_clarity", "meta_learning", "knowledge_transfer"],
                "rules": ["teacher_learner_roles", "short_transfer_check"],
                "rubric_spec": "teaching_v1",
                "check_keywords": ["explain", "example", "why", "check"],
                "prompt_template": "Teacher explains {concept} with one example, learner answers a short transfer check.",
                "surface_pools": {
                    "concept": ["recursion", "binary search", "backtracking"],
                },
                "private_context_rules": {"mode": "teacher_learner_private_feedback"},
            },
            {
                "id": "peer_teaching_medium",
                "name": "Task11_MediumGuidedPractice",
                "type": "peer_teaching",
                "difficulty_tier": "medium",
                "difficulty": 3.0,
                "agents_needed": 2,
                "agent_count": 2,
                "max_turns": 12,
                "reward_mode": "linked",
                "skills_tested": ["communication_clarity", "skill_transfer", "strategy_reflection"],
                "rules": ["guided_practice", "mistake_recovery", "feedback_loop"],
                "rubric_spec": "teaching_v2",
                "check_keywords": ["hint", "feedback", "revise", "transfer"],
                "prompt_template": "Guided practice on {concept}: teacher gives hints, learner revises and solves transfer variant {variant}.",
                "surface_pools": {
                    "concept": ["dynamic programming", "graph traversal", "probability basics"],
                    "variant": ["A", "B", "C"],
                },
                "private_context_rules": {"mode": "teacher_learner_private_feedback"},
            },
            {
                "id": "peer_teaching_hard",
                "name": "Task12_HardTransferTest",
                "type": "peer_teaching",
                "difficulty_tier": "hard",
                "difficulty": 4.2,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 15,
                "reward_mode": "linked",
                "skills_tested": ["meta_learning", "knowledge_transfer", "communication_clarity"],
                "rules": ["hidden_misconception", "hard_transfer", "teaching_adaptation"],
                "rubric_spec": "teaching_v3",
                "check_keywords": ["misconception", "adapt", "transfer", "explain"],
                "prompt_template": "Hard transfer test: teach {concept} and adapt strategy after diagnosing hidden misconception {misconception}.",
                "surface_pools": {
                    "concept": ["optimization constraints", "causal inference basics", "multi-agent planning"],
                    "misconception": ["greedy always optimal", "correlation implies causation", "single objective only"],
                },
                "private_context_rules": {"mode": "teacher_learner_private_feedback"},
            },
            # Debate: Task 13-15
            {
                "id": "debate_easy",
                "name": "Task13_EasyStructuredDebate",
                "type": "debate",
                "difficulty_tier": "easy",
                "difficulty": 2.6,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 8,
                "reward_mode": "judge_scored",
                "skills_tested": ["argumentation", "information_synthesis", "communication_clarity"],
                "rules": ["claim_evidence_rebuttal", "strict_turn_order"],
                "rubric_spec": "debate_v1",
                "check_keywords": ["claim", "evidence", "rebuttal", "conclusion"],
                "prompt_template": "Structured debate on {topic}. Side A and Side B present claim-evidence-rebuttal; judge scores.",
                "surface_pools": {
                    "topic": ["remote work policy", "public transit subsidy", "school device funding"],
                },
                "private_context_rules": {"mode": "private_stance_assignment"},
            },
            {
                "id": "debate_medium",
                "name": "Task14_MediumPolicyDebate",
                "type": "debate",
                "difficulty_tier": "medium",
                "difficulty": 3.4,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 12,
                "reward_mode": "judge_scored",
                "skills_tested": ["argumentation", "information_synthesis", "adaptive_strategy"],
                "rules": ["partial_info", "policy_tradeoffs", "rebuttal_depth"],
                "rubric_spec": "debate_v2",
                "check_keywords": ["policy", "trade-off", "counterexample", "evidence"],
                "prompt_template": "Policy debate for {city}: defend opposing approaches to {topic} with evidence and counterarguments.",
                "surface_pools": {
                    "city": ["Bengaluru", "Chennai", "Lucknow"],
                    "topic": ["congestion pricing", "water rationing", "air quality regulation"],
                },
                "private_context_rules": {"mode": "private_stance_assignment"},
            },
            {
                "id": "debate_hard",
                "name": "Task15_HardAdversarialRebuttal",
                "type": "debate",
                "difficulty_tier": "hard",
                "difficulty": 4.8,
                "agents_needed": 3,
                "agent_count": 3,
                "max_turns": 15,
                "reward_mode": "judge_scored",
                "skills_tested": ["argumentation", "adaptive_strategy", "adversarial_rebuttal"],
                "rules": ["hidden_info", "adversarial_pressure", "fixed_judge_prompt"],
                "rubric_spec": "debate_v3_judge",
                "check_keywords": ["premise", "rebuttal", "evidence", "logic"],
                "prompt_template": "Adversarial rebuttal debate on {topic} with hidden assumptions and strict judge rubric.",
                "surface_pools": {
                    "topic": ["AI model transparency", "autonomous vehicle liability", "algorithmic hiring audits"],
                },
                "private_context_rules": {"mode": "private_stance_assignment"},
            },
        ]
