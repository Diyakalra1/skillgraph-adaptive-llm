# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the AMASES multi-agent adaptive curriculum environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SkillgraphAdaptiveAction(Action):
    """Agent action for one turn in a multi-agent task."""

    agent_id: str = Field(..., description="Agent taking the current turn")
    task_id: str = Field(..., description="Task identifier received in the observation")
    response_text: str = Field(default="", description="Optional natural language answer from agent")
    self_rating: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent self-assessed quality/confidence for this attempt (0-1)",
    )
    merged_reward_override: float | None = Field(
        default=None,
        description="Optional external merged reward from rubric+judge scoring.",
    )


class SkillgraphAdaptiveObservation(Observation):
    """Observation containing task arena state, per-agent skills, and reward metadata."""

    task_id: str = Field(default="", description="Current task identifier")
    task_type: str = Field(default="", description="Task category: collaborative/competitive/mixed_motive/peer_teaching/debate")
    task_prompt: str = Field(default="", description="Task text shown to the agent")
    task_skills: list[str] = Field(default_factory=list, description="Skills required by this task")
    task_difficulty: float = Field(default=0.0, description="Task difficulty from 0 to 1")
    curriculum_bucket: str = Field(default="", description="easy, medium, hard, mixed, or balanced")
    current_agent_id: str = Field(default="", description="Agent who should act now")
    turn_index: int = Field(default=0, description="Current turn index in this task")
    max_turns: int = Field(default=0, description="Maximum turns for this task")
    team_agent_ids: list[str] = Field(default_factory=list, description="Agents in current task")
    success: bool = Field(default=False, description="Whether the task was solved")
    per_agent_reward: dict[str, float] = Field(default_factory=dict, description="Reward contribution by agent")
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed reward terms: task, improvement, consistency, drop penalty",
    )
    public_observation: dict[str, Any] = Field(
        default_factory=dict,
        description="Shared view of arena state and public messages visible to all agents",
    )
    private_observation: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific hidden notes and private context for current agent",
    )
    skill_snapshot: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict,
        description="Per-agent skill graph values by skill: level/confidence/streak",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra state and debug information")


class SkillgraphAdaptiveState(State):
    """Typed environment state for integration-pattern compatibility."""

    episode_id: str | None = Field(default=None, description="Current episode identifier")
    step_count: int = Field(default=0, description="Current environment step count")
