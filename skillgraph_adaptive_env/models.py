# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the SkillGraph adaptive curriculum environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SkillgraphAdaptiveAction(Action):
    """Agent action for solving a skill-tagged task."""

    task_id: str = Field(..., description="Task identifier received in the observation")
    response_text: str = Field(default="", description="Optional natural language answer from agent")
    self_rating: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent self-assessed quality/confidence for this attempt (0-1)",
    )


class SkillgraphAdaptiveObservation(Observation):
    """Observation containing task, skill graph snapshot, and reward metadata."""

    task_id: str = Field(default="", description="Current task identifier")
    task_prompt: str = Field(default="", description="Task text shown to the agent")
    task_skills: list[str] = Field(default_factory=list, description="Skills required by this task")
    task_difficulty: float = Field(default=0.0, description="Task difficulty from 0 to 1")
    curriculum_bucket: str = Field(default="", description="easy, medium, hard, or mixed")
    success: bool = Field(default=False, description="Whether the task was solved")
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed reward terms: task, improvement, consistency, drop penalty",
    )
    skill_snapshot: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Current skill graph node values: level/confidence/streak",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra state and debug information")
