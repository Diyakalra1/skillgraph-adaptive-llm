# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SkillGraph adaptive environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SkillgraphAdaptiveAction, SkillgraphAdaptiveObservation


class SkillgraphAdaptiveEnv(
    EnvClient[SkillgraphAdaptiveAction, SkillgraphAdaptiveObservation, State]
):
    """
    Client for the Skillgraph Adaptive Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SkillgraphAdaptiveEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(SkillgraphAdaptiveAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SkillgraphAdaptiveEnv.from_docker_image("skillgraph_adaptive_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SkillgraphAdaptiveAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SkillgraphAdaptiveAction) -> Dict:
        """
        Convert SkillgraphAdaptiveAction to JSON payload for step message.

        Args:
            action: SkillgraphAdaptiveAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "task_id": action.task_id,
            "response_text": action.response_text,
            "self_rating": action.self_rating,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SkillgraphAdaptiveObservation]:
        """
        Parse server response into StepResult[SkillgraphAdaptiveObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SkillgraphAdaptiveObservation
        """
        obs_data = payload.get("observation", {})
        observation = SkillgraphAdaptiveObservation(
            task_id=obs_data.get("task_id", ""),
            task_prompt=obs_data.get("task_prompt", ""),
            task_skills=obs_data.get("task_skills", []),
            task_difficulty=obs_data.get("task_difficulty", 0.0),
            curriculum_bucket=obs_data.get("curriculum_bucket", ""),
            success=obs_data.get("success", False),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            skill_snapshot=obs_data.get("skill_snapshot", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
