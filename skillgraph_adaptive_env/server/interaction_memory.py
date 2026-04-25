"""Interaction memory for AMASES multi-agent tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryEvent:
    turn: int
    agent_id: str
    visibility: str  # public | private
    content: str
    timestamp: str
    target_agent_id: str | None = None


class InteractionMemory:
    """Stores public and private messages and builds agent views."""

    def __init__(self, agent_ids: list[str]) -> None:
        self._agent_ids = list(agent_ids)
        self._events: list[MemoryEvent] = []
        self._private_notes: dict[str, list[str]] = {aid: [] for aid in self._agent_ids}
        self._task_type: str = ""
        self._secret_preferences: dict[str, str] = {}

    def reset(self, task_prompt: str, task_type: str, team_agent_ids: list[str]) -> None:
        self._events = []
        self._private_notes = {aid: [] for aid in self._agent_ids}
        self._task_type = task_type
        self._secret_preferences = {}
        self.add_public(
            turn=0,
            agent_id="system",
            content=f"Task started: {task_prompt}",
        )
        for aid in team_agent_ids:
            # Private preference masking by task type.
            if task_type == "competitive":
                pref = f"{aid} prefers maximizing own utility under budget limits."
            elif task_type == "mixed_motive":
                pref = f"{aid} prefers shared success + favorable individual allocation."
            elif task_type == "debate":
                pref = f"{aid} is assigned a private stance to defend strongly."
            else:
                pref = f"{aid} should optimize clarity, teamwork, and measurable progress."
            self._secret_preferences[aid] = pref
            self.add_private(
                turn=0,
                agent_id="system",
                target_agent_id=aid,
                content=f"Private goal for {aid}: {pref}",
            )

    def add_public(self, turn: int, agent_id: str, content: str) -> None:
        self._events.append(
            MemoryEvent(
                turn=turn,
                agent_id=agent_id,
                visibility="public",
                content=content,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )

    def add_private(self, turn: int, agent_id: str, target_agent_id: str, content: str) -> None:
        self._events.append(
            MemoryEvent(
                turn=turn,
                agent_id=agent_id,
                visibility="private",
                content=content,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                target_agent_id=target_agent_id,
            )
        )
        self._private_notes.setdefault(target_agent_id, []).append(content)

    def public_view(self, limit: int = 6) -> dict:
        events = [e for e in self._events if e.visibility == "public"][-limit:]
        return {
            "recent_public_messages": [
                {
                    "turn": e.turn,
                    "agent_id": e.agent_id,
                    "content": e.content,
                    "timestamp": e.timestamp,
                }
                for e in events
            ],
            "total_messages": len(self._events),
        }

    def private_view(self, agent_id: str, limit: int = 5) -> dict:
        personal = [
            e
            for e in self._events
            if e.visibility == "private" and e.target_agent_id == agent_id
        ][-limit:]
        own_public = [
            e for e in self._events if e.visibility == "public" and e.agent_id == agent_id
        ][-limit:]
        task_rules = {
            "collaborative": "All high-level plans are public. Private notes can include execution hints.",
            "peer_teaching": "Teacher/learner feedback remains private per agent until task ends.",
            "competitive": "Opponent private preferences are hidden from you.",
            "mixed_motive": "Shared objective is public; private utilities are hidden.",
            "debate": "Assigned stance and rebuttal strategy are private.",
        }
        masked_pref = self._secret_preferences.get(agent_id, "")
        return {
            "agent_id": agent_id,
            "task_type": self._task_type,
            "task_masking_rule": task_rules.get(self._task_type, "Default private masking applies."),
            "visible_private_preference": masked_pref,
            "private_notes": list(self._private_notes.get(agent_id, []))[-limit:],
            "recent_private_messages": [
                {"turn": e.turn, "from": e.agent_id, "content": e.content}
                for e in personal
            ],
            "recent_own_public_messages": [
                {"turn": e.turn, "content": e.content}
                for e in own_public
            ],
        }
