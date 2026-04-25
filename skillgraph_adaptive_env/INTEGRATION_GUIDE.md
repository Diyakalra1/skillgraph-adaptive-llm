# Part 10: Create Your Own Integration

This project already follows the OpenEnv pattern.  
Use this 5-step checklist to wrap any new environment quickly.

## Step 1: Define Types (`models.py`)

Create action + observation classes.

```python
from typing import Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class YourAction(Action):
    action_value: int = Field(...)


class YourObservation(Observation):
    state_data: list[float] = Field(default_factory=list)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

In this project, see:
- `models.py`

## Step 2: Implement Environment (`server/your_environment.py`)

Implement `reset`, `step`, and `state`.

```python
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import YourAction, YourObservation


class YourEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> YourObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return YourObservation(state_data=[0.0], done=False, reward=0.0)

    def step(self, action: YourAction) -> YourObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = float(action.action_value) * 0.1
        return YourObservation(state_data=[reward], done=False, reward=reward)

    @property
    def state(self) -> State:
        return self._state
```

In this project, see:
- `server/skillgraph_adaptive_env_environment.py`

## Step 3: Create Client (`client.py`)

Convert action -> JSON payload and parse result/state back.

```python
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import YourAction, YourObservation


class YourEnv(EnvClient[YourAction, YourObservation, State]):
    def _step_payload(self, action: YourAction) -> Dict:
        return {"action_value": action.action_value}

    def _parse_result(self, payload: Dict) -> StepResult[YourObservation]:
        obs_data = payload.get("observation", {})
        observation = YourObservation(
            state_data=obs_data.get("state_data", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
```

In this project, see:
- `client.py`

## Step 4: Create Server App (`server/app.py`)

Use `create_app` once with environment + models.

```python
from openenv.core.env_server.http_server import create_app
from .your_environment import YourEnvironment
from ..models import YourAction, YourObservation

app = create_app(
    YourEnvironment,
    YourAction,
    YourObservation,
    env_name="your_env_name",
    max_concurrent_envs=1,
)
```

In this project, see:
- `server/app.py`

## Step 5: Dockerize (`server/Dockerfile`)

Minimal container for deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

In this project, see:
- `server/Dockerfile`

## Quick Validation Checklist

- `python -m py_compile models.py client.py server/your_environment.py server/app.py`
- Run local server: `uvicorn server.app:app --reload`
- Call `reset`, `step`, `state` once from your client
- Add one short training smoke test
- Confirm outputs appear in your run directory
