# OpenEnv structure alignment

Compared against `openenv init skillgraph_dummy` (reference in repo `_openenv_reference/`).

## Required layout (present)

| Path | Status |
|------|--------|
| `models.py` | OK |
| `client.py` | OK |
| `openenv.yaml` | OK (`port: 8000`) |
| `README.md` | OK (`sdk: docker`, `app_port: 8000`, `base_path: /web`) |
| `server/app.py` | OK |
| `server/*_environment.py` | OK |
| `server/Dockerfile` | OK (matches scaffold `uv sync` flow) |
| `Dockerfile` (Space root) | OK (same as `server/Dockerfile`) |
| `pyproject.toml` | OK |

## Extra (hackathon; not required on Space)

- `training/` — GRPO / three-model scripts
- `ui/` — Streamlit dashboard (`pip install -e ".[ui]"`)
- `blogpost.md`, `INTEGRATION_GUIDE.md`

## Fixes applied vs broken build

1. **Dockerfile** — `uv sync --no-install-project` only (no second install step). Server code runs via `PYTHONPATH=/app/env`.
2. **pyproject packages** — Only `skillgraph_adaptive_env` + `.server` (like `openenv init`). No `ui` package (`.dockerignore` excludes `ui/`).
3. **README** — `app_port: 8000`, `base_path: /web`.

## Optional next step

Generate `uv.lock` for reproducible builds:

```bash
cd skillgraph_adaptive_env
uv lock
```

Then commit `uv.lock` and redeploy Space.
