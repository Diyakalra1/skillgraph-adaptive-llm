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

1. **Dockerfile** — Replaced custom `pip` fallback with scaffold **two-step `uv sync`** so `.venv` is always created on HF.
2. **README** — Added `base_path: /web` like scaffold.
3. **pyproject** — Server deps only (`openenv-core[core]`); Streamlit/matplotlib moved to `[ui]` / `[training]` extras (faster Space build).

## Optional next step

Generate `uv.lock` for reproducible builds:

```bash
cd skillgraph_adaptive_env
uv lock
```

Then commit `uv.lock` and redeploy Space.
