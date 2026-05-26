# Copyright (c) Meta Platforms, Inc. and affiliates.
# Custom Gradio UI for AMASES on Hugging Face Spaces.

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Type

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation
from openenv.core.env_server.web_interface import (
    WebInterfaceManager,
    get_quick_start_markdown,
    load_environment_metadata,
)

AGENT_CHOICES = ["agent_alpha", "agent_beta", "agent_gamma"]

AMASES_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill_dark="#0f172a",
    background_fill_primary_dark="#111827",
    background_fill_secondary_dark="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color_dark="#334155",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
)

AMASES_CSS = """
.amases-hero {
    background: linear-gradient(135deg, #312e81 0%, #1e3a8a 55%, #0f172a 100%);
    border: 1px solid #475569;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    color: #e2e8f0;
}
.amases-hero h1 { margin: 0 0 6px 0; font-size: 1.55rem; color: #f8fafc; }
.amases-hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; }
.amases-panel {
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.amases-metric {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
"""


def _format_amases_observation(data: Dict[str, Any]) -> str:
    obs = data.get("observation") or {}
    if not isinstance(obs, dict):
        return "*No observation*"

    lines = [
        "### Episode status",
        f"- **Reward:** `{data.get('reward', obs.get('reward', '—'))}`",
        f"- **Done:** `{data.get('done', obs.get('done', '—'))}`",
        f"- **Success:** `{obs.get('success', '—')}`",
        "",
        "### Task",
        f"- **Task ID:** `{obs.get('task_id', '—')}`",
        f"- **Type:** `{obs.get('task_type', '—')}`",
        f"- **Difficulty:** `{obs.get('task_difficulty', '—')}`",
        f"- **Turn:** `{obs.get('turn_index', '—')}` / `{obs.get('max_turns', '—')}`",
        f"- **Current agent:** `{obs.get('current_agent_id', '—')}`",
        "",
    ]
    prompt = obs.get("task_prompt") or ""
    if prompt:
        lines.extend(["**Task prompt**", "", prompt, ""])
    skills = obs.get("task_skills") or []
    if skills:
        lines.append(f"**Skills tested:** {', '.join(skills)}")
        lines.append("")
    breakdown = obs.get("reward_breakdown") or {}
    if breakdown:
        lines.append("**Reward breakdown**")
        for key, val in breakdown.items():
            lines.append(f"- `{key}`: `{val}`")
        lines.append("")
    team = obs.get("team_agent_ids") or []
    if team:
        lines.append(f"**Team:** {', '.join(team)}")
    return "\n".join(lines)


def build_amases_gradio_app(
    web_manager: WebInterfaceManager,
    metadata: Any,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            return (
                _format_amases_observation(data),
                json.dumps(data, indent=2),
                "Episode reset — new task from curriculum.",
            )
        except Exception as exc:
            return ("", "", f"Error: {exc}")

    async def step_env(
        agent_id: str,
        task_id: str,
        response_text: str,
        self_rating: float,
        merged_reward_override: str,
    ):
        action: Dict[str, Any] = {
            "agent_id": agent_id or AGENT_CHOICES[0],
            "task_id": task_id or "",
            "response_text": response_text or "",
            "self_rating": float(self_rating),
        }
        if merged_reward_override and str(merged_reward_override).strip():
            try:
                action["merged_reward_override"] = float(merged_reward_override)
            except ValueError:
                return ("", "", "Merged reward override must be a number or empty.")
        try:
            data = await web_manager.step_environment(action)
            return (
                _format_amases_observation(data),
                json.dumps(data, indent=2),
                "Step recorded.",
            )
        except Exception as exc:
            return ("", "", f"Error: {exc}")

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2)
        except Exception as exc:
            return f"Error: {exc}"

    title = metadata.name if metadata else "AMASES"
    with gr.Blocks(title=f"AMASES · {title}") as demo:
        gr.HTML(
            """
            <div class="amases-hero">
              <h1>AMASES — Adaptive Multi-Agent Skill Evolution</h1>
              <p>Three agents · curriculum tasks · rubric rewards · skill graph updates each turn</p>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes="amases-panel"):
                gr.Markdown("#### Quick guide")
                gr.Markdown(
                    "1. **Reset** — new task  \n"
                    "2. Use **current agent** from the panel →  \n"
                    "3. **Step** — include keywords: `evidence`, `synthesis`, `trade-off`, `summary`  \n"
                    "4. Leave **Merged reward** empty unless debugging"
                )
                if quick_start_md:
                    with gr.Accordion("Connect from Python", open=False):
                        gr.Markdown(quick_start_md)

            with gr.Column(scale=2, elem_classes="amases-panel"):
                obs_display = gr.Markdown(
                    value="### Ready\n\nClick **Reset episode** to sample a task from the curriculum.",
                )
                with gr.Group():
                    agent_id = gr.Dropdown(
                        choices=AGENT_CHOICES,
                        value=AGENT_CHOICES[0],
                        label="Agent",
                        info="Planner α · Debater β · Integrator γ",
                    )
                    task_id = gr.Textbox(
                        label="Task ID",
                        placeholder="Filled after reset (e.g. collaborative_medium)",
                    )
                    response_text = gr.Textbox(
                        label="Response",
                        lines=5,
                        placeholder="Agent message for this turn…",
                    )
                    self_rating = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        label="Self rating",
                    )
                    merged_reward_override = gr.Textbox(
                        label="Merged reward override (optional)",
                        placeholder="Leave empty",
                    )
                with gr.Row():
                    reset_btn = gr.Button("Reset episode", variant="secondary")
                    step_btn = gr.Button("Step", variant="primary")
                    state_btn = gr.Button("Get state", variant="secondary")
                status = gr.Textbox(label="Status", interactive=False)
                raw_json = gr.Code(label="Raw JSON", language="json", interactive=False)

        reset_btn.click(reset_env, outputs=[obs_display, raw_json, status])
        step_btn.click(
            step_env,
            inputs=[agent_id, task_id, response_text, self_rating, merged_reward_override],
            outputs=[obs_display, raw_json, status],
        )
        state_btn.click(get_state_sync, outputs=[raw_json])

    return demo


def create_amases_web_interface_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[Any] = None,
) -> FastAPI:
    """FastAPI app with AMASES-only Gradio UI at /web (no default OpenEnv tab)."""
    app = create_fastapi_app(
        env, action_cls, observation_cls, max_concurrent_envs, concurrency_config
    )
    metadata = load_environment_metadata(env, env_name)
    web_manager = WebInterfaceManager(env, action_cls, observation_cls, metadata)
    quick_start_md = get_quick_start_markdown(metadata, action_cls, observation_cls)

    @app.get("/", include_in_schema=False)
    async def web_root():
        return RedirectResponse(url="/web/")

    @app.get("/web", include_in_schema=False)
    async def web_root_no_slash():
        return RedirectResponse(url="/web/")

    @app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        action_data = request.get("action", request)
        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    blocks = build_amases_gradio_app(web_manager, metadata, quick_start_md)
    return gr.mount_gradio_app(app, blocks, path="/web", theme=AMASES_THEME, css=AMASES_CSS)


def create_amases_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[Any] = None,
) -> FastAPI:
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
    if enable_web:
        return create_amases_web_interface_app(
            env,
            action_cls,
            observation_cls,
            env_name,
            max_concurrent_envs,
            concurrency_config,
        )
    return create_fastapi_app(
        env, action_cls, observation_cls, max_concurrent_envs, concurrency_config
    )
