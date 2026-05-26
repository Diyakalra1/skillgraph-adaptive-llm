# Copyright (c) Meta Platforms, Inc. and affiliates.
# Custom Gradio UI for AMASES on Hugging Face Spaces.

from __future__ import annotations

import html
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
AGENT_LABELS = {
    "agent_alpha": ("Planner", "α", "#818cf8"),
    "agent_beta": ("Debater", "β", "#38bdf8"),
    "agent_gamma": ("Integrator", "γ", "#34d399"),
}
AGENT_DROPDOWN = [(f"{AGENT_LABELS[a][1]} {AGENT_LABELS[a][0]}", a) for a in AGENT_CHOICES]

AMASES_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("DM Sans"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill_dark="#070b14",
    background_fill_primary_dark="#0c1222",
    background_fill_secondary_dark="#131c31",
    block_background_fill_dark="rgba(19, 28, 49, 0.85)",
    block_border_color_dark="rgba(148, 163, 184, 0.18)",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color_dark="#f1f5f9",
    border_color_primary_dark="rgba(99, 102, 241, 0.35)",
    input_background_fill_dark="#0f1729",
    input_border_color_dark="rgba(99, 102, 241, 0.25)",
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill_dark="#1e293b",
    button_secondary_background_fill_hover_dark="#334155",
)

AMASES_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

/* Full-width, centered layout on HF Spaces (avoids left-sidelined narrow column). */
html, body, .gradio-container, .gradio-container .main,
.gradio-container .wrap, .gradio-container .contain {
    width: 100% !important;
    max-width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-sizing: border-box !important;
}
.gradio-container {
    font-family: 'DM Sans', 'Inter', system-ui, sans-serif !important;
    padding: 16px clamp(16px, 3vw, 40px) 32px !important;
}
.amases-root {
    width: 100% !important;
    max-width: min(1320px, 100%) !important;
    margin: 0 auto !important;
}
.amases-root > .gr-row {
    width: 100% !important;
    align-items: stretch !important;
}
.amases-root > .gr-row > .gr-column {
    flex: 1 1 auto !important;
}

/* Hero */
.amases-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(125deg, #1e1b4b 0%, #312e81 28%, #1e3a8a 58%, #0f172a 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 20px;
    padding: 28px 32px 24px;
    margin-bottom: 20px;
    box-shadow: 0 24px 48px rgba(15, 23, 42, 0.45), inset 0 1px 0 rgba(255,255,255,0.06);
}
.amases-hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.35) 0%, transparent 70%);
    pointer-events: none;
}
.amases-hero h1 {
    margin: 0 0 8px 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
    position: relative;
}
.amases-hero .tagline {
    margin: 0 0 18px 0;
    color: #cbd5e1;
    font-size: 1rem;
    line-height: 1.5;
    max-width: 52rem;
    position: relative;
}
.amases-agents {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    position: relative;
}
.amases-agent-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(8px);
    color: #e2e8f0;
}
.amases-agent-chip .badge {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #0f172a;
}

/* Panels */
.amases-panel {
    background: rgba(15, 23, 42, 0.55) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important;
}
.amases-sidebar-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 12px;
}
.amases-step {
    display: flex;
    gap: 12px;
    margin-bottom: 14px;
    align-items: flex-start;
}
.amases-step-num {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.amases-step-text { color: #cbd5e1; font-size: 0.9rem; line-height: 1.45; }
.amases-keywords {
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px dashed rgba(129, 140, 248, 0.4);
    font-size: 0.85rem;
    color: #a5b4fc;
}
.amases-keywords code {
    background: rgba(15, 23, 42, 0.6);
    padding: 2px 8px;
    border-radius: 6px;
    margin: 2px 4px 2px 0;
    font-size: 0.8rem;
    color: #e0e7ff;
}

/* Observation dashboard */
.amases-dash {
    border-radius: 14px;
    padding: 4px 2px 8px;
}
.amases-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
@media (max-width: 720px) {
    .amases-metrics { grid-template-columns: 1fr; }
}
.amases-metric {
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.amases-metric .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 6px;
}
.amases-metric .value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
}
.amases-metric.reward .value { color: #34d399; }
.amases-metric.done-true .value { color: #fbbf24; }
.amases-section {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(71, 85, 105, 0.35);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.amases-section h4 {
    margin: 0 0 10px 0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #818cf8;
}
.amases-section p, .amases-section li {
    margin: 0;
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 1.55;
}
.amases-prompt {
    color: #cbd5e1;
    white-space: pre-wrap;
}
.amases-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.amases-pill {
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.15);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.amases-empty {
    text-align: center;
    padding: 32px 16px;
    color: #64748b;
}
.amases-empty .icon { font-size: 2.5rem; margin-bottom: 8px; opacity: 0.7; }

/* Buttons row */
.amases-actions button { min-height: 44px !important; font-weight: 600 !important; }
.amases-status-ok {
    color: #34d399 !important;
    font-weight: 500 !important;
}
"""


def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "—")


def _format_amases_observation_html(data: Dict[str, Any]) -> str:
    obs = data.get("observation") or {}
    if not isinstance(obs, dict):
        return '<div class="amases-empty"><div class="icon">◇</div><p>No observation yet</p></div>'

    reward = data.get("reward", obs.get("reward", "—"))
    done = data.get("done", obs.get("done", False))
    success = obs.get("success", False)
    done_cls = "done-true" if done else ""
    reward_val = _esc(reward)

    task_id = _esc(obs.get("task_id", "—"))
    task_type = _esc(obs.get("task_type", "—"))
    difficulty = _esc(obs.get("task_difficulty", "—"))
    turn = _esc(obs.get("turn_index", "—"))
    max_turns = _esc(obs.get("max_turns", "—"))
    current_agent = _esc(obs.get("current_agent_id", "—"))

    prompt = obs.get("task_prompt") or ""
    skills = obs.get("task_skills") or []
    breakdown = obs.get("reward_breakdown") or {}
    team = obs.get("team_agent_ids") or []

    skills_html = "".join(f'<span class="amases-pill">{_esc(s)}</span>' for s in skills)
    team_html = ", ".join(_esc(a) for a in team) if team else "—"

    breakdown_rows = ""
    for key, val in list(breakdown.items())[:8]:
        breakdown_rows += f"<li><strong>{_esc(key)}</strong>: {_esc(val)}</li>"

    breakdown_block = ""
    if breakdown_rows:
        breakdown_block = f"""
        <div class="amases-section">
          <h4>Reward breakdown</h4>
          <ul style="padding-left: 1.1rem; margin: 0;">{breakdown_rows}</ul>
        </div>
        """

    prompt_block = ""
    if prompt:
        prompt_block = f"""
        <div class="amases-section">
          <h4>Task prompt</h4>
          <p class="amases-prompt">{_esc(prompt)}</p>
        </div>
        """

    skills_block = ""
    if skills_html:
        skills_block = f"""
        <div class="amases-section">
          <h4>Skills tested</h4>
          <div class="amases-pills">{skills_html}</div>
        </div>
        """

    return f"""
    <div class="amases-dash">
      <div class="amases-metrics">
        <div class="amases-metric reward">
          <div class="label">Reward</div>
          <div class="value">{reward_val}</div>
        </div>
        <div class="amases-metric {done_cls}">
          <div class="label">Done</div>
          <div class="value">{_esc(done)}</div>
        </div>
        <div class="amases-metric">
          <div class="label">Success</div>
          <div class="value">{_esc(success)}</div>
        </div>
      </div>
      <div class="amases-section">
        <h4>Task · Turn {turn} / {max_turns}</h4>
        <p><strong>ID</strong> {task_id} · <strong>Type</strong> {task_type} ·
           <strong>Difficulty</strong> {difficulty}</p>
        <p style="margin-top:8px"><strong>Acting now:</strong> {current_agent}</p>
        <p style="margin-top:8px; color:#94a3b8"><strong>Team:</strong> {team_html}</p>
      </div>
      {prompt_block}
      {skills_block}
      {breakdown_block}
    </div>
    """


def _hero_html() -> str:
    chips = []
    for aid in AGENT_CHOICES:
        role, glyph, color = AGENT_LABELS[aid]
        chips.append(
            f'<span class="amases-agent-chip">'
            f'<span class="badge" style="background:{color}">{glyph}</span>'
            f"{role} <span style='opacity:0.65;font-weight:400'>· {aid}</span></span>"
        )
    return f"""
    <div class="amases-hero">
      <h1>AMASES</h1>
      <p class="tagline">Adaptive Multi-Agent Skill Evolution — curriculum tasks, rubric rewards, and live skill-graph updates.</p>
      <div class="amases-agents">{''.join(chips)}</div>
    </div>
    """


def _sidebar_html() -> str:
    return """
    <div class="amases-sidebar-title">How to play</div>
    <div class="amases-step">
      <span class="amases-step-num">1</span>
      <span class="amases-step-text"><strong>Reset episode</strong> — samples a new task from the 15-task curriculum.</span>
    </div>
    <div class="amases-step">
      <span class="amases-step-num">2</span>
      <span class="amases-step-text"><strong>Pick the agent</strong> shown as “Acting now” in the dashboard (or rotate α → β → γ).</span>
    </div>
    <div class="amases-step">
      <span class="amases-step-num">3</span>
      <span class="amases-step-text"><strong>Step</strong> with a strong response — higher reward when rubric keywords appear.</span>
    </div>
    <div class="amases-keywords">
      <strong style="color:#c7d2fe">Tip:</strong> include
      <code>evidence</code><code>synthesis</code><code>trade-off</code><code>summary</code>
      for collaborative tasks.
    </div>
    """


def _empty_dashboard_html() -> str:
    return """
    <div class="amases-empty">
      <div class="icon">✦</div>
      <p style="font-size:1.05rem;color:#94a3b8;margin-bottom:4px">Ready for a new episode</p>
      <p style="font-size:0.88rem">Click <strong>Reset episode</strong> to draw a task from the curriculum.</p>
    </div>
    """


def build_amases_gradio_app(
    web_manager: WebInterfaceManager,
    metadata: Any,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            obs = data.get("observation") or {}
            tid = obs.get("task_id", "") if isinstance(obs, dict) else ""
            agent = obs.get("current_agent_id", AGENT_CHOICES[0]) if isinstance(obs, dict) else AGENT_CHOICES[0]
            if agent not in AGENT_CHOICES:
                agent = AGENT_CHOICES[0]
            return (
                _format_amases_observation_html(data),
                json.dumps(data, indent=2),
                "✓ New episode — task loaded from curriculum.",
                tid,
                agent,
            )
        except Exception as exc:
            return (_empty_dashboard_html(), "", f"✗ Error: {exc}", "", AGENT_CHOICES[0])

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
                return (
                    _empty_dashboard_html(),
                    "",
                    "✗ Merged reward must be a number or empty.",
                    task_id,
                    agent_id,
                )
        try:
            data = await web_manager.step_environment(action)
            obs = data.get("observation") or {}
            tid = obs.get("task_id", task_id) if isinstance(obs, dict) else task_id
            agent = obs.get("current_agent_id", agent_id) if isinstance(obs, dict) else agent_id
            if agent not in AGENT_CHOICES:
                agent = agent_id
            return (
                _format_amases_observation_html(data),
                json.dumps(data, indent=2),
                "✓ Step recorded — check reward & breakdown above.",
                tid,
                agent,
            )
        except Exception as exc:
            return (_empty_dashboard_html(), "", f"✗ Error: {exc}", task_id, agent_id)

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2)
        except Exception as exc:
            return f"Error: {exc}"

    title = metadata.name if metadata else "AMASES"
    with gr.Blocks(title=f"AMASES · {title}", elem_classes="amases-root") as demo:
        gr.HTML(_hero_html())
        with gr.Row(equal_height=False, elem_classes="amases-main-row"):
            with gr.Column(scale=1, min_width=300, elem_classes="amases-panel"):
                gr.HTML(_sidebar_html())
                if quick_start_md:
                    with gr.Accordion("Python client", open=False):
                        gr.Markdown(quick_start_md)

            with gr.Column(scale=3, elem_classes="amases-panel"):
                gr.Markdown("#### Live dashboard")
                obs_display = gr.HTML(value=_empty_dashboard_html())
                with gr.Group():
                    agent_id = gr.Dropdown(
                        choices=AGENT_DROPDOWN,
                        value=AGENT_CHOICES[0],
                        label="Active agent",
                    )
                    with gr.Row():
                        task_id = gr.Textbox(
                            label="Task ID",
                            placeholder="Auto-filled on reset",
                            scale=2,
                        )
                        self_rating = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            label="Confidence",
                            scale=1,
                        )
                    response_text = gr.Textbox(
                        label="Agent response",
                        lines=4,
                        placeholder="Write the agent's message for this turn…",
                    )
                    merged_reward_override = gr.Textbox(
                        label="Merged reward override (optional)",
                        placeholder="Leave empty for rubric scoring",
                        visible=True,
                    )
                with gr.Row(elem_classes="amases-actions"):
                    reset_btn = gr.Button("↺ Reset episode", variant="secondary", scale=1)
                    step_btn = gr.Button("▶ Step", variant="primary", scale=1)
                    state_btn = gr.Button("{ } State", variant="secondary", scale=1)
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes="amases-status-ok",
                )
                with gr.Accordion("Raw JSON", open=False):
                    raw_json = gr.Code(language="json", interactive=False)

        reset_btn.click(
            reset_env,
            outputs=[obs_display, raw_json, status, task_id, agent_id],
        )
        step_btn.click(
            step_env,
            inputs=[agent_id, task_id, response_text, self_rating, merged_reward_override],
            outputs=[obs_display, raw_json, status, task_id, agent_id],
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
