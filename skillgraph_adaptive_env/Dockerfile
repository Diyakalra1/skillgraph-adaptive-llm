# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# HF Spaces: README must set app_port: 8000 (default HF docker port is 7860).

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Prefer uv.lock when present; otherwise pip into a venv (reliable on HF without lockfile).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        python -m venv /app/env/.venv && \
        /app/env/.venv/bin/pip install --upgrade pip && \
        /app/env/.venv/bin/pip install --no-cache-dir -r requirements.txt; \
    fi

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env"
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
