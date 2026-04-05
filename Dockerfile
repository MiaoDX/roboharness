# Pre-built image for roboharness development and CI.
# Pre-installs system deps, uv, Python packages, and model weights.
#
# CI workflows use uv directly (no Docker dependency) for reliability.
# This image is available for local dev and future CI optimization.
#
# Rebuild triggers: Dockerfile, pyproject.toml changes, or weekly schedule.
# Published to: ghcr.io/miaodx/roboharness/ci

FROM python:3.12-slim

# System dependencies for MuJoCo headless rendering (OSMesa)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libosmesa6-dev \
        libgl1-mesa-glx \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set uv to use system Python (no virtualenv inside container)
ENV UV_SYSTEM_PYTHON=1

# Copy only dependency metadata first (for layer caching)
WORKDIR /opt/roboharness
COPY pyproject.toml .
RUN mkdir -p src/roboharness && \
    echo '__version__ = "0.0.0"' > src/roboharness/__init__.py

# Install all dependency groups
RUN uv pip install -e ".[demo,dev]"

# Install CPU-only PyTorch + lerobot (for native LeRobot examples)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install lerobot

# Install WBC dependencies (pinocchio + pink)
RUN uv pip install -e ".[wbc]" || true

# Pre-cache HuggingFace model weights used by examples
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('lerobot/unitree-g1-mujoco', repo_type='model')" || true

# Pre-cache robot description files used by MuJoCo examples
RUN python -c "\
import robot_descriptions; \
from pathlib import Path; \
print('robot_descriptions imported successfully')" || true

# Clean up the dummy package
RUN rm -rf /opt/roboharness

WORKDIR /workspace
