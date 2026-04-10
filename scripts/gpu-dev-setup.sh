#!/usr/bin/env bash
#
# gpu-dev-setup.sh
#
# Sets up a local GPU development environment for roboharness.
# Run this on any Linux machine with an NVIDIA GPU to get started
# with GPU-dependent tasks (demo debugging, visual QA, ONNX/rendering).
#
# Prerequisites:
#   - NVIDIA GPU with drivers installed (nvidia-smi should work)
#   - Python 3.10+ (python3 or conda/venv)
#   - pip
#
# Usage:
#   ./scripts/gpu-dev-setup.sh
#
# Environment variables:
#   CUDA_VERSION  - PyTorch CUDA version (default: cu121)
#   PYTHON        - Python executable (default: python3)
#   SKIP_TORCH    - Set to 1 to skip PyTorch installation
#

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-cu121}"
PYTHON="${PYTHON:-python3}"
SKIP_TORCH="${SKIP_TORCH:-0}"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── Preflight checks ───────────────────────────────────────────────────────
info "Checking prerequisites..."

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    error "nvidia-smi failed. GPU drivers may not be loaded."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
info "GPU detected: ${GPU_NAME}"

if ! command -v "${PYTHON}" &>/dev/null; then
    error "${PYTHON} not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("${PYTHON}" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("${PYTHON}" -c 'import sys; print(sys.version_info.minor)')

if [ "${PY_MAJOR}" -lt 3 ] || { [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -lt 10 ]; }; then
    error "Python 3.10+ required, found ${PY_VERSION}"
    exit 1
fi

info "Python ${PY_VERSION} OK"

# ─── Install roboharness with all optional deps ─────────────────────────────
info "Installing roboharness with demo + dev dependencies..."
"${PYTHON}" -m pip install -e ".[demo,dev]"

# ─── Install PyTorch with CUDA ──────────────────────────────────────────────
if [ "${SKIP_TORCH}" = "1" ]; then
    warn "Skipping PyTorch installation (SKIP_TORCH=1)"
else
    info "Installing PyTorch with CUDA (${CUDA_VERSION})..."
    "${PYTHON}" -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

# ─── Verify GPU rendering ───────────────────────────────────────────────────
info "Verifying MuJoCo GPU rendering..."
if MUJOCO_GL=egl "${PYTHON}" -c "import mujoco; print('MuJoCo OK')" 2>/dev/null; then
    info "MuJoCo EGL rendering: OK"
else
    warn "MuJoCo EGL rendering failed. Trying osmesa..."
    if MUJOCO_GL=osmesa "${PYTHON}" -c "import mujoco; print('MuJoCo OK')" 2>/dev/null; then
        info "MuJoCo osmesa rendering: OK (use MUJOCO_GL=osmesa)"
    else
        warn "MuJoCo rendering not available. Install libosmesa6-dev or EGL libraries."
    fi
fi

# ─── Verify PyTorch CUDA ────────────────────────────────────────────────────
if [ "${SKIP_TORCH}" != "1" ]; then
    info "Verifying PyTorch CUDA..."
    CUDA_AVAIL=$("${PYTHON}" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo "False")
    if [ "${CUDA_AVAIL}" = "True" ]; then
        GPU_COUNT=$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
        info "PyTorch CUDA: OK (${GPU_COUNT} GPU(s))"
    else
        warn "PyTorch CUDA not available. Check CUDA toolkit version matches ${CUDA_VERSION}."
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
info "Setup complete. Quick start:"
echo "  # Run MuJoCo grasp demo"
echo "  MUJOCO_GL=egl python examples/mujoco_grasp.py --report"
echo ""
echo "  # Run SONIC locomotion demo"
echo "  python examples/sonic_locomotion.py --report"
echo ""
echo "  # Run tests"
echo "  pytest"
echo ""
echo "  # Start Claude Code CLI"
echo "  claude"
