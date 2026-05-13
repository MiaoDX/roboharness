.PHONY: lint format typecheck test check-all check-gpu \
       demo-grasp demo-sonic demo-g1 demo-g1-native demo-wbc demos \
       init-showcase

# --- Development (CPU, works in web sessions) ---

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src/

test:
	pytest

test-quick:
	pytest --no-cov -x

check-all: lint typecheck test

# --- GPU environment verification ---

check-gpu:
	@python3 -c "import mujoco; print('mujoco: OK')"
	@python3 -c "import onnxruntime as ort; print('onnxruntime:', ort.get_device())"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
		|| echo "nvidia-smi: not available (CPU-only environment)"

# --- Demos (require GPU or MUJOCO_GL=osmesa for headless) ---

MUJOCO_GL ?= osmesa

demo-grasp:
	MUJOCO_GL=$(MUJOCO_GL) python examples/demos/mujoco/grasp.py --report

demo-sonic:
	MUJOCO_GL=$(MUJOCO_GL) python examples/demos/sonic/locomotion.py --report

demo-g1:
	MUJOCO_GL=$(MUJOCO_GL) python examples/demos/g1/lerobot_locomotion.py --report

demo-g1-native:
	MUJOCO_GL=$(MUJOCO_GL) python examples/demos/g1/lerobot_native.py --report

demo-wbc:
	MUJOCO_GL=$(MUJOCO_GL) python examples/demos/g1/wbc_reach.py --report

demos: demo-grasp demo-wbc demo-g1 demo-sonic
	@echo "All demos complete. Check output directories for reports."

# --- Showcase repo generation ---

init-showcase:
	python scripts/init-showcase-repo.py ./showcase-output --force
	@echo "\nShowcase repo generated in ./showcase-output/"
