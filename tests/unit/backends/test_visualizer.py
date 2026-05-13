"""Tests for the Visualizer protocol and implementations."""

import numpy as np
import pytest

from roboharness.core.capture import CameraView

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class FakeVisualizer:
    """Minimal Visualizer implementation for protocol conformance tests."""

    def __init__(self) -> None:
        self.sync_count = 0
        self.capture_count = 0

    def capture_camera(self, camera_name: str) -> CameraView:
        self.capture_count += 1
        return CameraView(
            name=camera_name,
            rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        )

    def sync(self) -> None:
        self.sync_count += 1


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


def test_fake_visualizer_implements_protocol():
    """Verify FakeVisualizer structurally matches Visualizer Protocol."""
    viz = FakeVisualizer()
    assert callable(getattr(viz, "capture_camera", None))
    assert callable(getattr(viz, "sync", None))


def test_fake_visualizer_capture():
    viz = FakeVisualizer()
    view = viz.capture_camera("front")
    assert view.name == "front"
    assert view.rgb.shape == (64, 64, 3)


# ------------------------------------------------------------------
# Integration: backend + custom visualizer
# ------------------------------------------------------------------


@pytest.fixture
def _has_mujoco():
    """Skip tests that need MuJoCo rendering (import + GL context).

    MuJoCo rendering requires a GL context (osmesa, egl, or display).
    Without one, ``mujoco.Renderer()`` calls C-level ``abort()`` which
    kills the entire pytest process. We check for a usable GL backend
    before attempting to create a renderer.
    """
    mujoco = pytest.importorskip("mujoco")
    # Check that a GL backend is configured (MUJOCO_GL env var or display)
    import os

    gl_backend = os.environ.get("MUJOCO_GL", "").lower()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not gl_backend and not has_display:
        pytest.skip("MuJoCo rendering not available (no MUJOCO_GL or DISPLAY set)")
    # Also verify the rendering module can be loaded
    try:
        mujoco.Renderer  # noqa: B018
    except AttributeError:
        pytest.skip("MuJoCo version does not support Renderer")


MINIMAL_MJCF = """\
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <camera name="front" pos="1 0 0.5" xyaxes="0 1 0 0 0 1"/>
    <body pos="0 0 0.1">
      <joint type="free"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.usefixtures("_has_mujoco")
def test_backend_default_native_visualizer():
    """Default visualizer should be MuJoCoNativeVisualizer."""
    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
    from roboharness.backends.visualizer import MuJoCoNativeVisualizer

    backend = MuJoCoMeshcatBackend(xml_string=MINIMAL_MJCF, cameras=["front"])
    assert isinstance(backend.visualizer, MuJoCoNativeVisualizer)


@pytest.mark.usefixtures("_has_mujoco")
def test_backend_with_custom_visualizer():
    """MuJoCoMeshcatBackend should accept and delegate to a custom Visualizer."""
    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend

    viz = FakeVisualizer()
    backend = MuJoCoMeshcatBackend(xml_string=MINIMAL_MJCF, cameras=["front"], visualizer=viz)
    assert callable(getattr(backend, "reset", None))
    assert callable(getattr(backend, "step", None))

    backend.reset()
    assert viz.sync_count == 1  # reset calls sync exactly once

    backend.step(None)
    assert viz.sync_count == 2  # step calls sync exactly once more

    view = backend.capture_camera("front")
    assert view.name == "front"
    assert viz.capture_count == 1


@pytest.mark.usefixtures("_has_mujoco")
def test_backend_visualizer_string_native():
    """Passing visualizer='native' should create MuJoCoNativeVisualizer."""
    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
    from roboharness.backends.visualizer import MuJoCoNativeVisualizer

    backend = MuJoCoMeshcatBackend(xml_string=MINIMAL_MJCF, cameras=["front"], visualizer="native")
    assert isinstance(backend.visualizer, MuJoCoNativeVisualizer)


@pytest.mark.usefixtures("_has_mujoco")
def test_backend_visualizer_string_unknown():
    """Unknown visualizer string should raise ValueError."""
    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend

    with pytest.raises(ValueError, match="Unknown visualizer"):
        MuJoCoMeshcatBackend(xml_string=MINIMAL_MJCF, cameras=["front"], visualizer="unknown")
