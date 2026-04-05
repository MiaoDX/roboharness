"""Tests for optional Rerun capture logging."""

from __future__ import annotations

import sys

from roboharness.core.harness import Harness
from tests.test_harness import MockBackend


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def init(self, app_id: str, spawn: bool = False) -> None:
        self.calls.append(("init", app_id, spawn))

    def save(self, path: str) -> None:
        self.calls.append(("save", path))

    def set_time_sequence(self, name: str, value: int) -> None:
        self.calls.append(("set_time_sequence", name, value))

    def set_time_seconds(self, name: str, value: float) -> None:
        self.calls.append(("set_time_seconds", name, value))

    def log(self, path: str, obj: object) -> None:
        self.calls.append(("log", path, type(obj).__name__))

    def send_blueprint(self, _bp: object) -> None:
        self.calls.append(("send_blueprint",))

    class Image:
        def __init__(self, _arr):
            pass

    class DepthImage:
        def __init__(self, _arr):
            pass

    class SegmentationImage:
        def __init__(self, _arr):
            pass

    class TextDocument:
        def __init__(self, _text: str):
            pass


class _BlueprintNS:
    class Blueprint:
        def __init__(self, _inner: object):
            pass

    class Horizontal:
        def __init__(self, *_views: object):
            pass

    class Spatial2D:
        def __init__(self, origin: str):
            self.origin = origin

    class TextDocumentView:
        def __init__(self, origin: str):
            self.origin = origin


def test_harness_rerun_logs_when_sdk_present(tmp_path, monkeypatch):
    rr = _Recorder()
    rr.blueprint = _BlueprintNS
    monkeypatch.setitem(sys.modules, "rerun", rr)

    harness = Harness(
        MockBackend(),
        output_dir=tmp_path,
        task_name="demo",
        enable_rerun=True,
        rerun_app_id="testapp",
    )
    harness.add_checkpoint("cp1", cameras=["front"])
    harness.reset()

    result = harness.run_to_next_checkpoint([None, None])

    assert result is not None
    assert any(call[0] == "save" and call[1].endswith("capture.rrd") for call in rr.calls)
    assert any(call[:2] == ("log", "camera/front/rgb") for call in rr.calls)
    assert result.metadata["rerun_rrd"].endswith("capture.rrd")


def test_harness_rerun_gracefully_disabled_when_sdk_missing(tmp_path, monkeypatch):
    import builtins

    monkeypatch.delitem(sys.modules, "rerun", raising=False)

    # Block rerun from being re-imported from disk
    _real_import = builtins.__import__

    def _block_rerun(name, *args, **kwargs):
        if name == "rerun" or name.startswith("rerun."):
            raise ImportError("No module named 'rerun'")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_rerun)

    harness = Harness(MockBackend(), output_dir=tmp_path, enable_rerun=True)
    harness.add_checkpoint("cp1", cameras=["front"])
    harness.reset()

    result = harness.run_to_next_checkpoint([None])

    assert result is not None
    assert "rerun_rrd" not in result.metadata
