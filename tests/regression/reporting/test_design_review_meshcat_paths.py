from __future__ import annotations

import base64
from pathlib import Path

from roboharness.reporting import generate_html_report

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yv3sAAAAASUVORK5CYII="
)


def _write_tiny_png(path: Path) -> None:
    path.write_bytes(_ONE_PIXEL_PNG)


def test_generate_html_report_uses_output_relative_meshcat_paths_and_titles(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task" / "trial_001"
    cp_dir = trial_dir / "plan"
    cp_dir.mkdir(parents=True)
    (cp_dir / "metadata.json").write_text('{"step": 0}')
    _write_tiny_png(cp_dir / "front_rgb.png")
    (cp_dir / "meshcat_scene.html").write_text("<html>meshcat</html>")

    report_path = generate_html_report(tmp_path, "task", meshcat_mode="iframe")

    html = report_path.read_text()
    assert 'src="task/trial_001/plan/meshcat_scene.html"' in html
    assert 'title="plan interactive 3D scene"' in html
