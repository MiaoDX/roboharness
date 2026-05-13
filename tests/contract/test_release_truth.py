from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib


def test_version_is_synced_between_pyproject_and_package() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_init = Path("src/roboharness/__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', package_init)
    assert match is not None
    assert pyproject["project"]["version"] == match.group(1)


def test_project_urls_point_to_lowercase_repo() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    urls = pyproject["project"]["urls"]
    assert "MiaoDX/roboharness" in urls["Homepage"]
    assert "MiaoDX/roboharness" in urls["Repository"]
    assert "MiaoDX/roboharness/issues" in urls["Issues"]
    assert "MiaoDX/roboharness/tree/main/docs" in urls["Documentation"]
