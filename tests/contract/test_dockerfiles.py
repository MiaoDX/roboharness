"""Contract tests for CI Docker image definitions."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dockerfile(name: str) -> str:
    return (REPO_ROOT / name).read_text()


def test_cpu_dockerfile_uses_current_debian_gl_package() -> None:
    dockerfile = _dockerfile("Dockerfile")

    assert "libgl1-mesa-glx" not in dockerfile
    assert "libgl1" in dockerfile


def test_dockerfiles_copy_readme_before_editable_install() -> None:
    for name in ("Dockerfile", "Dockerfile.gpu"):
        dockerfile = _dockerfile(name)
        readme_index = dockerfile.index("COPY README.md .")
        install_index = dockerfile.index('RUN uv pip install -e "')

        assert readme_index < install_index


def test_cpu_dockerfile_has_unitree_build_toolchain() -> None:
    dockerfile = _dockerfile("Dockerfile")
    unitree_install_index = dockerfile.index('RUN uv pip install "unitree-sdk2py')

    for package in ("build-essential", "cmake"):
        package_index = dockerfile.index(package)
        assert package_index < unitree_install_index
