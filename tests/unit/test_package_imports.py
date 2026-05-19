"""Package import boundary tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_core_package_import_does_not_require_gym() -> None:
    """The core wheel must import without optional simulator extras installed."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class BlockGymImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "gym" or fullname.startswith("gym."):
                    raise ModuleNotFoundError(f"No module named {fullname!r}")
                if fullname == "gymnasium" or fullname.startswith("gymnasium."):
                    raise ModuleNotFoundError(f"No module named {fullname!r}")
                return None


        sys.meta_path.insert(0, BlockGymImports())

        import roboharness
        from roboharness import AssertionEngine, Harness

        assert roboharness.__version__
        assert AssertionEngine is not None
        assert Harness is not None
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
