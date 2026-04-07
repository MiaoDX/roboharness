"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip GPU-marked tests when CUDA is not available."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
