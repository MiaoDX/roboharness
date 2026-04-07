"""Minimal GPU smoke tests to verify the GPU CI pipeline works."""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_cuda_available() -> None:
    """Verify CUDA is available and at least one GPU is detected."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA should be available on GPU runner"
    assert torch.cuda.device_count() >= 1, "At least one GPU should be detected"


@pytest.mark.gpu
def test_cuda_tensor_roundtrip() -> None:
    """Verify basic GPU tensor operations work."""
    torch = pytest.importorskip("torch")
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    gpu_tensor = cpu_tensor.cuda()
    result = gpu_tensor.cpu()
    assert torch.equal(cpu_tensor, result), "CPU -> GPU -> CPU roundtrip should preserve values"
