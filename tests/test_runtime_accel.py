from __future__ import annotations

import torch

from hypersphere_dpp.runtime_accel import (
    configure_cuda_runtime,
    format_runtime_accel_state,
)


def test_configure_cuda_runtime_cpu_is_noop():
    state = configure_cuda_runtime(torch.device("cpu"))
    assert state["device"] == "cpu"
    assert state["tf32_enabled"] is False
    assert state["cudnn_benchmark"] is False


def test_format_runtime_accel_state_contains_key_fields():
    formatted = format_runtime_accel_state(
        {
            "device": "cuda:0",
            "tf32_requested": True,
            "tf32_enabled": True,
            "cudnn_benchmark": True,
            "float32_matmul_precision": "high",
        }
    )
    assert "cuda:0" in formatted
    assert "tf32_enabled=True" in formatted
