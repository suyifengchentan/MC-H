from __future__ import annotations

from typing import Literal

import torch

Float32MatmulPrecision = Literal["highest", "high", "medium"]


def configure_cuda_runtime(
    device: torch.device,
    *,
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    float32_matmul_precision: Float32MatmulPrecision = "high",
) -> dict[str, object]:
    state: dict[str, object] = {
        "device": str(device),
        "tf32_requested": bool(enable_tf32),
        "tf32_enabled": False,
        "cudnn_benchmark": False,
        "float32_matmul_precision": float32_matmul_precision,
    }

    if device.type != "cuda":
        return state

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(float32_matmul_precision)

    if hasattr(torch.backends.cuda, "matmul") and hasattr(
        torch.backends.cuda.matmul, "allow_tf32"
    ):
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enable_tf32

    if hasattr(torch.backends.cudnn, "benchmark"):
        torch.backends.cudnn.benchmark = cudnn_benchmark

    state["tf32_enabled"] = bool(
        getattr(getattr(torch.backends.cuda, "matmul", None), "allow_tf32", False)
    ) and bool(getattr(torch.backends.cudnn, "allow_tf32", False))
    state["cudnn_benchmark"] = bool(getattr(torch.backends.cudnn, "benchmark", False))
    return state


def format_runtime_accel_state(state: dict[str, object]) -> str:
    return (
        "device={device} tf32_requested={tf32_requested} tf32_enabled={tf32_enabled} "
        "cudnn_benchmark={cudnn_benchmark} float32_matmul_precision={float32_matmul_precision}"
    ).format(**state)
