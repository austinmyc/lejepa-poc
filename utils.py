"""
Device, dtype, and mixed-precision utilities.

REVIEW: Understand why each device gets a different dtype, and why
autocast is a no-op on MPS/CPU. This affects every forward pass.
"""

import contextlib
import torch


def get_device() -> str:
    # REVIEW: priority order matters — MPS is Apple Silicon GPU.
    # It supports float32 but not all bf16 ops, so we fall back to float32 there.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    # REVIEW: bf16 gives ~2x memory saving over float32 on CUDA with no accuracy loss.
    # fp16 needs GradScaler to avoid underflow (gradients go to zero).
    # MPS/CPU: bf16 is not reliably supported — use float32.
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def make_autocast(device: str, dtype: torch.dtype):
    """
    Returns a context manager that casts operations to `dtype` on CUDA.
    On MPS/CPU returns a no-op — operations just run in float32.

    REVIEW: autocast only wraps the forward pass. Gradients and optimizer
    state stay in float32 (weights are stored in fp32, cast during forward).
    This is why model.parameters() shows float32 even during bf16 training.
    """
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_scaler(device: str, dtype: torch.dtype):
    """
    GradScaler prevents fp16 gradient underflow by scaling the loss up
    before backward, then unscaling before the optimizer step.

    REVIEW: bf16 has wider dynamic range than fp16, so it doesn't need
    scaling. Only enable for CUDA + fp16. On MPS/CPU this is a no-op.
    """
    enabled = (device == "cuda") and (dtype == torch.float16)
    return torch.cuda.amp.GradScaler(enabled=enabled)
