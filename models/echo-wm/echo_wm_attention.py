"""Select and verify Echo-WM attention kernels without changing upstream code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttentionBenchmark:
    """Report representative SDPA and FlashAttention 4 measurements."""

    pytorch_milliseconds: float
    flash_attention_4_milliseconds: float
    speedup: float
    max_absolute_error: float
    mean_absolute_error: float
    normalized_root_mean_square_error: float


class FlashAttention4:
    """Use FlashAttention 4 for unmasked attention and SDPA for masked calls."""

    def __init__(
        self,
        flash_attention: Any,
        masked_fallback: Any,
        torch_module: Any,
    ) -> None:
        self._flash_attention = flash_attention
        self._masked_fallback = masked_fallback
        self._torch = torch_module

    def __call__(
        self,
        query: Any,
        key: Any,
        value: Any,
        heads: int,
        mask: Any | None = None,
    ) -> Any:
        """Return attention over flattened-head Echo-WM tensors."""
        if mask is not None:
            return self._masked_fallback(query, key, value, heads, mask)
        batch, _, fused = query.shape
        head_dim = fused // heads
        query, key, value = (
            tensor.view(batch, -1, heads, head_dim) for tensor in (query, key, value)
        )
        output = self._flash_attention(
            query.to(self._torch.bfloat16),
            key.to(self._torch.bfloat16),
            value.to(self._torch.bfloat16),
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(batch, -1, fused).to(value.dtype)


def resolve_attention_backend(
    backend: str,
    *,
    pytorch_attention: Any,
    torch_module: Any,
) -> Any:
    """Return the configured Echo-WM attention callable.

    Raises:
        RuntimeError: FlashAttention 4 is selected but unavailable.
    """
    if backend == "pytorch":
        return pytorch_attention
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as error:
        raise RuntimeError(
            "inference.attention_backend is flash_attention_4 but flash-attn-4 "
            "is unavailable; install requirements-flash-attn.txt or select pytorch"
        ) from error
    return FlashAttention4(flash_attn_func, pytorch_attention, torch_module)


def set_attention_backend(
    root: Any,
    attention_function: Any,
    attention_class: type[Any],
) -> int:
    """Assign one callable to every upstream transformer attention module."""
    changed = 0
    for module in root.modules():
        if isinstance(module, attention_class):
            module.attention_function = attention_function
            changed += 1
    if changed == 0:
        raise RuntimeError("Echo-WM exposed no configurable attention modules")
    return changed


def benchmark_attention_backends(
    *,
    flash_attention: Any,
    pytorch_attention: Any,
    torch_module: Any,
    query_tokens: int,
    key_value_tokens: int,
    heads: int = 32,
    head_dim: int = 128,
    warmup_iterations: int = 2,
    measured_iterations: int = 5,
) -> AttentionBenchmark:
    """Verify and time both kernels on a representative video-attention shape.

    Raises:
        RuntimeError: Either kernel returns non-finite or materially different output.
    """
    device = torch_module.device("cuda")
    generator = torch_module.Generator(device=device).manual_seed(0)
    fused = heads * head_dim
    query = torch_module.randn(
        (1, query_tokens, fused),
        device=device,
        dtype=torch_module.bfloat16,
        generator=generator,
    )
    key = torch_module.randn(
        (1, key_value_tokens, fused),
        device=device,
        dtype=torch_module.bfloat16,
        generator=generator,
    )
    value = torch_module.randn(
        (1, key_value_tokens, fused),
        device=device,
        dtype=torch_module.bfloat16,
        generator=generator,
    )

    reference = pytorch_attention(query, key, value, heads)
    candidate = flash_attention(query, key, value, heads)
    torch_module.cuda.synchronize(device)
    if not bool(torch_module.isfinite(reference).all()) or not bool(
        torch_module.isfinite(candidate).all()
    ):
        raise RuntimeError("Echo-WM attention verification produced non-finite values")

    difference = (candidate.float() - reference.float()).abs()
    max_error = float(difference.max())
    mean_error = float(difference.mean())
    reference_rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
    normalized_rmse = float(difference.square().mean().sqrt() / reference_rms)
    if mean_error > 0.01 or normalized_rmse > 0.05:
        raise RuntimeError(
            "FlashAttention 4 differs materially from PyTorch SDPA: "
            f"mean_absolute_error={mean_error:.6f}, normalized_rmse={normalized_rmse:.6f}"
        )

    def measure(attention: Any) -> float:
        for _ in range(warmup_iterations):
            attention(query, key, value, heads)
        torch_module.cuda.synchronize(device)
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measured_iterations):
            attention(query, key, value, heads)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / measured_iterations

    pytorch_ms = measure(pytorch_attention)
    flash_attention_4_ms = measure(flash_attention)
    return AttentionBenchmark(
        pytorch_milliseconds=pytorch_ms,
        flash_attention_4_milliseconds=flash_attention_4_ms,
        speedup=pytorch_ms / flash_attention_4_ms,
        max_absolute_error=max_error,
        mean_absolute_error=mean_error,
        normalized_root_mean_square_error=normalized_rmse,
    )
