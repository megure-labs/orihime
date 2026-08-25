# SPDX-License-Identifier: Apache-2.0
"""Kernel binding for the Monotonic Alignment Search operator."""

from __future__ import annotations

from numbers import Real
from typing import Optional, Tuple

import torch
from torch import Tensor

from .. import _ops
from .._pt2_ops import functorch_tensor_batched
from .._pt2_utils import use_pt2_ops
from ..operator import (
    BackwardPass,
    ForwardPass,
    KernelCall,
    OperatorKernels,
    _graph_safe_assert,
    _validate_derivative_vector,
    _validate_temperature as _validate_temperature_param,
)


def _validate_temperature(scores: Tensor, temp: float | Tensor) -> None:
    if isinstance(temp, Tensor):
        if temp.device != scores.device:
            raise ValueError(
                "temp must be on the same device as scores, got "
                f"{temp.device} vs {scores.device}"
            )
        if temp.dtype != scores.dtype:
            raise ValueError(
                "temp must have the same dtype as scores, got "
                f"{temp.dtype} vs {scores.dtype}"
            )
        _validate_temperature_param(temp)
        return

    if not isinstance(temp, Real):
        raise TypeError("temp must be a real number or tensor")
    _validate_temperature_param(temp)


def _validate_shape(scores: Tensor, lengths: Tensor | None) -> None:
    if scores.ndim != 3:
        raise ValueError(
            f"scores must be 3D [B, T, S], got shape {tuple(scores.shape)}"
        )

    max_t, max_s = scores.shape[-2:]
    if lengths is None:
        if max_t < max_s:
            raise ValueError(
                "mas requires scores.shape[-2] >= scores.shape[-1] "
                "(T >= S), i.e. at least one row per column for a valid "
                f"monotonic alignment; got T={max_t} < S={max_s}"
            )
        return

    if lengths.ndim == 2 and lengths.shape[-1] == 2:
        time_lengths = lengths[:, 0]
        source_lengths = lengths[:, 1]
        in_range = (
            (time_lengths >= 1)
            & (time_lengths <= max_t)
            & (source_lengths >= 1)
            & (source_lengths <= max_s)
        )
        _graph_safe_assert(
            ~(in_range & (time_lengths < source_lengths)),
            "mas requires lengths[:, 0] >= lengths[:, 1] "
            "(T_len >= S_len) for every batch element",
        )


def _validate_call(call: KernelCall) -> float | Tensor:
    (temp,) = call.params
    _validate_temperature(call.primary, temp)
    _validate_shape(call.primary, call.lengths)
    return temp


def _default_lengths(scores: Tensor) -> Tensor:
    batch, max_t, max_s = scores.shape
    return torch.tensor(
        [[max_t, max_s]] * batch,
        dtype=torch.int32,
        device=scores.device,
    )


def _lengths(call: KernelCall) -> Tensor:
    if call.lengths is None:
        return _default_lengths(call.primary)
    return call.lengths


def _temp_value(temp: float | Tensor) -> float | Tensor:
    if isinstance(temp, Tensor):
        if use_pt2_ops(temp) or functorch_tensor_batched(temp):
            return temp
        return float(temp.detach().item())
    return float(temp)


def _promote_grad_map(
    scores: Tensor,
    grad_map: Tensor,
) -> tuple[Tensor, Tensor]:
    if not isinstance(grad_map, Tensor):
        raise TypeError("grad_map must be a tensor")
    if grad_map.shape != scores.shape:
        raise ValueError(
            "grad_map must have the same shape as scores, got "
            f"{tuple(grad_map.shape)} vs {tuple(scores.shape)}"
        )
    if grad_map.device != scores.device:
        raise ValueError(
            "grad_map must be on the same device as scores, got "
            f"{grad_map.device} vs {scores.device}"
        )
    dtype = torch.promote_types(scores.dtype, grad_map.dtype)
    return (
        scores.to(dtype=dtype),
        grad_map.to(dtype=dtype).contiguous(),
    )


def _forward(call: KernelCall) -> ForwardPass:
    temp = _validate_call(call)
    lengths = _lengths(call)
    if isinstance(temp, Tensor):
        score, marginals = _ops.mas_forward_t(call.primary, temp, lengths)
    else:
        score, marginals = _ops.mas_forward(
            call.primary, float(temp), lengths
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward(call: KernelCall, forward: ForwardPass) -> BackwardPass:
    if len(forward.saved_tensors) != 1:
        raise RuntimeError("MAS forward state is missing the marginals")

    temp = _validate_call(call)
    grad_temp = _ops.mas_value_grad_params(
        call.primary,
        _temp_value(temp),
        _lengths(call),
    )
    return BackwardPass(
        # The scaffold marks saved kernel state non-differentiable. Keep the
        # public map distinct so it retains _MapFn's autograd edge.
        marginals=forward.saved_tensors[0].clone(),
        entropy=grad_temp,
        param_grads=(grad_temp,),
    )


def _map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    temp = _validate_call(call)
    scores, grad_map = _promote_grad_map(call.primary, grad_map)
    return _ops.mas_marginals_hvp(
        scores,
        grad_map,
        _temp_value(temp),
        _lengths(call),
    )


def _param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    temp = _validate_call(call)
    if param != "temp":
        raise ValueError(f"mas has no parameter named {param!r}")

    scores = call.primary
    if grad_map is not None:
        scores, grad_map = _promote_grad_map(scores, grad_map)
    sensitivity = _ops.mas_marginals_grad_temp(
        scores,
        _temp_value(temp),
        _lengths(call),
    )
    if grad_map is None:
        return sensitivity

    contracted = (sensitivity * grad_map).sum()
    if isinstance(temp, Tensor):
        return contracted.reshape(temp.shape)
    return contracted.reshape(1)


kernels = OperatorKernels(
    forward=_forward,
    backward=_backward,
    map_backward=_map_backward,
    param_backward=_param_backward,
)


def _raw_forward(
    scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run MAS with a scalar temperature."""

    return _ops.mas_forward(scores, temp, lengths)


def _raw_forward_tensor(
    scores: Tensor,
    temp: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run MAS with a learnable tensor temperature."""

    return _ops.mas_forward_t(scores, temp, lengths)


def _raw_score_param_grads(
    scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the value gradient with respect to temperature."""

    return _ops.mas_value_grad_params(scores, temp, lengths)


def _raw_map_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Backpropagate through MAS marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.
    """

    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.mas_marginals_backward(
        scores, grad_marginals, temp, lengths
    )


def _raw_map_scores_backward(
    scores: Tensor,
    v: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Apply the score-space derivative to ``v``.

    The user-supplied tangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.
    """

    _validate_derivative_vector(
        scores,
        v,
        name="tangent",
        primary_name="scores",
    )
    return _ops.mas_marginals_hvp(scores, v, temp, lengths)


def _raw_map_temp_sensitivity(
    scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the full marginal derivative with respect to temperature."""

    return _ops.mas_marginals_grad_temp(scores, temp, lengths)


# Public low-level kernel bindings.
forward = _raw_forward
forward_t = _raw_forward_tensor
value_grad_params = _raw_score_param_grads
marginals_backward = _raw_map_backward
marginals_hvp = _raw_map_scores_backward
marginals_grad_temp = _raw_map_temp_sensitivity

__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_temp",
]
