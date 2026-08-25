# SPDX-License-Identifier: Apache-2.0
"""Internal Needleman-Wunsch kernel adapter."""

import warnings
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
    _validate_derivative_vector,
    _validate_temperature as _validate_temperature_param,
)


def _validate_temperature(temp: float | Tensor) -> None:
    _validate_temperature_param(temp)


def _validate_gap(gap: float | Tensor) -> None:
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"gap={gap} is positive. Gap penalties are typically negative.",
            UserWarning,
            stacklevel=4,
        )


def _make_lengths(scores: Tensor) -> Tensor:
    batch, length1, length2 = scores.shape
    return torch.tensor(
        [[length1, length2]] * batch,
        dtype=torch.int32,
        device=scores.device,
    )


def _lengths(call: KernelCall) -> Tensor:
    return _make_lengths(call.primary) if call.lengths is None else call.lengths


def _scalar_params(
    call: KernelCall,
) -> tuple[float | Tensor, float | Tensor]:
    gap, temp = call.params

    def scalar(value: float | Tensor) -> float | Tensor:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("NW scoring parameters must be scalar tensors")
            if use_pt2_ops(value) or functorch_tensor_batched(value):
                return value
            return float(value.detach().item())
        return float(value)

    return scalar(gap), scalar(temp)


def _tensor_param(value: float | Tensor, scores: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return scores.new_tensor([value])


def _forward_kernel(call: KernelCall) -> ForwardPass:
    gap, temp = call.params
    _validate_gap(gap)
    _validate_temperature(temp)
    lengths = _lengths(call)

    if isinstance(gap, Tensor) or isinstance(temp, Tensor):
        score, marginals = _ops.nw_forward_t(
            call.primary,
            _tensor_param(gap, call.primary),
            _tensor_param(temp, call.primary),
            lengths,
        )
    else:
        score, marginals = _ops.nw_forward(
            call.primary,
            float(gap),
            float(temp),
            lengths,
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward_kernel(
    call: KernelCall,
    forward: ForwardPass,
) -> BackwardPass:
    if len(forward.saved_tensors) != 1:
        raise RuntimeError("NW forward state is missing the marginals")

    gap, temp = _scalar_params(call)
    grad_gap, grad_temp = _ops.nw_value_grad_params(
        call.primary,
        gap,
        temp,
        _lengths(call),
    )
    return BackwardPass(
        # The shared scaffold marks saved forward state non-differentiable.
        # Keep the public map as a distinct tensor so it retains _MapFn's
        # autograd edge.
        marginals=forward.saved_tensors[0].clone(),
        entropy=grad_temp,
        param_grads=(grad_gap, grad_temp),
    )


def _map_backward_kernel(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    gap, temp = _scalar_params(call)
    return _ops.nw_marginals_hvp(
        call.primary,
        _normalize_grad_map(call, grad_map),
        gap,
        temp,
        _lengths(call),
    )


def _normalize_grad_map(call: KernelCall, grad_map: Tensor) -> Tensor:
    if grad_map.shape != call.primary.shape:
        raise ValueError("grad_map must have the same shape as scores")
    if grad_map.device != call.primary.device:
        raise ValueError("grad_map must be on the same device as scores")
    return grad_map.to(dtype=call.primary.dtype).contiguous()


def _param_backward_kernel(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    gap, temp = _scalar_params(call)
    if param == "gap":
        sensitivity = _ops.nw_marginals_grad_gap(
            call.primary, gap, temp, _lengths(call)
        )
        original_param = call.params[0]
    elif param == "temp":
        sensitivity = _ops.nw_marginals_grad_temp(
            call.primary, gap, temp, _lengths(call)
        )
        original_param = call.params[1]
    else:
        raise ValueError(f"unknown NW parameter: {param!r}")

    if grad_map is None:
        return sensitivity

    contracted = (sensitivity * _normalize_grad_map(call, grad_map)).sum()
    if isinstance(original_param, Tensor):
        return contracted.to(original_param).reshape(original_param.shape)
    return contracted.reshape(1)


kernels = OperatorKernels(
    forward=_forward_kernel,
    backward=_backward_kernel,
    map_backward=_map_backward_kernel,
    param_backward=_param_backward_kernel,
)


def _raw_forward(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with scalar parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty (typically negative)
        temp: Temperature for soft-max
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.nw_forward(scores, gap, temp, lengths)


def _raw_forward_tensor(
    scores: Tensor,
    gap: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with tensor parameters (for learnable params).

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty tensor [1]
        temp: Temperature tensor [1]
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.nw_forward_t(scores, gap, temp, lengths)


def _raw_score_param_grads(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Gradients of value w.r.t. parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_gap: Gradient of value w.r.t. gap [B]
        grad_temp: Gradient of value w.r.t. temperature [B]
    """
    return _ops.nw_value_grad_params(scores, gap, temp, lengths)


def _raw_map_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Full backward through marginals.

    Computes gradients of loss (through marginals) w.r.t. all inputs.
    The user-supplied cotangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Similarity matrix [B, L1, L2]
        grad_marginals: Gradient w.r.t. marginals [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_scores: Gradient w.r.t. scores [B, L1, L2]
        grad_gap: Gradient w.r.t. gap [B]
        grad_temp: Gradient w.r.t. temperature [B]
    """
    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.nw_marginals_backward(scores, grad_marginals, gap, temp, lengths)


def _raw_map_scores_backward(
    scores: Tensor,
    v: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Hessian-vector product: H @ v where H = d^2value/dscores^2.

    This efficiently computes the action of the Hessian on a vector
    without forming the full O(L^4) Hessian matrix.
    The user-supplied tangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Similarity matrix [B, L1, L2]
        v: Vector to multiply with Hessian [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        hvp: Hessian-vector product [B, L1, L2]
    """
    _validate_derivative_vector(
        scores,
        v,
        name="tangent",
        primary_name="scores",
    )
    return _ops.nw_marginals_hvp(scores, v, gap, temp, lengths)


def _raw_map_gap_sensitivity(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. gap (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dgap.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dgap [B, L1, L2]
    """
    return _ops.nw_marginals_grad_gap(scores, gap, temp, lengths)


def _raw_map_temp_sensitivity(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. temperature (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dtemp.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dtemp [B, L1, L2]
    """
    return _ops.nw_marginals_grad_temp(scores, gap, temp, lengths)


# Public low-level kernel bindings.
forward = _raw_forward
forward_t = _raw_forward_tensor
value_grad_params = _raw_score_param_grads
marginals_backward = _raw_map_backward
marginals_hvp = _raw_map_scores_backward
marginals_grad_gap = _raw_map_gap_sensitivity
marginals_grad_temp = _raw_map_temp_sensitivity

__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_gap",
    "marginals_grad_temp",
]
