# SPDX-License-Identifier: Apache-2.0
"""Internal affine Needleman-Wunsch kernel adapter."""

from __future__ import annotations

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
    Param,
    _validate_derivative_vector,
    _validate_temperature as _validate_temperature_param,
)


PARAMS = ("gap_open", "gap_ext", "temp")


def _validate_temperature(temp: Param) -> None:
    _validate_temperature_param(temp)


def _validate_gap(gap: Param, name: str) -> None:
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"{name}={gap} is positive. Gap penalties are typically negative.",
            UserWarning,
            stacklevel=4,
        )


def _scalar_value(param: Param, name: str) -> float | Tensor:
    if isinstance(param, Tensor):
        if param.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor")
        if use_pt2_ops(param) or functorch_tensor_batched(param):
            return param
        return float(param.detach().item())
    return float(param)


def _lengths(call: KernelCall) -> Tensor:
    if call.lengths is None:
        batch, length1, length2 = call.primary.shape
        return (
            torch.tensor(
                (length1, length2),
                dtype=torch.int32,
                device=call.primary.device,
            )
            .expand(batch, 2)
            .contiguous()
        )
    return call.lengths


def _kernel_args(
    call: KernelCall,
) -> tuple[float | Tensor, float | Tensor, float | Tensor, Tensor]:
    gap_open, gap_ext, temp = call.params
    return (
        _scalar_value(gap_open, "gap_open"),
        _scalar_value(gap_ext, "gap_ext"),
        _scalar_value(temp, "temp"),
        _lengths(call),
    )


def _tensor_param(param: Param, scores: Tensor) -> Tensor:
    if isinstance(param, Tensor):
        return param
    return scores.new_tensor([param])


def _validate_call(call: KernelCall) -> None:
    gap_open, gap_ext, temp = call.params
    _validate_temperature(temp)
    _validate_gap(gap_open, "gap_open")
    _validate_gap(gap_ext, "gap_ext")


def _forward(call: KernelCall) -> ForwardPass:
    _validate_call(call)
    gap_open, gap_ext, temp = call.params
    lengths = _lengths(call)
    if any(isinstance(param, Tensor) for param in call.params) or use_pt2_ops(
        call.primary, *call.params
    ):
        score, marginals = _ops.nw_affine_forward_t(
            call.primary,
            _tensor_param(gap_open, call.primary),
            _tensor_param(gap_ext, call.primary),
            _tensor_param(temp, call.primary),
            lengths,
        )
    else:
        score, marginals = _ops.nw_affine_forward(
            call.primary,
            float(gap_open),
            float(gap_ext),
            float(temp),
            lengths,
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward(call: KernelCall, forward: ForwardPass) -> BackwardPass:
    gap_open, gap_ext, temp, lengths = _kernel_args(call)
    _, grad_gap_open, grad_gap_ext, grad_temp = (
        _ops.nw_affine_value_grad_params(
            call.primary, gap_open, gap_ext, temp, lengths
        )
    )
    return BackwardPass(
        # _MapFn also emits the forward state as non-differentiable outputs.
        # Keep the public map distinct from that saved-state tensor.
        marginals=forward.saved_tensors[0].clone(),
        entropy=grad_temp,
        param_grads=(grad_gap_open, grad_gap_ext, grad_temp),
    )


def _normalize_grad_map(call: KernelCall, grad_map: Tensor) -> Tensor:
    if grad_map.shape != call.primary.shape:
        raise ValueError("grad_map must have the same shape as scores")
    if grad_map.device != call.primary.device:
        raise ValueError("grad_map must be on the same device as scores")
    return grad_map.to(dtype=call.primary.dtype).contiguous()


def _map_backward(
    call: KernelCall,
    _state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    gap_open, gap_ext, temp, lengths = _kernel_args(call)
    return _ops.nw_affine_marginals_hvp(
        call.primary,
        _normalize_grad_map(call, grad_map),
        gap_open,
        gap_ext,
        temp,
        lengths,
    )


_SENSITIVITY_KERNELS = {
    "gap_open": _ops.nw_affine_marginals_grad_gap_open,
    "gap_ext": _ops.nw_affine_marginals_grad_gap_ext,
    "temp": _ops.nw_affine_marginals_grad_temp,
}


def _param_backward(
    call: KernelCall,
    _state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    gap_open, gap_ext, temp, lengths = _kernel_args(call)
    sensitivity = _SENSITIVITY_KERNELS[param_name](
        call.primary, gap_open, gap_ext, temp, lengths
    )
    if grad_map is None:
        return sensitivity

    contracted = (
        sensitivity * _normalize_grad_map(call, grad_map)
    ).sum().reshape(1)
    param = call.params[PARAMS.index(param_name)]
    if isinstance(param, Tensor):
        return contracted.to(
            dtype=param.dtype, device=param.device
        ).sum_to_size(param.shape)
    return contracted


KERNELS = OperatorKernels(
    forward=_forward,
    backward=_backward,
    map_backward=_map_backward,
    param_backward=_param_backward,
)


def _raw_forward(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with scalar parameters."""

    return _ops.nw_affine_forward(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_forward_tensor(
    scores: Tensor,
    gap_open: Tensor,
    gap_ext: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with tensor parameters."""

    return _ops.nw_affine_forward_t(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_score_param_grads(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the value and its gradients for all scoring parameters."""

    return _ops.nw_affine_value_grad_params(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_map_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the full backward pass through the marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.
    """

    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.nw_affine_marginals_backward(
        scores, grad_marginals, gap_open, gap_ext, temp, lengths
    )


def _raw_map_scores_backward(
    scores: Tensor,
    v: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the score-space Hessian-vector product.

    The user-supplied tangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.
    """

    _validate_derivative_vector(
        scores,
        v,
        name="tangent",
        primary_name="scores",
    )
    return _ops.nw_affine_marginals_hvp(
        scores, v, gap_open, gap_ext, temp, lengths
    )


def _raw_map_gap_open_sensitivity(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginals sensitivity to the gap-opening penalty."""

    return _ops.nw_affine_marginals_grad_gap_open(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_map_gap_ext_sensitivity(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginals sensitivity to the gap-extension penalty."""

    return _ops.nw_affine_marginals_grad_gap_ext(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_map_temp_sensitivity(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginals sensitivity to temperature."""

    return _ops.nw_affine_marginals_grad_temp(
        scores, gap_open, gap_ext, temp, lengths
    )


# Public low-level kernel bindings.
forward = _raw_forward
forward_t = _raw_forward_tensor
value_grad_params = _raw_score_param_grads
marginals_backward = _raw_map_backward
marginals_hvp = _raw_map_scores_backward
marginals_grad_gap_open = _raw_map_gap_open_sensitivity
marginals_grad_gap_ext = _raw_map_gap_ext_sensitivity
marginals_grad_temp = _raw_map_temp_sensitivity

__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_gap_open",
    "marginals_grad_gap_ext",
    "marginals_grad_temp",
]
