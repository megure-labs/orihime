# SPDX-License-Identifier: Apache-2.0
"""Internal Longest Common Subsequence kernel adapter."""

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


def _raw_forward(
    scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with a scalar temperature.

    Args:
        scores: Match-score matrix [B, L1, L2].
        temp: Temperature for the smooth maximum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        value: Soft LCS value [B].
        marginals: Match marginals [B, L1, L2].
    """
    return _ops.lcs_forward(scores, temp, lengths)


def _raw_forward_tensor(
    scores: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with a learnable tensor temperature.

    Args:
        scores: Match-score matrix [B, L1, L2].
        temp: Temperature tensor [1].
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        value: Soft LCS value [B].
        marginals: Match marginals [B, L1, L2].
    """
    return _ops.lcs_forward_t(scores, temp, lengths)


def _raw_score_param_grads(
    scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the value gradient with respect to temperature.

    Args:
        scores: Match-score matrix [B, L1, L2].
        temp: Temperature for the smooth maximum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        grad_temp: Gradient of value with respect to temperature [B].
    """
    return _ops.lcs_value_grad_params(scores, temp, lengths)


def _raw_map_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run a full backward pass through the marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Match-score matrix [B, L1, L2].
        grad_marginals: Gradient with respect to marginals [B, L1, L2].
        temp: Temperature for the smooth maximum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        grad_scores: Gradient with respect to scores [B, L1, L2].
        grad_temp: Gradient with respect to temperature [1].
    """
    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.lcs_marginals_backward(scores, grad_marginals, temp, lengths)


def _raw_map_scores_backward(
    scores: Tensor,
    v: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the Hessian-vector product for the score marginals.

    The user-supplied tangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Match-score matrix [B, L1, L2].
        v: Vector to multiply by the score Hessian [B, L1, L2].
        temp: Temperature for the smooth maximum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        hvp: Hessian-vector product [B, L1, L2].
    """
    _validate_derivative_vector(
        scores,
        v,
        name="tangent",
        primary_name="scores",
    )
    return _ops.lcs_marginals_hvp(scores, v, temp, lengths)


def _raw_map_temp_sensitivity(
    scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginals Jacobian with respect to temperature.

    Args:
        scores: Match-score matrix [B, L1, L2].
        temp: Temperature for the smooth maximum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        jacobian: Derivative of marginals with respect to temperature
            [B, L1, L2].
    """
    return _ops.lcs_marginals_grad_temp(scores, temp, lengths)


def _validate_call(call: KernelCall) -> None:
    temp = call.params[0]
    _validate_temperature_param(temp)
    if isinstance(temp, Tensor):
        if temp.device != call.primary.device:
            raise ValueError(
                "temperature must be on the same device as match_scores"
            )


def _temp_value(call: KernelCall) -> float | Tensor:
    temp = call.params[0]
    if isinstance(temp, Tensor):
        if use_pt2_ops(temp) or functorch_tensor_batched(temp):
            return temp
        return float(temp.detach().item())
    return float(temp)


def _lengths(call: KernelCall) -> Tensor:
    if call.lengths is not None:
        return call.lengths
    batch, length_1, length_2 = call.primary.shape
    return torch.tensor(
        [[length_1, length_2]] * batch,
        dtype=torch.int32,
        device=call.primary.device,
    )


def _forward_pass(call: KernelCall) -> ForwardPass:
    _validate_call(call)
    temp = call.params[0]
    lengths = _lengths(call)
    if isinstance(temp, Tensor):
        score, marginals = _raw_forward_tensor(
            call.primary, temp, lengths
        )
    else:
        score, marginals = _raw_forward(
            call.primary, float(temp), lengths
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward_pass(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    marginals = forward_pass.saved_tensors[0]
    grad_temp = _raw_score_param_grads(
        call.primary, _temp_value(call), _lengths(call)
    )
    temp = call.params[0]
    param_grad = (
        grad_temp.to(dtype=temp.dtype).clone()
        if isinstance(temp, Tensor)
        else grad_temp.clone()
    )
    return BackwardPass(
        marginals=marginals.clone(),
        entropy=grad_temp,
        param_grads=(param_grad,),
    )


def _normalize_grad_map(call: KernelCall, grad_map: Tensor) -> Tensor:
    if not isinstance(grad_map, Tensor):
        raise TypeError("grad_map must be a tensor")
    if grad_map.shape != call.primary.shape:
        raise ValueError("grad_map must have the same shape as match_scores")
    if grad_map.device != call.primary.device:
        raise ValueError(
            "grad_map must be on the same device as match_scores"
        )
    return grad_map.to(dtype=call.primary.dtype).contiguous()


def _map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    return _raw_map_scores_backward(
        call.primary,
        _normalize_grad_map(call, grad_map),
        _temp_value(call),
        _lengths(call),
    )


def _param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    if param != "temp":
        raise ValueError(f"unknown LCS parameter {param!r}")
    sensitivity = _raw_map_temp_sensitivity(
        call.primary, _temp_value(call), _lengths(call)
    )
    if grad_map is None:
        return sensitivity

    cotangent = (sensitivity * _normalize_grad_map(call, grad_map)).sum()
    temp = call.params[0]
    if isinstance(temp, Tensor):
        return cotangent.to(device=temp.device, dtype=temp.dtype).reshape(
            temp.shape
        )
    return cotangent


LCS_KERNELS = OperatorKernels(
    forward=_forward_pass,
    backward=_backward_pass,
    map_backward=_map_backward,
    param_backward=_param_backward,
)


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
