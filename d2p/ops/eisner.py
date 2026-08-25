# SPDX-License-Identifier: Apache-2.0
"""Internal Eisner dependency-parsing kernel adapter."""

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
    arc_scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with a scalar temperature.

    Args:
        arc_scores: Arc-score chart ``[B, N, N]``, where entry ``[i, j]``
            scores the directed arc from token ``i`` to token ``j``.
        temp: Temperature for the soft dynamic program.
        lengths: Optional ``[B]`` tensor of actual sequence lengths.

    Returns:
        value: Log-partition value ``[B]``.
        marginals: Arc marginals ``[B, N, N]``.
    """
    return _ops.eisner_forward(arc_scores, temp, lengths)


def _raw_forward_tensor(
    arc_scores: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with a tensor temperature.

    Args:
        arc_scores: Arc-score chart ``[B, N, N]``.
        temp: Scalar temperature tensor.
        lengths: Optional ``[B]`` tensor of actual sequence lengths.

    Returns:
        value: Log-partition value ``[B]``.
        marginals: Arc marginals ``[B, N, N]``.
    """
    return _ops.eisner_forward_t(arc_scores, temp, lengths)


def _raw_score_param_grads(
    arc_scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the value gradient with respect to temperature.

    Args:
        arc_scores: Arc-score chart ``[B, N, N]``.
        temp: Temperature for the soft dynamic program.
        lengths: Optional ``[B]`` tensor of actual sequence lengths.

    Returns:
        Gradient of each batch value with respect to temperature, shaped
        ``[B]``.
    """
    return _ops.eisner_value_grad_params(arc_scores, temp, lengths)


def _raw_map_backward(
    arc_scores: Tensor,
    grad_marginals: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Backpropagate through the arc marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact
    arc_scores shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        arc_scores: Arc-score chart ``[B, N, N]``.
        grad_marginals: Incoming marginal gradient ``[B, N, N]``.
        temp: Temperature for the soft dynamic program.
        lengths: Optional ``[B]`` tensor of actual sequence lengths.

    Returns:
        grad_arc_scores: Gradient with respect to ``arc_scores``.
        grad_temp: Gradient with respect to the scalar temperature, shaped
            ``[1]``.
    """
    _validate_derivative_vector(
        arc_scores,
        grad_marginals,
        name="cotangent",
        primary_name="arc_scores",
    )
    return _ops.eisner_marginals_backward(
        arc_scores, grad_marginals, temp, lengths
    )


def _raw_map_scores_backward(
    arc_scores: Tensor,
    v: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the score Hessian-vector product for the marginals.

    The user-supplied tangent must be contiguous FP32 with the exact
    arc_scores shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        arc_scores: Arc-score chart ``[B, N, N]``.
        v: Score-space tangent ``[B, N, N]``.
        temp: Temperature for the soft dynamic program.
        lengths: Optional ``[B]`` tensor of actual sequence lengths.

    Returns:
        Hessian-vector product ``[B, N, N]``.
    """
    _validate_derivative_vector(
        arc_scores,
        v,
        name="tangent",
        primary_name="arc_scores",
    )
    return _ops.eisner_marginals_hvp(arc_scores, v, temp, lengths)


def _raw_map_temp_sensitivity(
    arc_scores: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the derivative of arc marginals with respect to temperature.

    Args:
        arc_scores: Arc-score chart ``[B, N, N]``.
        temp: Temperature for the soft dynamic program.
        lengths: Optional ``[B]`` tensor of actual sequence lengths.

    Returns:
        Temperature Jacobian ``[B, N, N]``.
    """
    return _ops.eisner_marginals_grad_temp(arc_scores, temp, lengths)


def _temperature_value(call: KernelCall) -> float | Tensor:
    temp = call.params[0]
    _validate_temperature_param(temp)
    if isinstance(temp, Tensor):
        if use_pt2_ops(temp) or functorch_tensor_batched(temp):
            return temp
        return float(temp.detach().item())
    return float(temp)


def _lengths(call: KernelCall) -> Tensor:
    if call.lengths is not None:
        return call.lengths
    batch, size, _ = call.primary.shape
    return torch.full(
        (batch,),
        size,
        dtype=torch.int32,
        device=call.primary.device,
    )


def _operator_forward(call: KernelCall) -> ForwardPass:
    temp = _temperature_value(call)
    if functorch_tensor_batched(call.params[0]):
        score, marginals = _raw_forward_tensor(
            call.primary,
            call.params[0],
            _lengths(call),
        )
    else:
        score, marginals = _raw_forward(
            call.primary,
            temp,
            _lengths(call),
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _operator_backward(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    if len(forward_pass.saved_tensors) != 1:
        raise RuntimeError("Eisner forward state must contain marginals")

    # _MapFn returns the public map alongside the forward state and marks only
    # the latter non-differentiable. Keep those outputs distinct so marking the
    # saved state cannot also mark the public map through tensor aliasing.
    marginals = forward_pass.saved_tensors[0].clone()
    entropy = _raw_score_param_grads(
        call.primary,
        _temperature_value(call),
        _lengths(call),
    )
    param_grad = entropy
    temp = call.params[0]
    if isinstance(temp, Tensor):
        param_grad = param_grad.to(dtype=temp.dtype)
    return BackwardPass(
        marginals=marginals,
        entropy=entropy,
        param_grads=(param_grad,),
    )


def _operator_map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    return _raw_map_scores_backward(
        call.primary,
        grad_map.contiguous(),
        _temperature_value(call),
        _lengths(call),
    )


def _operator_param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    if param_name != "temp":
        raise ValueError(f"unknown Eisner parameter {param_name!r}")

    sensitivity = _raw_map_temp_sensitivity(
        call.primary,
        _temperature_value(call),
        _lengths(call),
    )
    if grad_map is None:
        return sensitivity
    if grad_map.shape != call.primary.shape:
        raise ValueError("grad_map must have the same shape as arc_scores")

    contracted = (grad_map * sensitivity).sum()
    temp = call.params[0]
    if isinstance(temp, Tensor):
        return contracted.to(
            device=temp.device,
            dtype=temp.dtype,
        ).reshape(temp.shape)
    return contracted.reshape(1)


kernels = OperatorKernels(
    forward=_operator_forward,
    backward=_operator_backward,
    map_backward=_operator_map_backward,
    param_backward=_operator_param_backward,
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
