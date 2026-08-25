# SPDX-License-Identifier: Apache-2.0
"""Internal CKY parsing kernel adapter."""

from __future__ import annotations

from typing import Tuple

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
    merge_scores: Tensor,
    leaf_scores: Tensor,
    temp: float,
) -> Tuple[Tensor, Tensor]:
    """Run CKY with a scalar temperature.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        temp: Positive softmax temperature.

    Returns:
        value: Log partition function [B].
        marginals: Merge marginals [B, N, N, N].
    """
    return _ops.cky_forward(merge_scores, leaf_scores, temp)


def _raw_forward_tensor(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    temp: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run CKY with a differentiable temperature tensor.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        temp: Scalar tensor or local temperature chart [B, N, N].

    Returns:
        value: Log partition function [B].
        marginals: Merge marginals [B, N, N, N].
    """
    return _ops.cky_forward_t(merge_scores, leaf_scores, temp)


def _raw_score_param_grads(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    temp: float,
) -> Tuple[Tensor, Tensor]:
    """Compute value gradients for the leaf chart and temperature.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        temp: Positive softmax temperature.

    Returns:
        grad_leaf: Gradient of value with respect to leaf_scores [B, N].
        grad_temp: Per-batch gradient of value with respect to temp [B].
    """
    return _ops.cky_value_grad_params(merge_scores, leaf_scores, temp)


def _raw_map_backward(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    grad_marginals: Tensor,
    temp: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backpropagate through the merge marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact
    merge_scores shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        grad_marginals: Upstream gradient [B, N, N, N].
        temp: Positive softmax temperature.

    Returns:
        grad_merge: Gradient with respect to merge_scores [B, N, N, N].
        grad_leaf: Gradient with respect to leaf_scores [B, N].
        grad_temp: Gradient with respect to temp [1].
    """
    _validate_derivative_vector(
        merge_scores,
        grad_marginals,
        name="cotangent",
        primary_name="merge_scores",
    )
    return _ops.cky_marginals_backward(
        merge_scores, leaf_scores, grad_marginals, temp
    )


def _raw_map_scores_backward(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    v_merge: Tensor,
    v_leaf: Tensor,
    temp: float,
) -> Tensor:
    """Apply the merge-marginal Jacobian to both chart directions.

    Both user-supplied tangents must be contiguous FP32 tensors with the exact
    merge_scores and leaf_scores shapes and devices; invalid vectors are
    rejected before native dispatch.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        v_merge: Direction for merge_scores [B, N, N, N].
        v_leaf: Direction for leaf_scores [B, N].
        temp: Positive softmax temperature.

    Returns:
        Directional derivative of merge marginals [B, N, N, N].
    """
    _validate_derivative_vector(
        merge_scores,
        v_merge,
        name="merge tangent",
        primary_name="merge_scores",
    )
    _validate_derivative_vector(
        leaf_scores,
        v_leaf,
        name="leaf tangent",
        primary_name="leaf_scores",
    )
    return _ops.cky_marginals_hvp(
        merge_scores, leaf_scores, v_merge, v_leaf, temp
    )


def _raw_map_leaf_sensitivity(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    v_leaf: Tensor,
    temp: float,
) -> Tensor:
    """Apply the merge-marginal leaf-score Jacobian to a direction.

    A full Jacobian with respect to the tensor-valued leaf chart would have
    rank five, so this operation exposes its exact Jacobian-vector product.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        v_leaf: Direction for leaf_scores [B, N].
        temp: Positive softmax temperature.

    Returns:
        Leaf-direction derivative of merge marginals [B, N, N, N].
    """
    return _ops.cky_marginals_grad_leaf(
        merge_scores, leaf_scores, v_leaf, temp
    )


def _raw_map_temp_sensitivity(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    temp: float,
) -> Tensor:
    """Compute the derivative of merge marginals with respect to temperature.

    Args:
        merge_scores: Merge chart [B, N, N, N].
        leaf_scores: Leaf chart [B, N].
        temp: Positive softmax temperature.

    Returns:
        Temperature derivative of merge marginals [B, N, N, N].
    """
    return _ops.cky_marginals_grad_temp(merge_scores, leaf_scores, temp)


def _temperature_value(call: KernelCall) -> float | Tensor:
    temp = call.params[0]
    _validate_temperature_param(temp)
    if isinstance(temp, Tensor):
        if use_pt2_ops(temp) or functorch_tensor_batched(temp):
            return temp
        return float(temp.detach().item())
    return float(temp)


def _charts(call: KernelCall) -> tuple[Tensor, Tensor]:
    merge_scores, leaf_scores = call.tensor_inputs
    return merge_scores, leaf_scores


def _operator_forward(call: KernelCall) -> ForwardPass:
    merge_scores, leaf_scores = _charts(call)
    temp = _temperature_value(call)
    if functorch_tensor_batched(call.params[0]):
        score, marginals = _raw_forward_tensor(
            merge_scores,
            leaf_scores,
            call.params[0],
        )
    else:
        score, marginals = _raw_forward(
            merge_scores,
            leaf_scores,
            temp,
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _operator_backward(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    if len(forward_pass.saved_tensors) != 1:
        raise RuntimeError("CKY forward state must contain marginals")

    merge_scores, leaf_scores = _charts(call)
    grad_leaf, grad_temp = _raw_score_param_grads(
        merge_scores,
        leaf_scores,
        _temperature_value(call),
    )
    temp = call.params[0]
    if isinstance(temp, Tensor):
        grad_temp = grad_temp.to(dtype=temp.dtype)

    # Keep the public map distinct from the opaque forward state marked
    # non-differentiable by _MapFn.
    marginals = forward_pass.saved_tensors[0].clone()
    return BackwardPass(
        marginals=marginals,
        entropy=grad_temp,
        param_grads=(grad_temp,),
        input_grads=(marginals, grad_leaf),
    )


def _operator_map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> tuple[Tensor, Tensor]:
    del state
    merge_scores, leaf_scores = _charts(call)
    grad_merge, grad_leaf, _ = _raw_map_backward(
        merge_scores,
        leaf_scores,
        grad_map.contiguous(),
        _temperature_value(call),
    )
    return grad_merge, grad_leaf


def _operator_map_jvp(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    tangents: tuple[Tensor | None, ...],
) -> Tensor:
    merge_tangent, leaf_tangent = tangents
    result = None
    if merge_tangent is not None:
        result = _operator_map_backward(
            call,
            state,
            merge_tangent,
        )[0]
    if leaf_tangent is not None:
        merge_scores, leaf_scores = _charts(call)
        leaf_result = _raw_map_leaf_sensitivity(
            merge_scores,
            leaf_scores,
            leaf_tangent.contiguous(),
            _temperature_value(call),
        )
        result = leaf_result if result is None else result + leaf_result
    if result is None:
        merge_scores, _ = _charts(call)
        return merge_scores.new_zeros(merge_scores.shape)
    return result


def _operator_param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    if param_name != "temp":
        raise ValueError(f"unknown CKY parameter {param_name!r}")

    merge_scores, leaf_scores = _charts(call)
    sensitivity = _raw_map_temp_sensitivity(
        merge_scores,
        leaf_scores,
        _temperature_value(call),
    )
    if grad_map is None:
        return sensitivity
    if grad_map.shape != merge_scores.shape:
        raise ValueError("grad_map must have the same shape as merge_scores")

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
    map_jvp=_operator_map_jvp,
)


# Public low-level kernel bindings.
forward = _raw_forward
forward_t = _raw_forward_tensor
value_grad_params = _raw_score_param_grads
marginals_backward = _raw_map_backward
marginals_hvp = _raw_map_scores_backward
marginals_grad_leaf = _raw_map_leaf_sensitivity
marginals_grad_temp = _raw_map_temp_sensitivity

__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_leaf",
    "marginals_grad_temp",
]
