# SPDX-License-Identifier: Apache-2.0
"""Internal Levenshtein kernel adapter."""

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
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with scalar parameters.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        value: Soft edit distance [B].
        marginals: Substitution marginals [B, L1, L2].
    """
    return _ops.lev_forward(scores, ins_cost, del_cost, temp, lengths)


def _raw_forward_tensor(
    scores: Tensor,
    ins_cost: Tensor,
    del_cost: Tensor,
    temp: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with tensor parameters.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        ins_cost: Scalar insertion-cost tensor.
        del_cost: Scalar deletion-cost tensor.
        temp: Scalar temperature tensor.
        lengths: [B, 2] tensor of actual sequence lengths.

    Returns:
        value: Soft edit distance [B].
        marginals: Substitution marginals [B, L1, L2].
    """
    return _ops.lev_forward_t(scores, ins_cost, del_cost, temp, lengths)


def _raw_score_param_grads(
    scores: Tensor,
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute gradients of the value with respect to scalar parameters.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        grad_ins_cost: Gradient of value with respect to ins_cost [B].
        grad_del_cost: Gradient of value with respect to del_cost [B].
        grad_temp: Gradient of value with respect to temp [B].
    """
    return _ops.lev_value_grad_params(
        scores, ins_cost, del_cost, temp, lengths
    )


def _raw_map_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Backpropagate through the substitution marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        grad_marginals: Gradient with respect to marginals [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        grad_scores: Gradient with respect to scores [B, L1, L2].
        grad_ins_cost: Gradient with respect to ins_cost [1].
        grad_del_cost: Gradient with respect to del_cost [1].
        grad_temp: Gradient with respect to temp [1].
    """
    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.lev_marginals_backward(
        scores, grad_marginals, ins_cost, del_cost, temp, lengths
    )


def _raw_map_scores_backward(
    scores: Tensor,
    v: Tensor,
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute a Hessian-vector product through the marginals.

    This computes H @ v, where H is the derivative of substitution marginals
    with respect to the substitution-cost matrix.
    The user-supplied tangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        v: Vector to multiply by the Hessian [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        Hessian-vector product [B, L1, L2].
    """
    _validate_derivative_vector(
        scores,
        v,
        name="tangent",
        primary_name="scores",
    )
    return _ops.lev_marginals_hvp(
        scores, v, ins_cost, del_cost, temp, lengths
    )


def _raw_map_ins_sensitivity(
    scores: Tensor,
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the gradient of marginals with respect to insertion cost.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        Gradient of marginals with respect to ins_cost [B, L1, L2].
    """
    return _ops.lev_marginals_grad_ins_cost(
        scores, ins_cost, del_cost, temp, lengths
    )


def _raw_map_del_sensitivity(
    scores: Tensor,
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the gradient of marginals with respect to deletion cost.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        Gradient of marginals with respect to del_cost [B, L1, L2].
    """
    return _ops.lev_marginals_grad_del_cost(
        scores, ins_cost, del_cost, temp, lengths
    )


def _raw_map_temp_sensitivity(
    scores: Tensor,
    ins_cost: float,
    del_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the gradient of marginals with respect to temperature.

    Args:
        scores: Substitution-cost matrix [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        temp: Temperature for the soft minimum.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        Gradient of marginals with respect to temp [B, L1, L2].
    """
    return _ops.lev_marginals_grad_temp(
        scores, ins_cost, del_cost, temp, lengths
    )


# =============================================================================
# Operator binding
# =============================================================================


def _scalar_param(param: float | Tensor, name: str) -> float | Tensor:
    if isinstance(param, Tensor):
        if param.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor")
        if use_pt2_ops(param) or functorch_tensor_batched(param):
            return param
        return float(param.detach().item())
    return float(param)


def _kernel_args(
    call: KernelCall,
) -> tuple[
    float | Tensor,
    float | Tensor,
    float | Tensor,
    Optional[Tensor],
]:
    ins, delete, temp = call.params
    _validate_temperature_param(temp)
    ins_value = _scalar_param(ins, "ins")
    del_value = _scalar_param(delete, "del")
    temp_value = _scalar_param(temp, "temp")
    lengths = call.lengths
    if lengths is None:
        batch, length_1, length_2 = call.primary.shape
        lengths = torch.tensor(
            [[length_1, length_2]] * batch,
            dtype=torch.int32,
            device=call.primary.device,
        )
    return ins_value, del_value, temp_value, lengths


def _tensor_param(param: float | Tensor, primary: Tensor) -> Tensor:
    if isinstance(param, Tensor):
        return param
    return primary.new_tensor(param)


def _forward_pass(call: KernelCall) -> ForwardPass:
    ins, delete, temp, lengths = _kernel_args(call)
    if any(functorch_tensor_batched(param) for param in call.params):
        score, marginals = _raw_forward_tensor(
            call.primary,
            *(
                _tensor_param(param, call.primary)
                for param in call.params
            ),
            lengths,
        )
    else:
        score, marginals = _raw_forward(
            call.primary, ins, delete, temp, lengths
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward_pass(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    ins, delete, temp, lengths = _kernel_args(call)
    grad_ins, grad_del, grad_temp = _ops.lev_value_grad_params(
        call.primary, ins, delete, temp, lengths
    )
    return BackwardPass(
        # Keep the public map distinct from the forward state.  _MapFn marks
        # its state outputs non-differentiable, and PyTorch applies that mark
        # to aliases of those outputs as well.
        marginals=forward_pass.saved_tensors[0].clone(),
        entropy=-grad_temp,
        param_grads=(grad_ins, grad_del, grad_temp),
    )


def _map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    ins, delete, temp, lengths = _kernel_args(call)
    return _ops.lev_marginals_hvp(
        call.primary,
        grad_map.contiguous(),
        ins,
        delete,
        temp,
        lengths,
    )


def _param_field(
    call: KernelCall,
    param_name: str,
    ins: float | Tensor,
    delete: float | Tensor,
    temp: float | Tensor,
    lengths: Optional[Tensor],
) -> Tensor:
    if param_name == "ins":
        return _ops.lev_marginals_grad_ins_cost(
            call.primary, ins, delete, temp, lengths
        )
    if param_name == "del":
        return _ops.lev_marginals_grad_del_cost(
            call.primary, ins, delete, temp, lengths
        )
    if param_name == "temp":
        return _ops.lev_marginals_grad_temp(
            call.primary, ins, delete, temp, lengths
        )
    raise ValueError(f"unknown lev parameter {param_name!r}")


def _param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Optional[Tensor],
) -> Tensor:
    del state
    ins, delete, temp, lengths = _kernel_args(call)
    field = _param_field(
        call, param_name, ins, delete, temp, lengths
    )
    if grad_map is None:
        return field

    contracted = field * grad_map
    param = call.params[("ins", "del", "temp").index(param_name)]
    if isinstance(param, Tensor):
        return contracted.sum_to_size(param.shape)
    return contracted.sum().reshape(1)


KERNELS = OperatorKernels(
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
marginals_grad_ins_cost = _raw_map_ins_sensitivity
marginals_grad_del_cost = _raw_map_del_sensitivity
marginals_grad_temp = _raw_map_temp_sensitivity

__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_ins_cost",
    "marginals_grad_del_cost",
    "marginals_grad_temp",
]
