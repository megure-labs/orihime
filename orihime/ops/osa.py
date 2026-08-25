# SPDX-License-Identifier: Apache-2.0
"""Internal Optimal String Alignment kernel adapter."""

from __future__ import annotations

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
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with scalar parameters.

    Args:
        sub_costs: Substitution costs [B, L1, L2].
        trans_mask: Adjacent-transposition mask [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Adjacent-transposition cost.
        temp: Soft-min temperature.
        lengths: Optional actual sequence lengths [B, 2].

    Returns:
        value: Soft OSA distance [B].
        marginals: Substitution marginals [B, L1, L2].
    """
    return _ops.osa_forward(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_forward_tensor(
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: Tensor,
    del_cost: Tensor,
    trans_cost: Tensor,
    temp: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with learnable tensor parameters.

    Args:
        sub_costs: Substitution costs [B, L1, L2].
        trans_mask: Adjacent-transposition mask [B, L1, L2].
        ins_cost: Scalar insertion-cost tensor.
        del_cost: Scalar deletion-cost tensor.
        trans_cost: Scalar adjacent-transposition-cost tensor.
        temp: Scalar soft-min-temperature tensor.
        lengths: Actual sequence lengths [B, 2].

    Returns:
        value: Soft OSA distance [B].
        marginals: Substitution marginals [B, L1, L2].
    """
    return _ops.osa_forward_t(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_score_param_grads(
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute per-batch value gradients with respect to scalar parameters.

    Args:
        sub_costs: Substitution costs [B, L1, L2].
        trans_mask: Adjacent-transposition mask [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Adjacent-transposition cost.
        temp: Soft-min temperature.
        lengths: Optional actual sequence lengths [B, 2].

    Returns:
        grad_ins_cost: Value gradient with respect to insertion cost [B].
        grad_del_cost: Value gradient with respect to deletion cost [B].
        grad_trans_cost: Value gradient with respect to transposition cost [B].
        grad_temp: Value gradient with respect to temperature [B].
    """
    return _ops.osa_value_grad_params(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_backward(
    sub_costs: Tensor,
    trans_mask: Tensor,
    grad_marginals: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run a full backward pass through the substitution marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact
    sub_costs shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        sub_costs: Substitution costs [B, L1, L2].
        trans_mask: Adjacent-transposition mask [B, L1, L2].
        grad_marginals: Upstream gradient for marginals [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Adjacent-transposition cost.
        temp: Soft-min temperature.
        lengths: Optional actual sequence lengths [B, 2].

    Returns:
        grad_sub_costs: Gradient with respect to substitution costs [B, L1, L2].
        grad_ins_cost: Gradient with respect to insertion cost [1].
        grad_del_cost: Gradient with respect to deletion cost [1].
        grad_trans_cost: Gradient with respect to transposition cost [1].
        grad_temp: Gradient with respect to temperature [1].
    """
    _validate_derivative_vector(
        sub_costs,
        grad_marginals,
        name="cotangent",
        primary_name="sub_costs",
    )
    return _ops.osa_marginals_backward(
        sub_costs,
        trans_mask,
        grad_marginals,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_scores_backward(
    sub_costs: Tensor,
    trans_mask: Tensor,
    v: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the substitution-cost Hessian-vector product.

    The user-supplied tangent must be contiguous FP32 with the exact
    sub_costs shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        sub_costs: Substitution costs [B, L1, L2].
        trans_mask: Adjacent-transposition mask [B, L1, L2].
        v: Vector to multiply by the value Hessian [B, L1, L2].
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Adjacent-transposition cost.
        temp: Soft-min temperature.
        lengths: Optional actual sequence lengths [B, 2].

    Returns:
        Hessian-vector product [B, L1, L2].
    """
    _validate_derivative_vector(
        sub_costs,
        v,
        name="tangent",
        primary_name="sub_costs",
    )
    return _ops.osa_marginals_hvp(
        sub_costs,
        trans_mask,
        v,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_ins_cost_sensitivity(
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to insertion cost.

    Returns a [B, L1, L2] tensor containing d(marginals)/d(ins_cost).
    """
    return _ops.osa_marginals_grad_ins_cost(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_del_cost_sensitivity(
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to deletion cost.

    Returns a [B, L1, L2] tensor containing d(marginals)/d(del_cost).
    """
    return _ops.osa_marginals_grad_del_cost(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_trans_cost_sensitivity(
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to transposition cost.

    Returns a [B, L1, L2] tensor containing d(marginals)/d(trans_cost).
    """
    return _ops.osa_marginals_grad_trans_cost(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_temp_sensitivity(
    sub_costs: Tensor,
    trans_mask: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to temperature.

    Returns a [B, L1, L2] tensor containing d(marginals)/d(temp).
    """
    return _ops.osa_marginals_grad_temp(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


# =============================================================================
# Operator binding
# =============================================================================


_PARAM_NAMES = ("ins_cost", "del_cost", "trans_cost", "temp")
_RAW_SENSITIVITY_PARAMS = ("ins_cost", "del_cost", "trans_cost", "temp")


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
    Tensor,
    Tensor,
    float | Tensor,
    float | Tensor,
    float | Tensor,
    float | Tensor,
    Tensor,
]:
    sub_costs = call.primary
    trans_mask = call.config["trans_mask"]
    if trans_mask is None:
        trans_mask = torch.zeros_like(sub_costs)
    elif not isinstance(trans_mask, Tensor):
        raise TypeError("trans_mask must be a tensor or None")

    ins_cost, del_cost, trans_cost, temp = call.params
    _validate_temperature_param(temp)
    ins_value = _scalar_param(ins_cost, "ins_cost")
    del_value = _scalar_param(del_cost, "del_cost")
    trans_value = _scalar_param(trans_cost, "trans_cost")
    temp_value = _scalar_param(temp, "temp")

    lengths = call.lengths
    if lengths is None:
        batch, length_1, length_2 = sub_costs.shape
        lengths = torch.tensor(
            [[length_1, length_2]] * batch,
            dtype=torch.int32,
            device=sub_costs.device,
        )
    return (
        sub_costs,
        trans_mask,
        ins_value,
        del_value,
        trans_value,
        temp_value,
        lengths,
    )


def _operator_forward(call: KernelCall) -> ForwardPass:
    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    ) = _kernel_args(call)
    if any(functorch_tensor_batched(param) for param in call.params):
        tensor_params = tuple(
            param
            if isinstance(param, Tensor)
            else sub_costs.new_tensor(param)
            for param in call.params
        )
        score, marginals = _raw_forward_tensor(
            sub_costs,
            trans_mask,
            *tensor_params,
            lengths,
        )
    else:
        score, marginals = _raw_forward(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temp,
            lengths,
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _param_grad_for_input(
    grad: Tensor,
    param: float | Tensor,
) -> Tensor:
    if isinstance(param, Tensor):
        return grad.to(device=param.device, dtype=param.dtype)
    return grad


def _operator_backward(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    if len(forward_pass.saved_tensors) != 1:
        raise RuntimeError("OSA forward state must contain marginals")

    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    ) = _kernel_args(call)
    param_grads = _raw_score_param_grads(
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )
    normalized_param_grads = tuple(
        _param_grad_for_input(grad, param)
        for grad, param in zip(param_grads, call.params, strict=True)
    )
    marginals = forward_pass.saved_tensors[0].clone()
    return BackwardPass(
        marginals=marginals,
        entropy=-param_grads[3],
        param_grads=normalized_param_grads,
        input_grads=(marginals,),
    )


def _normalized_grad_map(call: KernelCall, grad_map: Tensor) -> Tensor:
    if not isinstance(grad_map, Tensor):
        raise TypeError("grad_map must be a tensor")
    if grad_map.shape != call.primary.shape:
        raise ValueError(
            "grad_map must have the same shape as sub_costs"
        )
    if grad_map.device != call.primary.device:
        raise ValueError(
            "grad_map must be on the same device as sub_costs"
        )
    return grad_map.to(dtype=call.primary.dtype).contiguous()


def _operator_map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    ) = _kernel_args(call)
    return _raw_map_scores_backward(
        sub_costs,
        trans_mask,
        _normalized_grad_map(call, grad_map),
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_sensitivity(
    call: KernelCall,
    param_name: str,
) -> Tensor:
    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    ) = _kernel_args(call)
    if param_name == "ins_cost":
        return _raw_map_ins_cost_sensitivity(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temp,
            lengths,
        )
    if param_name == "del_cost":
        return _raw_map_del_cost_sensitivity(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temp,
            lengths,
        )
    if param_name == "trans_cost":
        return _raw_map_trans_cost_sensitivity(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temp, lengths
        )
    if param_name == "temp":
        return _raw_map_temp_sensitivity(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temp, lengths
        )
    raise ValueError(f"unknown OSA parameter {param_name!r}")


def _operator_param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    if param_name not in _PARAM_NAMES:
        raise ValueError(f"unknown OSA parameter {param_name!r}")
    if grad_map is None:
        return _raw_sensitivity(call, param_name)

    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    ) = _kernel_args(call)
    backward_result = _raw_map_backward(
        sub_costs,
        trans_mask,
        _normalized_grad_map(call, grad_map),
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )
    grad = backward_result[_PARAM_NAMES.index(param_name) + 1]
    param = call.params[_PARAM_NAMES.index(param_name)]
    if isinstance(param, Tensor):
        return grad.to(
            device=param.device,
            dtype=param.dtype,
        ).reshape(param.shape)
    return grad.reshape(1)


KERNELS = OperatorKernels(
    forward=_operator_forward,
    backward=_operator_backward,
    map_backward=_operator_map_backward,
    param_backward=_operator_param_backward,
    config={"trans_mask": None},
)


# Public low-level kernel bindings.
forward = _raw_forward
forward_t = _raw_forward_tensor
value_grad_params = _raw_score_param_grads
marginals_backward = _raw_map_backward
marginals_hvp = _raw_map_scores_backward
marginals_grad_ins_cost = _raw_map_ins_cost_sensitivity
marginals_grad_del_cost = _raw_map_del_cost_sensitivity
marginals_grad_trans_cost = _raw_map_trans_cost_sensitivity
marginals_grad_temp = _raw_map_temp_sensitivity

__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_ins_cost",
    "marginals_grad_del_cost",
    "marginals_grad_trans_cost",
    "marginals_grad_temp",
]
