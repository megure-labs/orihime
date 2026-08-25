# SPDX-License-Identifier: Apache-2.0
"""Internal true Damerau-Levenshtein kernel adapter."""

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
    trans_src: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with scalar parameters.

    Args:
        sub_costs: Substitution costs with shape ``[B, L1, L2]``.
        trans_src: Transposition source indices with shape ``[B, L1, L2, 2]``.
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Transposition cost.
        temp: Softmin temperature.
        lengths: Optional actual sequence lengths with shape ``[B, 2]``.

    Returns:
        The per-batch soft edit distance and substitution marginals.
    """
    return _ops.damerau_forward(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_forward_tensor(
    sub_costs: Tensor,
    trans_src: Tensor,
    ins_cost: Tensor,
    del_cost: Tensor,
    trans_cost: Tensor,
    temp: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run the forward pass with learnable scalar tensors.

    Args:
        sub_costs: Substitution costs with shape ``[B, L1, L2]``.
        trans_src: Transposition source indices with shape ``[B, L1, L2, 2]``.
        ins_cost: Scalar insertion-cost tensor.
        del_cost: Scalar deletion-cost tensor.
        trans_cost: Scalar transposition-cost tensor.
        temp: Scalar softmin-temperature tensor.
        lengths: Actual sequence lengths with shape ``[B, 2]``.

    Returns:
        The per-batch soft edit distance and substitution marginals.
    """
    return _ops.damerau_forward_t(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_score_param_grads(
    sub_costs: Tensor,
    trans_src: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute value gradients with respect to all scalar parameters.

    Returns:
        Gradients with respect to ``ins_cost``, ``del_cost``, ``trans_cost``,
        and ``temp``, each with shape ``[B]``.
    """
    return _ops.damerau_value_grad_params(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_map_backward(
    sub_costs: Tensor,
    trans_src: Tensor,
    grad_marginals: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Backpropagate through the substitution marginals.

    The user-supplied cotangent must be contiguous FP32 with the exact
    sub_costs shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        sub_costs: Substitution costs with shape ``[B, L1, L2]``.
        trans_src: Transposition source indices with shape ``[B, L1, L2, 2]``.
        grad_marginals: Upstream marginal gradients with shape ``[B, L1, L2]``.
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Transposition cost.
        temp: Softmin temperature.
        lengths: Optional actual sequence lengths with shape ``[B, 2]``.

    Returns:
        Gradients with respect to ``sub_costs``, ``ins_cost``, ``del_cost``,
        ``trans_cost``, and ``temp``.
    """
    _validate_derivative_vector(
        sub_costs,
        grad_marginals,
        name="cotangent",
        primary_name="sub_costs",
    )
    return _ops.damerau_marginals_backward(
        sub_costs,
        trans_src,
        grad_marginals,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _raw_map_scores_backward(
    sub_costs: Tensor,
    trans_src: Tensor,
    v: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute a Hessian-vector product for the substitution costs.

    The user-supplied tangent must be contiguous FP32 with the exact
    sub_costs shape and device; invalid vectors are rejected before native
    dispatch.

    Args:
        sub_costs: Substitution costs with shape ``[B, L1, L2]``.
        trans_src: Transposition source indices with shape ``[B, L1, L2, 2]``.
        v: Tangent tensor with shape ``[B, L1, L2]``.
        ins_cost: Insertion cost.
        del_cost: Deletion cost.
        trans_cost: Transposition cost.
        temp: Softmin temperature.
        lengths: Optional actual sequence lengths with shape ``[B, 2]``.

    Returns:
        The Hessian-vector product with shape ``[B, L1, L2]``.
    """
    _validate_derivative_vector(
        sub_costs,
        v,
        name="tangent",
        primary_name="sub_costs",
    )
    return _ops.damerau_marginals_hvp(
        sub_costs, trans_src, v, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_map_ins_cost_sensitivity(
    sub_costs: Tensor,
    trans_src: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to insertion cost."""
    return _ops.damerau_marginals_grad_ins_cost(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_map_del_cost_sensitivity(
    sub_costs: Tensor,
    trans_src: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to deletion cost."""
    return _ops.damerau_marginals_grad_del_cost(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_map_trans_cost_sensitivity(
    sub_costs: Tensor,
    trans_src: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to transposition cost."""
    return _ops.damerau_marginals_grad_trans_cost(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


def _raw_map_temp_sensitivity(
    sub_costs: Tensor,
    trans_src: Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Compute the marginal Jacobian with respect to temperature."""
    return _ops.damerau_marginals_grad_temp(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temp, lengths
    )


# =============================================================================
# Operator binding
# =============================================================================


_PARAM_NAMES = ("ins_cost", "del_cost", "trans_cost", "temp")


def _inputs(call: KernelCall) -> tuple[Tensor, Tensor]:
    sub_costs = call.primary
    trans_src = call.config["trans_src"]
    if trans_src is None:
        trans_src = torch.full(
            (*sub_costs.shape, 2),
            -1,
            dtype=torch.int32,
            device=sub_costs.device,
        )
    elif not isinstance(trans_src, Tensor):
        raise TypeError("trans_src must be a tensor or None")
    return sub_costs, trans_src


def _scalar_param(
    value: float | Tensor,
    name: str,
    sub_costs: Tensor,
) -> float | Tensor:
    if name == "temp":
        _validate_temperature_param(value)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor")
        if value.device != sub_costs.device:
            raise ValueError(
                f"{name} must be on the same device as sub_costs"
            )
        if use_pt2_ops(value) or functorch_tensor_batched(value):
            return value
        scalar = float(value.detach().item())
    else:
        scalar = float(value)

    return scalar


def _kernel_args(
    call: KernelCall,
) -> tuple[
    float | Tensor,
    float | Tensor,
    float | Tensor,
    float | Tensor,
    Tensor,
]:
    sub_costs, _ = _inputs(call)
    ins_cost, del_cost, trans_cost, temp = (
        _scalar_param(value, name, sub_costs)
        for name, value in zip(_PARAM_NAMES, call.params, strict=True)
    )

    if call.lengths is None:
        batch, length_1, length_2 = sub_costs.shape
        lengths = torch.tensor(
            [[length_1, length_2]] * batch,
            dtype=torch.int32,
            device=sub_costs.device,
        )
    else:
        lengths = call.lengths
    return ins_cost, del_cost, trans_cost, temp, lengths


def _parameter_grad(
    grad: Tensor,
    parameter: float | Tensor,
) -> Tensor:
    if isinstance(parameter, Tensor):
        return grad.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
    return grad


def _normalized_grad_map(call: KernelCall, grad_map: Tensor) -> Tensor:
    sub_costs, _ = _inputs(call)
    if grad_map.shape != sub_costs.shape:
        raise ValueError(
            "grad_map must have the same shape as sub_costs"
        )
    if grad_map.device != sub_costs.device:
        raise ValueError(
            "grad_map must be on the same device as sub_costs"
        )
    return grad_map.to(dtype=sub_costs.dtype).contiguous()


def _operator_forward(call: KernelCall) -> ForwardPass:
    sub_costs, trans_src = _inputs(call)
    ins_cost, del_cost, trans_cost, temp, lengths = _kernel_args(call)
    if any(functorch_tensor_batched(param) for param in call.params):
        tensor_params = tuple(
            param
            if isinstance(param, Tensor)
            else sub_costs.new_tensor(param)
            for param in call.params
        )
        score, marginals = _raw_forward_tensor(
            sub_costs,
            trans_src,
            *tensor_params,
            lengths,
        )
    else:
        score, marginals = _raw_forward(
            sub_costs,
            trans_src,
            ins_cost,
            del_cost,
            trans_cost,
            temp,
            lengths,
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _operator_backward(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    if len(forward_pass.saved_tensors) != 1:
        raise RuntimeError("Damerau forward state must contain marginals")

    sub_costs, trans_src = _inputs(call)
    ins_cost, del_cost, trans_cost, temp, lengths = _kernel_args(call)
    param_grads = _raw_score_param_grads(
        sub_costs,
        trans_src,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )
    normalized_param_grads = tuple(
        _parameter_grad(grad, parameter)
        for grad, parameter in zip(
            param_grads, call.params, strict=True
        )
    )
    marginals = forward_pass.saved_tensors[0].clone()
    return BackwardPass(
        marginals=marginals,
        entropy=-param_grads[3],
        param_grads=normalized_param_grads,
        input_grads=(marginals,),
    )


def _operator_map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    sub_costs, trans_src = _inputs(call)
    ins_cost, del_cost, trans_cost, temp, lengths = _kernel_args(call)
    return _raw_map_scores_backward(
        sub_costs,
        trans_src,
        _normalized_grad_map(call, grad_map),
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )


def _parameter_field(
    call: KernelCall,
    param_name: str,
    ins_cost: float | Tensor,
    del_cost: float | Tensor,
    trans_cost: float | Tensor,
    temp: float | Tensor,
    lengths: Tensor,
) -> Tensor:
    sub_costs, trans_src = _inputs(call)
    arguments = (
        sub_costs,
        trans_src,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )
    if param_name == "ins_cost":
        return _raw_map_ins_cost_sensitivity(*arguments)
    if param_name == "del_cost":
        return _raw_map_del_cost_sensitivity(*arguments)
    if param_name == "trans_cost":
        return _raw_map_trans_cost_sensitivity(*arguments)
    if param_name == "temp":
        return _raw_map_temp_sensitivity(*arguments)
    raise ValueError(f"unknown Damerau parameter {param_name!r}")


def _operator_param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    ins_cost, del_cost, trans_cost, temp, lengths = _kernel_args(call)
    sensitivity = _parameter_field(
        call,
        param_name,
        ins_cost,
        del_cost,
        trans_cost,
        temp,
        lengths,
    )
    if grad_map is None:
        return sensitivity

    contracted = (
        sensitivity * _normalized_grad_map(call, grad_map)
    ).sum()
    parameter = call.params[_PARAM_NAMES.index(param_name)]
    if isinstance(parameter, Tensor):
        return contracted.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).reshape(parameter.shape)
    return contracted.reshape(1)


kernels = OperatorKernels(
    forward=_operator_forward,
    backward=_operator_backward,
    map_backward=_operator_map_backward,
    param_backward=_operator_param_backward,
    config={"trans_src": None},
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
