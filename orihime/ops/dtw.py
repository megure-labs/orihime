# SPDX-License-Identifier: Apache-2.0
"""Internal Dynamic Time Warping kernel adapter."""

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
    costs: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
    bandwidth: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with a scalar temperature.

    Args:
        costs: Cost matrix [B, L1, L2].
        temp: Temperature for softmin.
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        bandwidth: Optional Sakoe-Chiba bandwidth; None disables the limit.

    Returns:
        value: Soft DTW cost [B].
        marginals: Alignment marginals [B, L1, L2].
    """
    return _ops.dtw_forward(costs, temp, lengths, bandwidth)


def _raw_forward_tensor(
    costs: Tensor,
    temp: Tensor,
    lengths: Tensor,
    bandwidth: int,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with a tensor temperature.

    Args:
        costs: Cost matrix [B, L1, L2].
        temp: Temperature tensor [1].
        lengths: [B, 2] tensor of actual sequence lengths.
        bandwidth: Sakoe-Chiba bandwidth; a negative value disables the limit.

    Returns:
        value: Soft DTW cost [B].
        marginals: Alignment marginals [B, L1, L2].
    """
    return _ops.dtw_forward_t(costs, temp, lengths, bandwidth)


def _raw_score_param_grads(
    costs: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
    bandwidth: Optional[int] = None,
) -> Tensor:
    """Gradient of the value with respect to temperature.

    Args:
        costs: Cost matrix [B, L1, L2].
        temp: Temperature.
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        bandwidth: Optional Sakoe-Chiba bandwidth; None disables the limit.

    Returns:
        grad_temp: Gradient of the value w.r.t. temperature [B].
    """
    return _ops.dtw_value_grad_params(costs, temp, lengths, bandwidth)


def _raw_map_backward(
    costs: Tensor,
    grad_marginals: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
    bandwidth: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Full backward through marginals.

    Computes gradients of a loss through the marginals with respect to the
    costs and temperature.
    The user-supplied cotangent must be contiguous FP32 with the exact costs
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        costs: Cost matrix [B, L1, L2].
        grad_marginals: Gradient w.r.t. marginals [B, L1, L2].
        temp: Temperature.
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        bandwidth: Optional Sakoe-Chiba bandwidth; None disables the limit.

    Returns:
        grad_costs: Gradient w.r.t. costs [B, L1, L2].
        grad_temp: Gradient w.r.t. temperature [1].
    """
    _validate_derivative_vector(
        costs,
        grad_marginals,
        name="cotangent",
        primary_name="costs",
    )
    return _ops.dtw_marginals_backward(
        costs, grad_marginals, temp, lengths, bandwidth
    )


def _raw_map_scores_backward(
    costs: Tensor,
    v: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
    bandwidth: Optional[int] = None,
) -> Tensor:
    """Hessian-vector product: H @ v where H = d^2value/dcosts^2.

    This computes the Hessian action without forming the full O(L^4) Hessian.
    The user-supplied tangent must be contiguous FP32 with the exact costs
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        costs: Cost matrix [B, L1, L2].
        v: Vector to multiply with the Hessian [B, L1, L2].
        temp: Temperature.
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        bandwidth: Optional Sakoe-Chiba bandwidth; None disables the limit.

    Returns:
        hvp: Hessian-vector product [B, L1, L2].
    """
    _validate_derivative_vector(
        costs,
        v,
        name="tangent",
        primary_name="costs",
    )
    return _ops.dtw_marginals_hvp(costs, v, temp, lengths, bandwidth)


def _raw_map_temp_sensitivity(
    costs: Tensor,
    temp: float,
    lengths: Optional[Tensor] = None,
    bandwidth: Optional[int] = None,
) -> Tensor:
    """Gradient of marginals with respect to temperature.

    Returns the full [B, L1, L2] tensor of dmarginals/dtemp.

    Args:
        costs: Cost matrix [B, L1, L2].
        temp: Temperature.
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        bandwidth: Optional Sakoe-Chiba bandwidth; None disables the limit.

    Returns:
        jacobian: dmarginals/dtemp [B, L1, L2].
    """
    return _ops.dtw_marginals_grad_temp(costs, temp, lengths, bandwidth)


def _temperature(call: KernelCall) -> float | Tensor:
    (temp,) = call.params
    _validate_temperature_param(temp)
    if isinstance(temp, Tensor):
        if temp.device != call.primary.device:
            raise ValueError(
                "temperature must be on the same device as costs, "
                f"got {temp.device} and {call.primary.device}"
            )
        if use_pt2_ops(temp) or functorch_tensor_batched(temp):
            return temp
        return float(temp.detach().item())

    return float(temp)


def _bandwidth(call: KernelCall) -> int:
    bandwidth = call.config["bandwidth"]
    return bandwidth if bandwidth is not None and bandwidth > 0 else -1


def _lengths(call: KernelCall) -> Tensor:
    if call.lengths is not None:
        return call.lengths
    batch, length_1, length_2 = call.primary.shape
    return call.primary.new_tensor(
        [[length_1, length_2]] * batch,
        dtype=torch.int32,
    )


def _forward_kernel(call: KernelCall) -> ForwardPass:
    temp = _temperature(call)
    if functorch_tensor_batched(call.params[0]):
        score, marginals = _raw_forward_tensor(
            call.primary,
            call.params[0],
            _lengths(call),
            _bandwidth(call),
        )
    else:
        score, marginals = _raw_forward(
            call.primary,
            temp,
            call.lengths,
            _bandwidth(call),
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward_kernel(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    if len(forward_pass.saved_tensors) != 1:
        raise RuntimeError("DTW forward state must contain marginals")
    grad_temp = _ops.dtw_value_grad_params(
        call.primary,
        _temperature(call),
        call.lengths,
        _bandwidth(call),
    )
    return BackwardPass(
        marginals=forward_pass.saved_tensors[0].clone(),
        entropy=-grad_temp,
        param_grads=(grad_temp,),
    )


def _normalized_grad_map(call: KernelCall, grad_map: Tensor) -> Tensor:
    if grad_map.shape != call.primary.shape:
        raise ValueError(
            "grad_map must have the same shape as costs, "
            f"got {tuple(grad_map.shape)} and {tuple(call.primary.shape)}"
        )
    if grad_map.device != call.primary.device:
        raise ValueError(
            "grad_map must be on the same device as costs, "
            f"got {grad_map.device} and {call.primary.device}"
        )
    return grad_map.to(dtype=call.primary.dtype).contiguous()


def _map_backward_kernel(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    return _ops.dtw_marginals_hvp(
        call.primary,
        _normalized_grad_map(call, grad_map),
        _temperature(call),
        call.lengths,
        _bandwidth(call),
    )


def _param_backward_kernel(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    if param != "temp":
        raise ValueError(f"unsupported DTW parameter {param!r}")

    sensitivity = _ops.dtw_marginals_grad_temp(
        call.primary,
        _temperature(call),
        call.lengths,
        _bandwidth(call),
    )
    if grad_map is None:
        return sensitivity

    contracted = (
        sensitivity * _normalized_grad_map(call, grad_map)
    ).sum()
    (temp,) = call.params
    if isinstance(temp, Tensor):
        return contracted.reshape(temp.shape)
    return contracted.reshape(1)


kernels = OperatorKernels(
    forward=_forward_kernel,
    backward=_backward_kernel,
    map_backward=_map_backward_kernel,
    param_backward=_param_backward_kernel,
    config={"bandwidth": 0},
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
