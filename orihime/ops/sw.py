# SPDX-License-Identifier: Apache-2.0
"""Internal Smith-Waterman kernel adapter."""

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
    _validate_derivative_vector,
    _validate_temperature as _validate_temperature_param,
)


def _validate_temperature(temp: float | Tensor) -> None:
    _validate_temperature_param(temp)


def _validate_gap(gap: float | Tensor) -> None:
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"gap={gap} is positive. Gap penalties are typically negative. "
            "If this is intentional, pass as tensor to suppress this warning.",
            UserWarning,
            stacklevel=4,
        )


def _normalize_param_tensor(param: Tensor, scores: Tensor) -> Tensor:
    if param.dim() == 0:
        param = param.view(1)
    if param.device != scores.device or param.dtype != scores.dtype:
        param = param.to(device=scores.device, dtype=scores.dtype)
    return param


def _make_lengths(scores: Tensor) -> Tensor:
    batch, length1, length2 = scores.shape
    return torch.tensor(
        [[length1, length2]] * batch,
        dtype=torch.int32,
        device=scores.device,
    )


def _validate_prefix_mask(mask: Tensor, name: str) -> Tensor:
    lengths = mask.sum(dim=1)
    mask_int = mask.to(torch.int32)
    differences = mask_int[:, 1:] - mask_int[:, :-1]
    has_hole = (differences > 0).any(dim=1)
    if has_hole.any():
        bad_indices = has_hole.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"{name} is not prefix-only (has holes) at batch indices: "
            f"{bad_indices}. Masks must have all True values at the "
            "beginning, followed by all False values."
        )
    return lengths.to(torch.int32)


def _masks_to_lengths(
    mask1: Tensor | None,
    mask2: Tensor | None,
    scores: Tensor,
) -> Tensor:
    batch, length1, length2 = scores.shape
    lengths = torch.empty(
        (batch, 2), dtype=torch.int32, device=scores.device
    )
    if mask1 is None:
        lengths[:, 0] = length1
    else:
        lengths[:, 0] = _validate_prefix_mask(mask1, "mask1")
    if mask2 is None:
        lengths[:, 1] = length2
    else:
        lengths[:, 1] = _validate_prefix_mask(mask2, "mask2")
    return lengths


def _resolve_lengths(call: KernelCall) -> Tensor:
    mask1 = call.config["mask1"]
    mask2 = call.config["mask2"]
    if call.lengths is not None and (mask1 is not None or mask2 is not None):
        raise ValueError("Cannot specify both 'lengths' and 'mask1'/'mask2'")
    if mask1 is not None or mask2 is not None:
        return _masks_to_lengths(mask1, mask2, call.primary)
    if call.lengths is None:
        return _make_lengths(call.primary)
    return call.lengths


def _normalized_params(call: KernelCall) -> tuple[Tensor, Tensor]:
    gap, temp = call.params
    if isinstance(gap, Tensor):
        gap_tensor = _normalize_param_tensor(gap, call.primary)
    else:
        gap_tensor = torch.tensor(
            [gap], device=call.primary.device, dtype=call.primary.dtype
        )
    if isinstance(temp, Tensor):
        temp_tensor = _normalize_param_tensor(temp, call.primary)
    else:
        temp_tensor = torch.tensor(
            [temp], device=call.primary.device, dtype=call.primary.dtype
        )
    return gap_tensor, temp_tensor


def _scalar_params(
    call: KernelCall,
) -> tuple[float | Tensor, float | Tensor]:
    gap, temp = _normalized_params(call)
    if (
        use_pt2_ops(gap, temp)
        or functorch_tensor_batched(gap)
        or functorch_tensor_batched(temp)
    ):
        return gap, temp
    return float(gap.detach().item()), float(temp.detach().item())


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
    return _ops.sw_forward(scores, gap, temp, lengths)


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
    return _ops.sw_forward_t(scores, gap, temp, lengths)


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
    return _ops.sw_value_grad_params(scores, gap, temp, lengths)


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
    return _ops.sw_marginals_backward(scores, grad_marginals, gap, temp, lengths)


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
    return _ops.sw_marginals_hvp(scores, v, gap, temp, lengths)


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
    return _ops.sw_marginals_grad_gap(scores, gap, temp, lengths)


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
    return _ops.sw_marginals_grad_temp(scores, gap, temp, lengths)


def _forward_kernel(call: KernelCall) -> ForwardPass:
    gap, temp = call.params
    _validate_gap(gap)
    _validate_temperature(temp)
    lengths = _resolve_lengths(call)

    if isinstance(gap, Tensor) or isinstance(temp, Tensor):
        gap_tensor, temp_tensor = _normalized_params(call)
        score, marginals = _raw_forward_tensor(
            call.primary, gap_tensor, temp_tensor, lengths
        )
    else:
        score, marginals = _raw_forward(
            call.primary, float(gap), float(temp), lengths
        )
    return ForwardPass(
        value=score,
        saved_tensors=(marginals, lengths.clone()),
    )


def _backward_kernel(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    gap, temp = _scalar_params(call)
    lengths = forward_pass.saved_tensors[-1]
    grad_gap, grad_temp = _raw_score_param_grads(
        call.primary, gap, temp, lengths
    )
    return BackwardPass(
        # _MapFn marks saved kernel state non-differentiable. Keep the public
        # map as a distinct tensor rather than aliasing that saved state.
        marginals=forward_pass.saved_tensors[0].clone(),
        entropy=grad_temp,
        param_grads=(grad_gap, grad_temp),
        saved_tensors=(lengths,),
    )


def _map_backward_kernel(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    gap, temp = _scalar_params(call)
    return _raw_map_scores_backward(
        call.primary,
        grad_map.contiguous(),
        gap,
        temp,
        state.saved_tensors[-1],
    )


def _contract_param_field(
    field: Tensor,
    grad_map: Tensor,
    param: float | Tensor,
) -> Tensor:
    contracted = (field * grad_map).sum()
    if isinstance(param, Tensor):
        return contracted.to(device=param.device, dtype=param.dtype).reshape(
            param.shape
        )
    return contracted.reshape(1)


def _param_backward_kernel(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    name: str,
    grad_map: Tensor | None,
) -> Tensor:
    gap, temp = _scalar_params(call)
    lengths = state.saved_tensors[-1]
    if name == "gap":
        field = _raw_map_gap_sensitivity(
            call.primary, gap, temp, lengths
        )
        param = call.params[0]
    elif name == "temp":
        field = _raw_map_temp_sensitivity(
            call.primary, gap, temp, lengths
        )
        param = call.params[1]
    else:
        raise ValueError(f"unknown sw parameter {name!r}")

    if grad_map is None:
        return field
    return _contract_param_field(field, grad_map, param)


kernels = OperatorKernels(
    forward=_forward_kernel,
    backward=_backward_kernel,
    map_backward=_map_backward_kernel,
    param_backward=_param_backward_kernel,
    config={"mask1": None, "mask2": None},
)


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
