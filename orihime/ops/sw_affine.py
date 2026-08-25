# SPDX-License-Identifier: Apache-2.0
"""Internal affine Smith-Waterman kernel adapter."""

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


_PARAM_NAMES = ("gap_open", "gap_ext", "temp")


def _validate_temperature(temp: float | Tensor) -> None:
    _validate_temperature_param(temp)


def _validate_gap(gap: float | Tensor, name: str) -> None:
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"{name}={gap} is positive. Gap penalties are typically negative. "
            "If this is intentional, pass as tensor to suppress this warning.",
            UserWarning,
            stacklevel=4,
        )


def _normalize_param(param: float | Tensor, scores: Tensor) -> float | Tensor:
    if not isinstance(param, Tensor):
        return float(param)
    if param.dim() == 0:
        param = param.view(1)
    if param.device != scores.device or param.dtype != scores.dtype:
        param = param.to(device=scores.device, dtype=scores.dtype)
    return param


def _validate_prefix_mask(mask: Tensor, name: str) -> Tensor:
    mask_int = mask.to(torch.int32)
    has_hole = (mask_int[:, 1:] - mask_int[:, :-1] > 0).any(dim=1)
    if has_hole.any():
        bad_indices = has_hole.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"{name} is not prefix-only (has holes) at batch indices: "
            f"{bad_indices}. Masks must have all True values at the beginning, "
            "followed by all False values."
        )
    return mask.sum(dim=1).to(torch.int32)


def _resolve_lengths(call: KernelCall) -> Tensor:
    batch, length1, length2 = call.primary.shape
    mask1 = call.config["mask1"]
    mask2 = call.config["mask2"]
    if call.lengths is not None and (mask1 is not None or mask2 is not None):
        raise ValueError("Cannot specify both 'lengths' and 'mask1'/'mask2'")
    if mask1 is None and mask2 is None:
        if call.lengths is not None:
            return call.lengths
        return torch.tensor(
            [[length1, length2]] * batch,
            dtype=torch.int32,
            device=call.primary.device,
        )

    lengths = torch.empty(
        (batch, 2), dtype=torch.int32, device=call.primary.device
    )
    lengths[:, 0] = (
        _validate_prefix_mask(mask1, "mask1") if mask1 is not None else length1
    )
    lengths[:, 1] = (
        _validate_prefix_mask(mask2, "mask2") if mask2 is not None else length2
    )
    return lengths


def _prepare(call: KernelCall) -> tuple[tuple[float | Tensor, ...], Tensor]:
    gap_open, gap_ext, temp = call.params
    _validate_temperature(temp)
    _validate_gap(gap_open, "gap_open")
    _validate_gap(gap_ext, "gap_ext")
    params = tuple(_normalize_param(param, call.primary) for param in call.params)
    return params, _resolve_lengths(call)


def _scalar_params(
    params: tuple[float | Tensor, ...],
) -> tuple[float | Tensor, ...]:
    if use_pt2_ops(*params) or any(
        functorch_tensor_batched(param) for param in params
    ):
        return params
    return tuple(
        float(param.detach().item()) if isinstance(param, Tensor) else param
        for param in params
    )


def _forward_pass(call: KernelCall) -> ForwardPass:
    params, lengths = _prepare(call)
    if any(isinstance(param, Tensor) for param in call.params):
        tensor_params = tuple(
            param
            if isinstance(param, Tensor)
            else torch.tensor(
                [param], dtype=call.primary.dtype, device=call.primary.device
            )
            for param in params
        )
        score, marginals = _ops.sw_affine_forward_t(
            call.primary, *tensor_params, lengths
        )
    else:
        score, marginals = _ops.sw_affine_forward(
            call.primary, *_scalar_params(params), lengths
        )
    return ForwardPass(value=score, saved_tensors=(marginals,))


def _backward_pass(call: KernelCall, forward: ForwardPass) -> BackwardPass:
    params, lengths = _prepare(call)
    param_grads = tuple(
        _ops.sw_affine_value_grad_params(
            call.primary, *_scalar_params(params), lengths
        )
    )
    return BackwardPass(
        # Keep the public map output distinct from the copy carried as opaque
        # forward state. _MapFn marks saved state non-differentiable.
        marginals=forward.saved_tensors[0].clone(),
        entropy=param_grads[2],
        param_grads=param_grads,
    )


def _map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    del state
    params, lengths = _prepare(call)
    return _ops.sw_affine_marginals_hvp(
        call.primary,
        grad_map.contiguous(),
        *_scalar_params(params),
        lengths,
    )


def _sensitivity(
    call: KernelCall,
    param_name: str,
    params: tuple[float | Tensor, ...],
    lengths: Tensor,
) -> Tensor:
    kernels = {
        "gap_open": _ops.sw_affine_marginals_grad_gap_open,
        "gap_ext": _ops.sw_affine_marginals_grad_gap_ext,
        "temp": _ops.sw_affine_marginals_grad_temp,
    }
    return kernels[param_name](
        call.primary, *_scalar_params(params), lengths
    )


def _param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    del state
    params, lengths = _prepare(call)
    derivative = _sensitivity(call, param_name, params, lengths)
    if grad_map is None:
        return derivative

    result = (grad_map * derivative).sum()
    param = call.params[_PARAM_NAMES.index(param_name)]
    if isinstance(param, Tensor):
        return result.reshape(param.shape).to(
            device=param.device, dtype=param.dtype
        )
    return result.reshape(1)


SW_AFFINE_KERNELS = OperatorKernels(
    forward=_forward_pass,
    backward=_backward_pass,
    map_backward=_map_backward,
    param_backward=_param_backward,
    config={"mask1": None, "mask2": None},
)


def _raw_forward(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with scalar parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty (typically negative)
        gap_ext: Gap extension penalty (typically negative)
        temp: Temperature for soft-max
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.sw_affine_forward(scores, gap_open, gap_ext, temp, lengths)


def _raw_forward_tensor(
    scores: Tensor,
    gap_open: Tensor,
    gap_ext: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with tensor parameters (for learnable params).

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty tensor [1]
        gap_ext: Gap extension penalty tensor [1]
        temp: Temperature tensor [1]
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.sw_affine_forward_t(scores, gap_open, gap_ext, temp, lengths)


def _raw_score_param_grads(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Gradients of value w.r.t. parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_gap_open: Gradient of value w.r.t. gap_open [B]
        grad_gap_ext: Gradient of value w.r.t. gap_ext [B]
        grad_temp: Gradient of value w.r.t. temperature [B]
    """
    return _ops.sw_affine_value_grad_params(scores, gap_open, gap_ext, temp, lengths)


def _raw_map_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Full backward through marginals.

    Computes gradients of loss (through marginals) w.r.t. all inputs.
    The user-supplied cotangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Similarity matrix [B, L1, L2]
        grad_marginals: Gradient w.r.t. marginals [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_scores: Gradient w.r.t. scores [B, L1, L2]
        grad_gap_open: Gradient w.r.t. gap_open [1]
        grad_gap_ext: Gradient w.r.t. gap_ext [1]
        grad_temp: Gradient w.r.t. temperature [1]
    """
    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.sw_affine_marginals_backward(
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
    """Hessian-vector product: H @ v where H = d^2value/dscores^2.

    This efficiently computes the action of the Hessian on a vector
    without forming the full O(L^4) Hessian matrix.
    The user-supplied tangent must be contiguous FP32 with the exact scores
    shape and device; invalid vectors are rejected before native dispatch.

    Args:
        scores: Similarity matrix [B, L1, L2]
        v: Vector to multiply with Hessian [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
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
    return _ops.sw_affine_marginals_hvp(scores, v, gap_open, gap_ext, temp, lengths)


def _raw_map_gap_open_sensitivity(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. gap_open (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dgap_open.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dgap_open [B, L1, L2]
    """
    return _ops.sw_affine_marginals_grad_gap_open(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_map_gap_ext_sensitivity(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. gap_ext (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dgap_ext.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dgap_ext [B, L1, L2]
    """
    return _ops.sw_affine_marginals_grad_gap_ext(
        scores, gap_open, gap_ext, temp, lengths
    )


def _raw_map_temp_sensitivity(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. temperature (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dtemp.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dtemp [B, L1, L2]
    """
    return _ops.sw_affine_marginals_grad_temp(
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
