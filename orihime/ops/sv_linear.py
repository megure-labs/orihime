# SPDX-License-Identifier: Apache-2.0
"""Canonical Saigo–Vert local alignment with a linear gap cost."""

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
    _validate_temperature,
)


_PARAM_NAMES = ("gap", "temp")


def _validate_gap(gap: float | Tensor) -> None:
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"gap={gap} is positive. Gap penalties are typically negative. "
            "If this is intentional, pass it as a tensor to suppress this "
            "warning.",
            UserWarning,
            stacklevel=4,
        )


def _default_lengths(scores: Tensor) -> Tensor:
    batch, length_1, length_2 = scores.shape
    return torch.tensor(
        [[length_1, length_2]] * batch,
        dtype=torch.int32,
        device=scores.device,
    )


def _normalize_param(param: float | Tensor, scores: Tensor) -> float | Tensor:
    if not isinstance(param, Tensor):
        return float(param)
    if param.dim() == 0:
        param = param.view(1)
    if param.device != scores.device or param.dtype != scores.dtype:
        param = param.to(device=scores.device, dtype=scores.dtype)
    return param


def _prepare(call: KernelCall) -> tuple[tuple[float | Tensor, ...], Tensor]:
    gap, temp = call.params
    _validate_gap(gap)
    _validate_temperature(temp)
    params = tuple(_normalize_param(param, call.primary) for param in call.params)
    lengths = (
        call.lengths
        if call.lengths is not None
        else _default_lengths(call.primary)
    )
    return params, lengths


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


def forward(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return the log-partition value and score marginals."""
    return _ops.sv_linear_forward(scores, gap, temp, lengths)


def forward_t(
    scores: Tensor,
    gap: Tensor,
    temp: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Tensor-parameter form of :func:`forward`."""
    return _ops.sv_linear_forward_t(scores, gap, temp, lengths)


def value_grad_params(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return per-batch value gradients for ``gap`` and temperature."""
    return _ops.sv_linear_value_grad_params(scores, gap, temp, lengths)


def marginals_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backpropagate through the marginal matrix."""
    _validate_derivative_vector(
        scores,
        grad_marginals,
        name="cotangent",
        primary_name="scores",
    )
    return _ops.sv_linear_marginals_backward(
        scores, grad_marginals, gap, temp, lengths
    )


def marginals_hvp(
    scores: Tensor,
    v: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return ``d²value/dscores² @ v``."""
    _validate_derivative_vector(
        scores,
        v,
        name="tangent",
        primary_name="scores",
    )
    return _ops.sv_linear_marginals_hvp(scores, v, gap, temp, lengths)


def marginals_grad_gap(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginal derivative with respect to ``gap``."""
    return _ops.sv_linear_marginals_grad_gap(scores, gap, temp, lengths)


def marginals_grad_temp(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginal derivative with respect to temperature."""
    return _ops.sv_linear_marginals_grad_temp(scores, gap, temp, lengths)


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
        value, marginals = forward_t(call.primary, *tensor_params, lengths)
    else:
        value, marginals = forward(
            call.primary, *_scalar_params(params), lengths
        )
    return ForwardPass(
        value=value,
        saved_tensors=(marginals, lengths.clone()),
    )


def _backward_pass(
    call: KernelCall,
    forward_pass: ForwardPass,
) -> BackwardPass:
    params, _ = _prepare(call)
    lengths = forward_pass.saved_tensors[-1]
    grad_gap, grad_temp = value_grad_params(
        call.primary, *_scalar_params(params), lengths
    )
    return BackwardPass(
        marginals=forward_pass.saved_tensors[0].clone(),
        entropy=grad_temp,
        param_grads=(grad_gap, grad_temp),
        saved_tensors=(lengths,),
    )


def _map_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> Tensor:
    params, _ = _prepare(call)
    return marginals_hvp(
        call.primary,
        grad_map.contiguous(),
        *_scalar_params(params),
        state.saved_tensors[-1],
    )


def _param_backward(
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    param_name: str,
    grad_map: Tensor | None,
) -> Tensor:
    params, _ = _prepare(call)
    lengths = state.saved_tensors[-1]
    sensitivities = {
        "gap": marginals_grad_gap,
        "temp": marginals_grad_temp,
    }
    if param_name not in sensitivities:
        raise ValueError(f"unknown sv_linear parameter {param_name!r}")
    field = sensitivities[param_name](
        call.primary, *_scalar_params(params), lengths
    )
    if grad_map is None:
        return field
    result = (grad_map * field).sum()
    param = call.params[_PARAM_NAMES.index(param_name)]
    if isinstance(param, Tensor):
        return result.reshape(param.shape).to(
            device=param.device, dtype=param.dtype
        )
    return result.reshape(1)


SV_LINEAR_KERNELS = OperatorKernels(
    forward=_forward_pass,
    backward=_backward_pass,
    map_backward=_map_backward,
    param_backward=_param_backward,
)


__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_gap",
    "marginals_grad_temp",
]
