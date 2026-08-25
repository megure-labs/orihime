# SPDX-License-Identifier: Apache-2.0
"""Low-level canonical Saigo-Vert local-alignment operations.

Unlike :mod:`orihime.ops.sw_affine`, this operator enumerates every monotone
alignment exactly once and terminates only at match states (plus the empty
alignment). The gap cost is ``gap_open + (length - 1) * gap_ext``.
"""

from typing import Optional, Tuple

from torch import Tensor

from .. import _ops


def forward(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return the log-partition value and score marginals."""
    return _ops.sv_affine_forward(scores, gap_open, gap_ext, temp, lengths)


def forward_t(
    scores: Tensor,
    gap_open: Tensor,
    gap_ext: Tensor,
    temp: Tensor,
    lengths: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Tensor-parameter form of :func:`forward`."""
    return _ops.sv_affine_forward_t(scores, gap_open, gap_ext, temp, lengths)


def value_grad_params(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return per-batch value gradients for gap-open, gap-extension, and T."""
    return _ops.sv_affine_value_grad_params(scores, gap_open, gap_ext, temp, lengths)


def marginals_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Backpropagate through the marginal matrix."""
    return _ops.sv_affine_marginals_backward(
        scores, grad_marginals, gap_open, gap_ext, temp, lengths
    )


def marginals_hvp(
    scores: Tensor,
    v: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return ``d²value/dscores² @ v``."""
    return _ops.sv_affine_marginals_hvp(scores, v, gap_open, gap_ext, temp, lengths)


def marginals_grad_gap_open(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginal derivative with respect to ``gap_open``."""
    return _ops.sv_affine_marginals_grad_gap_open(
        scores, gap_open, gap_ext, temp, lengths
    )


def marginals_grad_gap_ext(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginal derivative with respect to ``gap_ext``."""
    return _ops.sv_affine_marginals_grad_gap_ext(
        scores, gap_open, gap_ext, temp, lengths
    )


def marginals_grad_temp(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Return the marginal derivative with respect to temperature."""
    return _ops.sv_affine_marginals_grad_temp(
        scores, gap_open, gap_ext, temp, lengths
    )


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
