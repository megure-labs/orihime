# SPDX-License-Identifier: Apache-2.0
"""Low-level canonical Saigo-Vert local alignment with a linear gap cost.

This is a distinct three-state path-space operator, not ordinary
Smith-Waterman: it permits exactly one ``I -> D`` cross, forbids ``D -> I``,
terminates only at match states, and includes one explicit empty alignment.
Every consumed gap symbol contributes the same scalar ``gap`` penalty.
"""

from typing import Optional, Tuple

from torch import Tensor

from .. import _ops


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


__all__ = [
    "forward",
    "forward_t",
    "value_grad_params",
    "marginals_backward",
    "marginals_hvp",
    "marginals_grad_gap",
    "marginals_grad_temp",
]
