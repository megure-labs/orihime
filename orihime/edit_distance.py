# SPDX-License-Identifier: Apache-2.0
"""Edit-distance operator instances."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.lcs import LCS_KERNELS as _lcs_kernels
from .ops.lev import KERNELS as _lev_kernels
from .ops.osa import KERNELS as _osa_kernels


_lev_operator = Operator(
    "lev",
    params=("ins", "del", "temp"),
    kernels=_lev_kernels,
    tensor_inputs=("substitution_costs",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)


@_document_public_function
def lev(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the cost-native soft Levenshtein substitution map.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    return _lev_operator(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def lev_value(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the cost-native soft Levenshtein value.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    return _lev_operator.value(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def lev_entropy(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft Levenshtein path entropy.

    Differentiation is supported through ``substitution_costs`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    return _lev_operator.entropy(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


_lcs_operator = Operator(
    "lcs",
    params=("temp",),
    kernels=_lcs_kernels,
    tensor_inputs=("match_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)
_osa_operator = Operator(
    "osa",
    params=("ins_cost", "del_cost", "trans_cost", "temp"),
    kernels=_osa_kernels,
    tensor_inputs=("substitution_costs",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)
__all__ = [
    "lev",
    "lev_value",
    "lev_entropy",
]
