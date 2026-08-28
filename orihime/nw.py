# SPDX-License-Identifier: Apache-2.0
"""Score-native Needleman-Wunsch global-alignment functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.nw import kernels as _nw_kernels
from .ops.nw_affine import KERNELS as _nw_affine_kernels


_nw_operator = Operator(
    "nw",
    params=("gap", "temp"),
    kernels=_nw_kernels,
    tensor_inputs=("pair_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)
_nw_affine_operator = Operator(
    "nw_affine",
    params=("gap_open", "gap_ext", "temp"),
    kernels=_nw_affine_kernels,
    tensor_inputs=("pair_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)


@_document_public_function
def nw(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native soft Needleman-Wunsch global-alignment map.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _nw_operator(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def nw_value(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native soft Needleman-Wunsch global-alignment value.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _nw_operator.value(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def nw_entropy(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native soft Needleman-Wunsch global-alignment entropy.

    Differentiation is supported through ``pair_scores`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _nw_operator.entropy(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def nw_affine(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native affine Needleman-Wunsch map.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _nw_affine_operator(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def nw_affine_value(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native affine Needleman-Wunsch value.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _nw_affine_operator.value(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def nw_affine_entropy(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native affine Needleman-Wunsch entropy.

    Differentiation is supported through ``pair_scores`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _nw_affine_operator.entropy(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


__all__ = [
    "nw",
    "nw_entropy",
    "nw_value",
    "nw_affine",
    "nw_affine_entropy",
    "nw_affine_value",
]
