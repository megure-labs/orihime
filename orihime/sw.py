# SPDX-License-Identifier: Apache-2.0
"""Smith-Waterman public functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.sw import kernels as _sw_kernels
from .ops.sw_affine import SW_AFFINE_KERNELS as _sw_affine_kernels


_sw_operator = Operator(
    "sw",
    params=("gap", "temp"),
    kernels=_sw_kernels,
    tensor_inputs=("pair_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)
_sw_affine_operator = Operator(
    "sw_affine",
    params=("gap_open", "gap_ext", "temp"),
    kernels=_sw_affine_kernels,
    tensor_inputs=("pair_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)


@_document_public_function
def sw(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft local-alignment map for score-native pair scores.

    Higher pair scores favor a match. Smith-Waterman local alignment may
    start and end within the two input sequences.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sw_operator(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sw_value(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft Smith-Waterman value.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sw_operator.value(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sw_entropy(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the soft Smith-Waterman path distribution.

    Differentiation is supported through ``pair_scores`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sw_operator.entropy(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sw_affine(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native affine Smith-Waterman alignment map.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sw_affine_operator(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sw_affine_value(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native affine Smith-Waterman value.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sw_affine_operator.value(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sw_affine_entropy(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native affine Smith-Waterman entropy.

    Differentiation is supported through ``pair_scores`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sw_affine_operator.entropy(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


__all__ = [
    "sw",
    "sw_entropy",
    "sw_value",
    "sw_affine",
    "sw_affine_entropy",
    "sw_affine_value",
]
