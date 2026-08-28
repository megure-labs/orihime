# SPDX-License-Identifier: Apache-2.0
"""Canonical Saigo–Vert local-alignment functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.sv_affine import SV_AFFINE_KERNELS
from .ops.sv_linear import SV_LINEAR_KERNELS


_sv_linear_operator = Operator(
    "sv_linear",
    params=("gap", "temp"),
    kernels=SV_LINEAR_KERNELS,
    tensor_inputs=("pair_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)
_sv_affine_operator = Operator(
    "sv_affine",
    params=("gap_open", "gap_ext", "temp"),
    kernels=SV_AFFINE_KERNELS,
    tensor_inputs=("pair_scores",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)


@_document_public_function
def sv(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the canonical score-native Saigo–Vert linear-gap alignment map.

    The recurrence counts each monotone matched-pair skeleton once, permits
    the canonical insertion-to-deletion transition, forbids the reverse
    transition, and includes one explicit empty alignment.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sv_linear_operator(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sv_value(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the canonical Saigo–Vert linear-gap value.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sv_linear_operator.value(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sv_entropy(
    pair_scores: Tensor,
    *,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the Saigo–Vert linear-gap path distribution.

    Differentiation is supported through ``pair_scores`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sv_linear_operator.entropy(
        pair_scores,
        gap_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


# The native adapter retains ``sv_linear`` to distinguish its kernel family
# from ``sv_affine``. These Python aliases keep old internal regression
# fixtures working without exporting that implementation spelling publicly.
sv_linear = sv
sv_linear_value = sv_value
sv_linear_entropy = sv_entropy


@_document_public_function
def sv_affine(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the canonical score-native Saigo–Vert affine-gap alignment map.

    A gap of length ``k`` receives
    ``gap_open_score + (k - 1) * gap_extend_score``. The recurrence counts
    each monotone matched-pair skeleton exactly once and includes one explicit
    empty alignment.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sv_affine_operator(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sv_affine_value(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the canonical Saigo–Vert affine-gap value.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sv_affine_operator.value(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def sv_affine_entropy(
    pair_scores: Tensor,
    *,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the Saigo–Vert affine-gap path distribution.

    Differentiation is supported through ``pair_scores`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite scores and scoring
    parameters must satisfy ``abs(value) / temperature <= 80``.
    """

    return _sv_affine_operator.entropy(
        pair_scores,
        gap_open_score,
        gap_extend_score,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


__all__ = [
    "sv",
    "sv_value",
    "sv_entropy",
    "sv_affine",
    "sv_affine_value",
    "sv_affine_entropy",
]
