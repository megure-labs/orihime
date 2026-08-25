# SPDX-License-Identifier: Apache-2.0
"""Frozen v3 Longest Common Subsequence surface."""

from __future__ import annotations

import torch
from torch import Tensor

from .edit_distance import _lcs_operator
from .operator import _document_public_function


@_document_public_function
def lcs(
    match_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native soft Longest Common Subsequence map.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _lcs_operator(
        match_scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def lcs_value(
    match_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft Longest Common Subsequence value.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _lcs_operator.value(
        match_scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def lcs_entropy(
    match_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft Longest Common Subsequence entropy.

    Differentiation is supported through ``match_scores`` only; the
    temperature derivative is not provided.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _lcs_operator.entropy(
        match_scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


__all__ = ["lcs", "lcs_entropy", "lcs_value"]
