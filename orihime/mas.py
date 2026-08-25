# SPDX-License-Identifier: Apache-2.0
"""Score-native Monotonic Alignment Search functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.mas import kernels as _kernels


_mas_operator = Operator(
    "mas",
    params=("temp",),
    kernels=_kernels,
    tensor_inputs=("scores",),
    tensor_shapes=("B,T,S",),
    length_axes=(-2, -1),
)


@_document_public_function
def mas(
    scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native soft monotonic alignment map.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _mas_operator(
        scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def mas_value(
    scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native monotonic alignment value.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _mas_operator.value(
        scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def mas_entropy(
    scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the soft monotonic alignment distribution.

    Differentiation is supported through ``scores`` only; the temperature
    derivative is not provided.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _mas_operator.entropy(
        scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


__all__ = ["mas", "mas_value", "mas_entropy"]
