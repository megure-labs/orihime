# SPDX-License-Identifier: Apache-2.0
"""Score-native Eisner dependency-parse functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.eisner import kernels as _kernels


_eisner_operator = Operator(
    "eisner",
    params=("temp",),
    kernels=_kernels,
    tensor_inputs=("arc_scores",),
    tensor_shapes=("B,N,N",),
    length_axes=(-1,),
)


@_document_public_function
def eisner(
    arc_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native soft Eisner arc-marginal map.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _eisner_operator(
        arc_scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def eisner_value(
    arc_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native Eisner parse value.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _eisner_operator.value(
        arc_scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def eisner_entropy(
    arc_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the soft Eisner parse distribution.

    Differentiation is supported through ``arc_scores`` only; the temperature
    derivative is not provided.

    Temperature must be finite and positive; finite scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _eisner_operator.entropy(
        arc_scores,
        temperature,
        lengths=lengths,
        mask=mask,
        dtype=dtype,
    )


__all__ = ["eisner", "eisner_value", "eisner_entropy"]
