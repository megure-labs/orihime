# SPDX-License-Identifier: Apache-2.0
"""Frozen-v3 CKY parsing functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function
from .ops.cky import kernels as _kernels
from .ops.cky import value_grad_params as _value_grad_params


_cky_operator = Operator(
    "cky",
    params=("temp",),
    kernels=_kernels,
    tensor_inputs=("merge_scores", "leaf_scores"),
    tensor_shapes=("B,N,N,N", "B,N"),
    length_axes=None,
)


@_document_public_function
def cky(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the score-native merge map with shape ``[B, N, N, N]``.

    Temperature must be finite and positive; finite chart scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _cky_operator(
        merge_scores,
        leaf_scores,
        temperature,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def cky_leaf_map(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return a non-differentiable leaf-score derivative view of CKY value.

    Temperature must be finite and positive; finite chart scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    tensor_inputs, params, _ = _cky_operator._split_call_inputs(
        merge_scores,
        (leaf_scores, temperature),
        {},
        dtype,
        None,
    )
    merge_scores, leaf_scores = tensor_inputs
    (temperature,) = params
    if isinstance(temperature, Tensor):
        temperature_value = float(temperature.detach().item())
    elif isinstance(temperature, (float, int)):
        temperature_value = float(temperature)
    else:
        raise TypeError("temperature must be a float or tensor")
    leaf_map, _ = _value_grad_params(
        merge_scores,
        leaf_scores,
        temperature_value,
    )
    return leaf_map.detach()


@_document_public_function
def cky_value(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the CKY soft value for each batch item.

    Temperature must be finite and positive; finite chart scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _cky_operator.value(
        merge_scores,
        leaf_scores,
        temperature,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def cky_entropy(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the CKY entropy for each batch item.

    Differentiation is supported through ``merge_scores`` only; derivatives
    for ``leaf_scores`` and temperature are not provided.

    Temperature must be finite and positive; finite chart scores must satisfy
    ``abs(value) / temperature <= 80``.
    """

    return _cky_operator.entropy(
        merge_scores,
        leaf_scores,
        temperature,
        mask=mask,
        dtype=dtype,
    )


__all__ = [
    "cky",
    "cky_entropy",
    "cky_leaf_map",
    "cky_value",
]
