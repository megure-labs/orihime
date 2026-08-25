# SPDX-License-Identifier: Apache-2.0
"""Frozen v3 Optimal String Alignment surface."""

from __future__ import annotations

import torch
from torch import Tensor

from ._pt2_utils import use_pt2_ops
from .edit_distance import _osa_operator
from .operator import _document_public_function


def _raise_if_true(condition: Tensor, message: str) -> None:
    functorch = torch._C._functorch
    if functorch.is_batchedtensor(condition):
        physical_condition = condition
        while functorch.is_batchedtensor(physical_condition):
            physical_condition = functorch.get_unwrapped(physical_condition)
        torch._assert_async(~physical_condition.any(), message)
    elif use_pt2_ops(condition):
        torch._assert_async(~condition, message)
    elif bool(condition.item()):
        raise ValueError(message)


def _kernel_allowed_transpositions(
    substitution_costs: Tensor,
    lengths: Tensor | None,
    allowed_transpositions: Tensor | None,
) -> Tensor:
    if not isinstance(substitution_costs, Tensor):
        raise TypeError("substitution_costs must be a tensor")
    if substitution_costs.ndim != 3:
        raise ValueError(
            "substitution_costs must have shape [B, L1, L2], got "
            f"{tuple(substitution_costs.shape)}"
        )
    lengths = _osa_operator._validate_lengths_contract(
        substitution_costs,
        lengths,
    )

    if allowed_transpositions is None:
        return torch.zeros_like(
            substitution_costs,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
    if not isinstance(allowed_transpositions, Tensor):
        raise TypeError("allowed_transpositions must be a tensor or None")
    if allowed_transpositions.dtype != torch.bool:
        raise TypeError(
            "allowed_transpositions must have dtype torch.bool, got "
            f"{allowed_transpositions.dtype}"
        )
    if allowed_transpositions.shape != substitution_costs.shape:
        raise ValueError(
            "allowed_transpositions must have the same shape as "
            "substitution_costs, got "
            f"{tuple(allowed_transpositions.shape)} vs "
            f"{tuple(substitution_costs.shape)}"
        )
    if allowed_transpositions.device != substitution_costs.device:
        raise ValueError(
            "allowed_transpositions must be on the same device as "
            "substitution_costs, got "
            f"{allowed_transpositions.device} vs {substitution_costs.device}"
        )

    boundary_true = (
        allowed_transpositions[:, :1, :].any()
        | allowed_transpositions[:, :, :1].any()
    )
    _raise_if_true(
        boundary_true,
        "allowed_transpositions must be false in its first row and first column",
    )

    if lengths is not None:
        length_1, length_2 = substitution_costs.shape[-2:]
        rows = torch.arange(
            length_1,
            device=substitution_costs.device,
        ).view(1, length_1, 1)
        columns = torch.arange(
            length_2,
            device=substitution_costs.device,
        ).view(1, 1, length_2)
        padded_true = (
            allowed_transpositions
            & (
                (rows >= lengths[:, 0].view(-1, 1, 1))
                | (columns >= lengths[:, 1].view(-1, 1, 1))
            )
        ).any()
        _raise_if_true(
            padded_true,
            "allowed_transpositions must be false at padded positions",
        )

    return allowed_transpositions.to(dtype=torch.float32).contiguous()


@_document_public_function
def osa(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    allowed_transpositions: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the cost-native soft Optimal String Alignment map.

    The shipped kernels accept exactly one batch dimension, so
    ``substitution_costs`` has shape ``[B, L1, L2]``. A boolean
    ``allowed_transpositions`` enables valid adjacent-transposition edges;
    ``None`` disables all transpositions.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    kernel_mask = _kernel_allowed_transpositions(
        substitution_costs,
        lengths,
        allowed_transpositions,
    )
    return _osa_operator(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        lengths=lengths,
        trans_mask=kernel_mask,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def osa_value(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    allowed_transpositions: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft Optimal String Alignment value.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    kernel_mask = _kernel_allowed_transpositions(
        substitution_costs,
        lengths,
        allowed_transpositions,
    )
    return _osa_operator.value(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        lengths=lengths,
        trans_mask=kernel_mask,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def osa_entropy(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    allowed_transpositions: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft Optimal String Alignment entropy.

    Differentiation is supported through ``substitution_costs`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    kernel_mask = _kernel_allowed_transpositions(
        substitution_costs,
        lengths,
        allowed_transpositions,
    )
    return _osa_operator.entropy(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        lengths=lengths,
        trans_mask=kernel_mask,
        mask=mask,
        dtype=dtype,
    )


__all__ = ["osa", "osa_entropy", "osa_value"]
