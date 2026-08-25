# SPDX-License-Identifier: Apache-2.0
"""Cost-native differentiable Damerau-Levenshtein functions.

Lower substitution costs make a cell more preferred.  The shipped kernels
accept exactly one batch dimension, so substitution costs have shape
``[B, L1, L2]`` and lengths have shape ``[B, 2]``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._pt2_utils import use_pt2_ops
from .operator import Operator, _document_public_function
from .ops.damerau import kernels as _damerau_kernels


_damerau_operator = Operator(
    "damerau",
    params=("ins_cost", "del_cost", "trans_cost", "temp"),
    kernels=_damerau_kernels,
    tensor_inputs=("substitution_costs",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)


def _raise_if_true(condition: Tensor, message: str) -> None:
    functorch = torch._C._functorch
    if functorch.is_batchedtensor(condition):
        physical_condition = condition
        while functorch.is_batchedtensor(physical_condition):
            physical_condition = functorch.get_unwrapped(
                physical_condition
            )
        torch._assert_async(~physical_condition.any(), message)
    elif use_pt2_ops(condition):
        torch._assert_async(~condition, message)
    elif bool(condition.item()):
        raise ValueError(message)


def _validate_primary_input(substitution_costs: Tensor) -> None:
    if not isinstance(substitution_costs, Tensor):
        raise TypeError("substitution_costs must be a tensor")
    if substitution_costs.ndim != 3:
        raise ValueError(
            "substitution_costs must have shape [B, L1, L2]"
        )


def _validate_builder_lengths(
    lengths: Tensor | None,
    *,
    batch_size: int,
    length_1: int,
    length_2: int,
    device: torch.device,
) -> Tensor | None:
    if lengths is None:
        return None
    if not isinstance(lengths, Tensor):
        raise TypeError("lengths must be a tensor or None")
    if lengths.dtype != torch.int32:
        raise TypeError("lengths must have dtype torch.int32")
    if lengths.shape != (batch_size, 2):
        raise ValueError(
            f"lengths must have shape [{batch_size}, 2]"
        )
    if lengths.device != device:
        raise ValueError(
            "lengths must be on the same device as the primary input"
        )
    if not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")

    lower_invalid = lengths < 0
    upper_bounds = torch.tensor(
        (length_1, length_2),
        dtype=torch.int32,
        device=device,
    )
    upper_invalid = lengths > upper_bounds
    _raise_if_true(
        torch.any(lower_invalid | upper_invalid),
        "lengths entries must lie within the padded input shape",
    )
    return lengths


def _validate_transposition_sources(
    substitution_costs: Tensor,
    lengths: Tensor | None,
    transposition_sources: Tensor | None,
) -> None:
    if transposition_sources is None:
        return
    if not isinstance(transposition_sources, Tensor):
        raise TypeError(
            "transposition_sources must be a tensor or None"
        )
    if transposition_sources.dtype != torch.int32:
        raise TypeError(
            "transposition_sources must have dtype torch.int32"
        )
    expected_shape = (*substitution_costs.shape, 2)
    if transposition_sources.shape != expected_shape:
        raise ValueError(
            "transposition_sources must have shape [B, L1, L2, 2] "
            "matching substitution_costs"
        )
    if transposition_sources.device != substitution_costs.device:
        raise ValueError(
            "transposition_sources must be on the same device as "
            "substitution_costs"
        )
    if not transposition_sources.is_contiguous():
        raise ValueError("transposition_sources must be contiguous")

    source_rows = transposition_sources[..., 0]
    source_columns = transposition_sources[..., 1]
    row_sentinel = source_rows == -1
    column_sentinel = source_columns == -1
    _raise_if_true(
        torch.any(row_sentinel != column_sentinel),
        "transposition_sources entries must use the exact "
        "(-1, -1) sentinel",
    )

    valid = ~row_sentinel
    batch_size, length_1, length_2 = substitution_costs.shape
    row = torch.arange(
        length_1,
        dtype=torch.int32,
        device=substitution_costs.device,
    ).view(1, length_1, 1)
    column = torch.arange(
        length_2,
        dtype=torch.int32,
        device=substitution_costs.device,
    ).view(1, 1, length_2)
    invalid_predecessor = valid & (
        (source_rows < 0)
        | (source_columns < 0)
        | (source_rows >= row)
        | (source_columns >= column)
    )
    _raise_if_true(
        torch.any(invalid_predecessor),
        "each transposition source must be (-1, -1) or a valid "
        "earlier predecessor",
    )

    if lengths is not None:
        active = (
            row < lengths[:, 0].view(batch_size, 1, 1)
        ) & (
            column < lengths[:, 1].view(batch_size, 1, 1)
        )
        _raise_if_true(
            torch.any(valid & ~active),
            "padded transposition_sources entries must be (-1, -1)",
        )


def _validate_structural_inputs(
    substitution_costs: Tensor,
    lengths: Tensor | None,
    transposition_sources: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    _validate_primary_input(substitution_costs)
    lengths = _damerau_operator._validate_lengths_contract(
        substitution_costs,
        lengths,
    )
    _validate_transposition_sources(
        substitution_costs,
        lengths,
        transposition_sources,
    )
    return lengths, transposition_sources


@_document_public_function
def build_damerau_transposition_sources(
    source_tokens: Tensor,
    target_tokens: Tensor,
    *,
    lengths: Tensor | None = None,
) -> Tensor:
    """Build true-Damerau predecessor coordinates from token sequences.

    ``source_tokens`` and ``target_tokens`` have shapes ``[B, L1]`` and
    ``[B, L2]``.  The returned ``torch.int32`` tensor has shape
    ``[B, L1, L2, 2]``.  Each active cell contains the most recent earlier
    source and target positions that form its transposition edge, or the
    exact ``(-1, -1)`` sentinel when no such edge exists.  Padded cells are
    always sentinel entries.
    """

    if not isinstance(source_tokens, Tensor):
        raise TypeError("source_tokens must be a tensor")
    if not isinstance(target_tokens, Tensor):
        raise TypeError("target_tokens must be a tensor")
    if source_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError(
            "source_tokens and target_tokens must have shapes [B, L]"
        )
    if source_tokens.shape[0] != target_tokens.shape[0]:
        raise ValueError(
            "source_tokens and target_tokens must have the same batch size"
        )
    if source_tokens.device != target_tokens.device:
        raise ValueError(
            "source_tokens and target_tokens must be on the same device"
        )
    if source_tokens.requires_grad or target_tokens.requires_grad:
        raise ValueError("token sequences must not require gradients")

    batch_size, length_1 = source_tokens.shape
    _, length_2 = target_tokens.shape
    lengths = _validate_builder_lengths(
        lengths,
        batch_size=batch_size,
        length_1=length_1,
        length_2=length_2,
        device=source_tokens.device,
    )
    result = torch.full(
        (batch_size, length_1, length_2, 2),
        -1,
        dtype=torch.int32,
        device=source_tokens.device,
    )
    if length_1 == 0 or length_2 == 0:
        return result

    matches = source_tokens.unsqueeze(2) == target_tokens.unsqueeze(1)
    row_indices = torch.arange(
        length_1,
        dtype=torch.int32,
        device=source_tokens.device,
    ).view(1, length_1, 1)
    column_indices = torch.arange(
        length_2,
        dtype=torch.int32,
        device=source_tokens.device,
    ).view(1, 1, length_2)

    matched_rows = torch.where(matches, row_indices, -1)
    previous_rows = torch.full(
        (batch_size, length_1, length_2),
        -1,
        dtype=torch.int32,
        device=source_tokens.device,
    )
    previous_rows[:, 1:] = torch.cummax(
        matched_rows, dim=1
    ).values[:, :-1]

    matched_columns = torch.where(matches, column_indices, -1)
    previous_columns = torch.full_like(previous_rows, -1)
    previous_columns[:, :, 1:] = torch.cummax(
        matched_columns, dim=2
    ).values[:, :, :-1]

    valid = (previous_rows >= 0) & (previous_columns >= 0)
    if lengths is not None:
        active = (
            row_indices < lengths[:, 0].view(batch_size, 1, 1)
        ) & (
            column_indices < lengths[:, 1].view(batch_size, 1, 1)
        )
        valid &= active
    result[..., 0] = torch.where(valid, previous_rows, -1)
    result[..., 1] = torch.where(valid, previous_columns, -1)
    return result


@_document_public_function
def damerau(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    transposition_sources: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the cost-native soft Damerau-Levenshtein map.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    lengths, transposition_sources = _validate_structural_inputs(
        substitution_costs,
        lengths,
        transposition_sources,
    )
    return _damerau_operator(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        lengths=lengths,
        trans_src=transposition_sources,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def damerau_value(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    transposition_sources: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the cost-native soft Damerau-Levenshtein value.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    lengths, transposition_sources = _validate_structural_inputs(
        substitution_costs,
        lengths,
        transposition_sources,
    )
    return _damerau_operator.value(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        lengths=lengths,
        trans_src=transposition_sources,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def damerau_entropy(
    substitution_costs: Tensor,
    *,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    transposition_sources: Tensor | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the soft Damerau-Levenshtein path model.

    Differentiation is supported through ``substitution_costs`` only; scalar
    parameter derivatives are not provided.

    Temperature must be finite and positive; finite costs and cost parameters
    must satisfy ``abs(value) / temperature <= 80``.
    """

    lengths, transposition_sources = _validate_structural_inputs(
        substitution_costs,
        lengths,
        transposition_sources,
    )
    return _damerau_operator.entropy(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        lengths=lengths,
        trans_src=transposition_sources,
        mask=mask,
        dtype=dtype,
    )


__all__ = [
    "build_damerau_transposition_sources",
    "damerau",
    "damerau_entropy",
    "damerau_value",
]
