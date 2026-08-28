# SPDX-License-Identifier: Apache-2.0
"""Cost-native Dynamic Time Warping functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .operator import Operator, _document_public_function, _graph_safe_assert
from .ops.dtw import kernels as _kernels


_dtw_operator = Operator(
    "dtw",
    params=("temp",),
    kernels=_kernels,
    tensor_inputs=("costs",),
    tensor_shapes=("B,L1,L2",),
    length_axes=(-2, -1),
)


def _validate_structural_inputs(
    lengths: Tensor | None,
    bandwidth: int | None,
) -> None:
    if lengths is not None and not isinstance(lengths, Tensor):
        raise TypeError("lengths must be a tensor or None")
    if (
        isinstance(lengths, Tensor)
        and lengths.ndim == 2
        and lengths.shape[-1] == 2
    ):
        one_sided_empty = (lengths[:, 0] == 0) ^ (lengths[:, 1] == 0)
        _graph_safe_assert(
            ~one_sided_empty,
            "dtw received an infeasible/empty instance: one-sided empty "
            "lengths are not supported",
        )
    if bandwidth is not None:
        if isinstance(bandwidth, bool) or not isinstance(bandwidth, int):
            raise TypeError("bandwidth must be an integer or None")
        if bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")


@_document_public_function
def dtw(
    costs: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    bandwidth: int | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft alignment map for cost-native Dynamic Time Warping.

    Temperature must be finite and positive; finite costs must satisfy
    ``abs(value) / temperature <= 80``.
    """

    _validate_structural_inputs(lengths, bandwidth)
    return _dtw_operator(
        costs,
        temperature,
        lengths=lengths,
        bandwidth=bandwidth,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def dtw_value(
    costs: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    bandwidth: int | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the soft minimum path cost.

    Temperature must be finite and positive; finite costs must satisfy
    ``abs(value) / temperature <= 80``.
    """

    _validate_structural_inputs(lengths, bandwidth)
    return _dtw_operator.value(
        costs,
        temperature,
        lengths=lengths,
        bandwidth=bandwidth,
        mask=mask,
        dtype=dtype,
    )


@_document_public_function
def dtw_entropy(
    costs: Tensor,
    *,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    bandwidth: int | None = None,
    mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the entropy of the cost-native soft path distribution.

    Differentiation is supported through ``costs`` only; the temperature
    derivative is not provided.

    Temperature must be finite and positive; finite costs must satisfy
    ``abs(value) / temperature <= 80``.
    """

    _validate_structural_inputs(lengths, bandwidth)
    return _dtw_operator.entropy(
        costs,
        temperature,
        lengths=lengths,
        bandwidth=bandwidth,
        mask=mask,
        dtype=dtype,
    )


__all__ = ["dtw", "dtw_entropy", "dtw_value"]
