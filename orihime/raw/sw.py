# SPDX-License-Identifier: Apache-2.0
"""No-graph vector-Jacobian products for Smith-Waterman."""

from __future__ import annotations

from torch import Tensor

from ..sw import _sw_operator
from ._base import VJPFields, _RawVJP
from ..ops import sw as _kernel_bindings  # noqa: F401
from ..ops.sw import *  # noqa: F401,F403


vjp_fields: VJPFields = ("gap_score", "temperature")


def _vjp_one_impl(
    pair_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    gap_score: float | Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
) -> Tensor:
    internal = {
        "gap_score": "gap",
        "temperature": "temp",
    }[wrt]
    result = _sw_operator._vjp_one(
        pair_scores,
        gap_score,
        temperature,
        grad_map=cotangent,
        wrt=internal,
        lengths=lengths,
    )
    assert isinstance(result, Tensor)
    return result


_raw = _RawVJP(vjp_fields=vjp_fields, vjp_one=_vjp_one_impl)


def vjp_one(
    pair_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return one selected Smith-Waterman parameter VJP.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        pair_scores,
        wrt=wrt,
        cotangent=cotangent,
        gap_score=gap_score,
        temperature=temperature,
        lengths=lengths,
    )


def vjp(
    pair_scores: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    gap_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected Smith-Waterman parameter VJPs in a dictionary.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        pair_scores,
        wrt=wrt,
        cotangent=cotangent,
        gap_score=gap_score,
        temperature=temperature,
        lengths=lengths,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
