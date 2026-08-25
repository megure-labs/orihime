# SPDX-License-Identifier: Apache-2.0
"""No-graph vector-Jacobian products for Eisner dependency parsing."""

from __future__ import annotations

from torch import Tensor

from ..eisner import _eisner_operator
from ._base import VJPFields, _RawVJP
from ..ops import eisner as _kernel_bindings  # noqa: F401
from ..ops.eisner import *  # noqa: F401,F403


vjp_fields: VJPFields = ("temperature",)


def _vjp_one_impl(
    arc_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
) -> Tensor:
    internal = {"temperature": "temp"}[wrt]
    result = _eisner_operator._vjp_one(
        arc_scores,
        temperature,
        grad_map=cotangent,
        wrt=internal,
        lengths=lengths,
    )
    assert isinstance(result, Tensor)
    return result


_raw = _RawVJP(vjp_fields=vjp_fields, vjp_one=_vjp_one_impl)


def vjp_one(
    arc_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return one selected Eisner VJP field.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        arc_scores,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
        lengths=lengths,
    )


def vjp(
    arc_scores: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected Eisner VJP fields in a dictionary.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        arc_scores,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
        lengths=lengths,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
