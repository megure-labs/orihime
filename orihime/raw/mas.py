# SPDX-License-Identifier: Apache-2.0
"""No-graph vector-Jacobian products for Monotonic Alignment Search."""

from __future__ import annotations

from torch import Tensor

from ..mas import _mas_operator
from ._base import VJPFields, _RawVJP
from ..ops import mas as _kernel_bindings  # noqa: F401
from ..ops.mas import *  # noqa: F401,F403


vjp_fields: VJPFields = ("temperature",)


def _vjp_one_impl(
    scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
) -> Tensor:
    internal = {"temperature": "temp"}[wrt]
    result = _mas_operator._vjp_one(
        scores,
        temperature,
        grad_map=cotangent,
        wrt=internal,
        lengths=lengths,
    )
    assert isinstance(result, Tensor)
    return result


_raw = _RawVJP(vjp_fields=vjp_fields, vjp_one=_vjp_one_impl)


def vjp_one(
    scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return one selected MAS VJP field.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        scores,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
        lengths=lengths,
    )


def vjp(
    scores: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected MAS VJP fields in a dictionary.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        scores,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
        lengths=lengths,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
