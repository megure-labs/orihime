# SPDX-License-Identifier: Apache-2.0
"""Type-stable raw vector-Jacobian products for CKY."""

from __future__ import annotations

from torch import Tensor

from ..cky import _cky_operator
from ._base import VJPFields, _RawVJP
from ..ops import cky as _kernel_bindings  # noqa: F401
from ..ops.cky import *  # noqa: F401,F403


vjp_fields: VJPFields = ("leaf_scores", "temperature")


def _vjp_one_impl(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor,
) -> Tensor:
    internal = {
        "leaf_scores": "leaf_scores",
        "temperature": "temp",
    }[wrt]
    result = _cky_operator._vjp_one(
        merge_scores,
        leaf_scores,
        temperature,
        grad_map=cotangent,
        wrt=internal,
    )
    assert isinstance(result, Tensor)
    return result


_raw = _RawVJP(vjp_fields=vjp_fields, vjp_one=_vjp_one_impl)


def vjp_one(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
) -> Tensor:
    """Return one selected CKY merge-map VJP field.

    ``cotangent`` must be contiguous FP32 with the exact merge-map shape and
    device.
    """

    return _raw.vjp_one(
        merge_scores,
        leaf_scores,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
    )


def vjp(
    merge_scores: Tensor,
    leaf_scores: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
) -> dict[str, Tensor]:
    """Return selected CKY merge-map VJP fields in a dictionary.

    ``cotangent`` must be contiguous FP32 with the exact merge-map shape and
    device.
    """

    return _raw.vjp(
        merge_scores,
        leaf_scores,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
