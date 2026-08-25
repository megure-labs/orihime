# SPDX-License-Identifier: Apache-2.0
"""No-graph vector-Jacobian products for Dynamic Time Warping."""

from __future__ import annotations

from torch import Tensor

from ..dtw import _dtw_operator, _validate_structural_inputs
from ._base import VJPFields, _RawVJP
from ..ops import dtw as _kernel_bindings  # noqa: F401
from ..ops.dtw import *  # noqa: F401,F403


vjp_fields: VJPFields = ("temperature",)


def _vjp_one_impl(
    costs: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
    bandwidth: int | None,
) -> Tensor:
    _validate_structural_inputs(lengths, bandwidth)
    internal = {"temperature": "temp"}[wrt]
    result = _dtw_operator._vjp_one(
        costs,
        temperature,
        grad_map=cotangent,
        wrt=internal,
        lengths=lengths,
        bandwidth=bandwidth,
    )
    assert isinstance(result, Tensor)
    return result


_raw = _RawVJP(vjp_fields=vjp_fields, vjp_one=_vjp_one_impl)


def vjp_one(
    costs: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    bandwidth: int | None = None,
) -> Tensor:
    """Return the selected DTW map VJP as a tensor.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        costs,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
        lengths=lengths,
        bandwidth=bandwidth,
    )


def vjp(
    costs: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    bandwidth: int | None = None,
) -> dict[str, Tensor]:
    """Return selected DTW map VJPs in a type-stable dictionary.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        costs,
        wrt=wrt,
        cotangent=cotangent,
        temperature=temperature,
        lengths=lengths,
        bandwidth=bandwidth,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
