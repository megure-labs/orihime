# SPDX-License-Identifier: Apache-2.0
"""Raw vector-Jacobian products for affine Needleman-Wunsch."""

from torch import Tensor

from ..nw import _nw_affine_operator
from ._base import VJPFields, _RawVJP
from ..ops import nw_affine as _kernel_bindings  # noqa: F401
from ..ops.nw_affine import *  # noqa: F401,F403


vjp_fields: VJPFields = (
    "gap_open_score",
    "gap_extend_score",
    "temperature",
)


def _vjp_one_impl(
    pair_scores: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    gap_open_score: float | Tensor,
    gap_extend_score: float | Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
) -> Tensor:
    internal = {
        "gap_open_score": "gap_open",
        "gap_extend_score": "gap_ext",
        "temperature": "temp",
    }[wrt]
    result = _nw_affine_operator._vjp_one(
        pair_scores,
        gap_open_score,
        gap_extend_score,
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
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return one selected affine Needleman-Wunsch map VJP.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        pair_scores,
        wrt=wrt,
        cotangent=cotangent,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        temperature=temperature,
        lengths=lengths,
    )


def vjp(
    pair_scores: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected affine Needleman-Wunsch map VJPs.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        pair_scores,
        wrt=wrt,
        cotangent=cotangent,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        temperature=temperature,
        lengths=lengths,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
