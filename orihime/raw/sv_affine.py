# SPDX-License-Identifier: Apache-2.0
"""No-graph vector–Jacobian products for affine Saigo–Vert alignment."""

from __future__ import annotations

from torch import Tensor

from ..ops import sv_affine as _kernel_bindings  # noqa: F401
from ..ops.sv_affine import *  # noqa: F401,F403
from ..sv import _sv_affine_operator
from ._base import VJPFields, _RawVJP


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
    result = _sv_affine_operator._vjp_one(
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
    """Return one selected affine Saigo–Vert parameter VJP."""

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
    """Return selected affine Saigo–Vert parameter VJPs."""

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
