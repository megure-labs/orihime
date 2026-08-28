# SPDX-License-Identifier: Apache-2.0
"""No-graph vector-Jacobian products for Damerau-Levenshtein."""

from __future__ import annotations

from torch import Tensor

from ..damerau import (
    _damerau_operator,
    _validate_structural_inputs,
)
from ._base import VJPFields, _RawVJP
from ..ops import damerau as _kernel_bindings  # noqa: F401
from ..ops.damerau import *  # noqa: F401,F403


vjp_fields: VJPFields = (
    "insertion_cost",
    "deletion_cost",
    "transposition_cost",
    "temperature",
)

_FIELD_TO_INTERNAL = {
    "insertion_cost": "ins_cost",
    "deletion_cost": "del_cost",
    "transposition_cost": "trans_cost",
    "temperature": "temp",
}


def _vjp_one_impl(
    substitution_costs: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    insertion_cost: float | Tensor,
    deletion_cost: float | Tensor,
    transposition_cost: float | Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
    transposition_sources: Tensor | None,
) -> Tensor:
    lengths, transposition_sources = _validate_structural_inputs(
        substitution_costs,
        lengths,
        transposition_sources,
    )
    result = _damerau_operator._vjp_one(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        grad_map=cotangent,
        wrt=_FIELD_TO_INTERNAL[wrt],
        lengths=lengths,
        trans_src=transposition_sources,
    )
    assert isinstance(result, Tensor)
    return result


_raw = _RawVJP(vjp_fields=vjp_fields, vjp_one=_vjp_one_impl)


def vjp_one(
    substitution_costs: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    transposition_sources: Tensor | None = None,
) -> Tensor:
    """Return one selected Damerau-Levenshtein map VJP.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        substitution_costs,
        wrt=wrt,
        cotangent=cotangent,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        transposition_cost=transposition_cost,
        temperature=temperature,
        lengths=lengths,
        transposition_sources=transposition_sources,
    )


def vjp(
    substitution_costs: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
    transposition_sources: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected Damerau-Levenshtein map VJPs.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        substitution_costs,
        wrt=wrt,
        cotangent=cotangent,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        transposition_cost=transposition_cost,
        temperature=temperature,
        lengths=lengths,
        transposition_sources=transposition_sources,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
