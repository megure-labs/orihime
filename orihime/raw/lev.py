# SPDX-License-Identifier: Apache-2.0
"""No-graph vector-Jacobian products for soft Levenshtein."""

from __future__ import annotations

from torch import Tensor

from ..edit_distance import _lev_operator
from ._base import VJPFields, _RawVJP
from ..ops import lev as _kernel_bindings  # noqa: F401
from ..ops.lev import *  # noqa: F401,F403


vjp_fields: VJPFields = (
    "insertion_cost",
    "deletion_cost",
    "temperature",
)


def _vjp_one_impl(
    substitution_costs: Tensor,
    *,
    wrt: str,
    cotangent: Tensor,
    insertion_cost: float | Tensor,
    deletion_cost: float | Tensor,
    temperature: float | Tensor,
    lengths: Tensor | None,
) -> Tensor:
    internal = {
        "insertion_cost": "ins",
        "deletion_cost": "del",
        "temperature": "temp",
    }[wrt]
    result = _lev_operator._vjp_one(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        temperature,
        grad_map=cotangent,
        wrt=internal,
        lengths=lengths,
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
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> Tensor:
    """Return one selected Levenshtein map VJP field.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp_one(
        substitution_costs,
        wrt=wrt,
        cotangent=cotangent,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        temperature=temperature,
        lengths=lengths,
    )


def vjp(
    substitution_costs: Tensor,
    *,
    wrt: VJPFields,
    cotangent: Tensor,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected Levenshtein map VJP fields as a dictionary.

    ``cotangent`` must be contiguous FP32 with the exact map shape and device.
    """

    return _raw.vjp(
        substitution_costs,
        wrt=wrt,
        cotangent=cotangent,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        temperature=temperature,
        lengths=lengths,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
