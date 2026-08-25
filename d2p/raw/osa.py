# SPDX-License-Identifier: Apache-2.0
"""No-graph VJPs for Optimal String Alignment."""

from __future__ import annotations

from torch import Tensor

from ..osa import _kernel_allowed_transpositions, _osa_operator
from ._base import VJPFields, _RawVJP
from ..ops import osa as _kernel_bindings  # noqa: F401
from ..ops.osa import *  # noqa: F401,F403


vjp_fields: VJPFields = (
    "insertion_cost",
    "deletion_cost",
    "transposition_cost",
    "temperature",
)


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
    allowed_transpositions: Tensor | None,
) -> Tensor:
    internal = {
        "insertion_cost": "ins_cost",
        "deletion_cost": "del_cost",
        "transposition_cost": "trans_cost",
        "temperature": "temp",
    }[wrt]
    kernel_mask = _kernel_allowed_transpositions(
        substitution_costs,
        lengths,
        allowed_transpositions,
    )
    result = _osa_operator._vjp_one(
        substitution_costs,
        insertion_cost,
        deletion_cost,
        transposition_cost,
        temperature,
        grad_map=cotangent,
        wrt=internal,
        lengths=lengths,
        trans_mask=kernel_mask,
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
    allowed_transpositions: Tensor | None = None,
) -> Tensor:
    """Return one selected Optimal String Alignment map VJP.

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
        allowed_transpositions=allowed_transpositions,
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
    allowed_transpositions: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return selected Optimal String Alignment map VJPs.

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
        allowed_transpositions=allowed_transpositions,
    )


__all__ = ["vjp_fields", "vjp_one", "vjp", *_kernel_bindings.__all__]
