# SPDX-License-Identifier: Apache-2.0
"""The `mask=` boolean kwarg equals the manual orientation-correct infinity fill.

Users flag cells to exclude with a boolean mask; the op applies ``-inf`` (score-native)
or ``+inf`` (cost-native) internally, so callers never handle infinities. This pins that
``orihime.<op>(x, mask=m)`` is identical to writing the orientation-correct infinity by hand.
"""
from __future__ import annotations

import pytest
import torch

import orihime
from operator_cases import OPERATOR_CASES

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# op -> (cost_native, primary shape)
_CASES = {
    spec.name: (spec.cost_native, spec.shapes("mask_kwarg")[0])
    for spec in OPERATOR_CASES
    if spec.name != "cky"
}


@pytest.mark.parametrize("op", list(_CASES))
@pytest.mark.parametrize("observable", ("", "_value", "_entropy"))
def test_mask_kwarg_equals_manual_infinity(op: str, observable: str) -> None:
    cost, shape = _CASES[op]
    g = torch.Generator(device=_DEVICE).manual_seed(11)
    x = (torch.rand if cost else torch.randn)(shape, generator=g, device=_DEVICE)
    mask = torch.zeros(shape, dtype=torch.bool, device=_DEVICE)
    mask[(0,) + tuple(s - 1 for s in shape[1:])] = True
    x_inf = x.masked_fill(mask, float("inf") if cost else float("-inf"))
    fn = getattr(orihime, f"{op}{observable}")
    torch.testing.assert_close(fn(x, mask=mask), fn(x_inf), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("observable", ("", "_value", "_entropy"))
def test_cky_mask_kwarg_equals_manual_infinity(observable: str) -> None:
    g = torch.Generator(device=_DEVICE).manual_seed(12)
    merge = torch.randn(1, 4, 4, 4, generator=g, device=_DEVICE)
    leaf = torch.randn(1, 4, generator=g, device=_DEVICE)
    mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool, device=_DEVICE)
    mask[0, 0, 1, 3] = True
    fn = getattr(orihime, f"cky{observable}")
    torch.testing.assert_close(
        fn(merge, leaf, mask=mask),
        fn(merge.masked_fill(mask, float("-inf")), leaf),
        rtol=1e-5, atol=1e-6,
    )


def test_mask_must_be_boolean() -> None:
    x = torch.randn(1, 4, 4, device=_DEVICE)
    with pytest.raises(TypeError):
        orihime.sw(x, mask=torch.zeros(1, 4, 4, device=_DEVICE))  # float, not bool


def test_mask_shape_must_match() -> None:
    x = torch.randn(1, 4, 4, device=_DEVICE)
    with pytest.raises(ValueError):
        orihime.sw(x, mask=torch.zeros(1, 3, 3, dtype=torch.bool, device=_DEVICE))
