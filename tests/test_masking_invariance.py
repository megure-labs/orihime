"""Masking must not change the answer, at any temperature AND any in-domain input scale.

Two independent properties, both of which a fixed- or zero-relative sentinel violates:

1. answer-invariance (zero-centered): masked value/map/entropy equal a reference that
   excludes the same cell via an independent large ratio the test constructs (ratio 75).
2. domain-edge exclusion: with retained scores near the domain edge (|value|/T ~ 79), a
   masked cell's marginal must be ~0 (excluded), not dominant. This is the case a
   zero-relative sentinel gets wrong (masked marginal becomes 1.0).

Covers all 12 operators (CKY included) across low and high temperature.
"""
from __future__ import annotations

import pytest
import torch

import orihime
from operator_cases import OPERATOR_CASES

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TEMPS = (0.1, 1.0, 100.0, 10000.0)
_REF_RATIO = 75.0

# name -> (cost_native, primary shape, masked cell in the primary map)
_CASES = {
    spec.name: (
        spec.cost_native,
        spec.shapes("invariance")[0],
        spec.invariance_index,
    )
    for spec in OPERATOR_CASES
}


def _call(op, observable, primary, temperature):
    suffix = "" if observable == "map" else f"_{observable}"
    fn = getattr(orihime, f"{op}{suffix}")
    if op == "cky":  # cky also needs finite leaf scores
        leaf = torch.zeros(primary.shape[0], primary.shape[1], device=_DEVICE)
        return fn(primary, leaf, temperature=temperature)
    return fn(primary, temperature=temperature)


def _base(op, cost, shape, T, edge):
    g = torch.Generator(device=_DEVICE).manual_seed(4200 + abs(hash(op)) % 1000)
    if edge:  # retained cells near the domain edge (ratio ~79), still |value|/T <= 80
        r = 0.5 * T * torch.rand(shape, generator=g, device=_DEVICE)
        return (79.0 * T - r) if cost else (-79.0 * T + r)
    return torch.rand(shape, generator=g, device=_DEVICE) if cost else torch.randn(shape, generator=g, device=_DEVICE)


@pytest.mark.parametrize("op", list(_CASES))
@pytest.mark.parametrize("observable", ("value", "map", "entropy"))
@pytest.mark.parametrize("temperature", _TEMPS)
def test_masking_answer_invariant_zero_centered(op, observable, temperature):
    cost, shape, cell = _CASES[op]
    base = _base(op, cost, shape, temperature, edge=False)
    masked = base.clone(); masked[cell] = float("inf") if cost else float("-inf")
    reference = base.clone()
    reference[cell] = (_REF_RATIO if cost else -_REF_RATIO) * temperature
    got = _call(op, observable, masked, temperature)
    ref = _call(op, observable, reference, temperature)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-4), f"{op}/{observable}/T={temperature}: masking changed the answer"


@pytest.mark.parametrize("op", list(_CASES))
@pytest.mark.parametrize("temperature", _TEMPS)
def test_masked_cell_excluded_at_domain_edge(op, temperature):
    cost, shape, cell = _CASES[op]
    base = _base(op, cost, shape, temperature, edge=True)
    masked = base.clone(); masked[cell] = float("inf") if cost else float("-inf")
    gmap = _call(op, "map", masked, temperature)
    assert torch.isfinite(gmap).all(), f"{op}/T={temperature}: map not finite"
    assert gmap[cell].abs().item() < 1e-3, (
        f"{op}/T={temperature}: masked cell not excluded at domain edge "
        f"(marginal={gmap[cell].item()}; a zero-relative sentinel bug gives ~1.0)"
    )
