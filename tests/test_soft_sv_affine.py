"""Canonical Saigo-Vert affine local-alignment regression tests."""

import itertools
import math

import numpy as np
import pytest
import torch

from d2p import ops


GAP_OPEN = -2.0
GAP_EXT = -0.5
TEMP = 1.0


def _gap_cost(length, gap_open=GAP_OPEN, gap_ext=GAP_EXT):
    return 0.0 if length == 0 else gap_open + (length - 1) * gap_ext


def _canonical_z_brute(scores, gap_open=GAP_OPEN, gap_ext=GAP_EXT, temp=TEMP):
    """Enumerate each monotone matched-pair skeleton exactly once."""
    length1, length2 = scores.shape
    total = 1.0  # Empty alignment.
    for num_matches in range(1, min(length1, length2) + 1):
        for indices1 in itertools.combinations(range(length1), num_matches):
            for indices2 in itertools.combinations(range(length2), num_matches):
                value = sum(scores[indices1[k], indices2[k]] for k in range(num_matches))
                value += sum(
                    _gap_cost(indices1[k + 1] - indices1[k] - 1, gap_open, gap_ext)
                    + _gap_cost(indices2[k + 1] - indices2[k] - 1, gap_open, gap_ext)
                    for k in range(num_matches - 1)
                )
                total += np.exp(value / temp)
    return total


def _scores(seed=1234, shape=(1, 4, 5)):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def test_operator_names_resolve():
    assert hasattr(torch.ops.d2p, "sv_affine_forward")
    assert hasattr(torch.ops.d2p, "soft_sv_affine_float")


def test_forward_matches_canonical_brute_force():
    scores = _scores(shape=(1, 4, 4))
    value, _ = ops.sv_affine.forward(scores, GAP_OPEN, GAP_EXT, TEMP)
    expected = TEMP * math.log(_canonical_z_brute(scores[0].double().numpy()))
    assert value.item() == pytest.approx(expected, abs=1e-4)


def test_marginals_match_value_finite_difference():
    scores = _scores(shape=(1, 3, 4))
    _, marginals = ops.sv_affine.forward(scores, GAP_OPEN, GAP_EXT, TEMP)
    eps = 1e-3
    finite_difference = torch.empty_like(scores)
    for i in range(scores.size(1)):
        for j in range(scores.size(2)):
            plus = scores.clone()
            minus = scores.clone()
            plus[0, i, j] += eps
            minus[0, i, j] -= eps
            value_plus = ops.sv_affine.forward(plus, GAP_OPEN, GAP_EXT, TEMP)[0]
            value_minus = ops.sv_affine.forward(minus, GAP_OPEN, GAP_EXT, TEMP)[0]
            finite_difference[0, i, j] = (value_plus - value_minus) / (2 * eps)
    torch.testing.assert_close(marginals, finite_difference, rtol=2e-3, atol=3e-4)


def test_both_sided_skip_distinguishes_sv_from_sw():
    scores = torch.full((1, 3, 3), -8.0)
    scores[0, 0, 0] = 4.0
    scores[0, 2, 2] = 4.0
    sv_value = ops.sv_affine.forward(scores, GAP_OPEN, GAP_EXT, TEMP)[0]
    sw_value = ops.sw_affine.forward(scores, GAP_OPEN, GAP_EXT, TEMP)[0]
    assert (sv_value - sw_value).abs().item() > 0.1


def test_hvp_matches_marginal_directional_finite_difference():
    scores = _scores(shape=(1, 4, 5))
    tangent = _scores(seed=5678, shape=(1, 4, 5))
    actual = ops.sv_affine.marginals_hvp(
        scores, tangent, GAP_OPEN, GAP_EXT, TEMP
    )
    eps = 2e-3
    plus = ops.sv_affine.forward(
        scores + eps * tangent, GAP_OPEN, GAP_EXT, TEMP
    )[1]
    minus = ops.sv_affine.forward(
        scores - eps * tangent, GAP_OPEN, GAP_EXT, TEMP
    )[1]
    expected = (plus - minus) / (2 * eps)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-5)


def test_parameter_gradients_and_marginal_jacobians_match_finite_difference():
    scores = _scores(seed=4321, shape=(1, 4, 5))
    grad_open, grad_ext, grad_temp = ops.sv_affine.value_grad_params(
        scores, GAP_OPEN, GAP_EXT, TEMP
    )
    jacobians = (
        ops.sv_affine.marginals_grad_gap_open(scores, GAP_OPEN, GAP_EXT, TEMP),
        ops.sv_affine.marginals_grad_gap_ext(scores, GAP_OPEN, GAP_EXT, TEMP),
        ops.sv_affine.marginals_grad_temp(scores, GAP_OPEN, GAP_EXT, TEMP),
    )
    eps = 2e-3
    cases = (
        ((GAP_OPEN + eps, GAP_EXT, TEMP), (GAP_OPEN - eps, GAP_EXT, TEMP), grad_open),
        ((GAP_OPEN, GAP_EXT + eps, TEMP), (GAP_OPEN, GAP_EXT - eps, TEMP), grad_ext),
        ((GAP_OPEN, GAP_EXT, TEMP + eps), (GAP_OPEN, GAP_EXT, TEMP - eps), grad_temp),
    )
    for jacobian, (plus_params, minus_params, value_grad) in zip(jacobians, cases):
        value_plus, marginals_plus = ops.sv_affine.forward(scores, *plus_params)
        value_minus, marginals_minus = ops.sv_affine.forward(scores, *minus_params)
        value_expected = (value_plus - value_minus) / (2 * eps)
        jacobian_expected = (marginals_plus - marginals_minus) / (2 * eps)
        torch.testing.assert_close(value_grad, value_expected, rtol=2e-3, atol=3e-4)
        torch.testing.assert_close(jacobian, jacobian_expected, rtol=3e-3, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_parity():
    scores = _scores(seed=20260723, shape=(2, 5, 6))
    lengths = torch.tensor([[5, 6], [3, 4]], dtype=torch.int32)
    value_cpu, marginals_cpu = ops.sv_affine.forward(
        scores, GAP_OPEN, GAP_EXT, TEMP, lengths
    )
    value_cuda, marginals_cuda = ops.sv_affine.forward(
        scores.cuda(), GAP_OPEN, GAP_EXT, TEMP, lengths.cuda()
    )
    torch.testing.assert_close(value_cpu, value_cuda.cpu(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(marginals_cpu, marginals_cuda.cpu(), rtol=1e-4, atol=1e-5)
