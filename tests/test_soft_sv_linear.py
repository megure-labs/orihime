"""Canonical Saigo-Vert linear-gap source regression tests."""

import itertools
import math

import numpy as np
import pytest
import torch

from orihime import ops


GAP = -0.75
TEMP = 0.9


def _canonical_oracle(scores, gap=GAP, temp=TEMP):
    """Exhaustively enumerate each monotone matched-pair skeleton once."""
    length1, length2 = scores.shape
    paths = [(0.0, (), 0)]  # Exactly one explicit empty alignment.
    for num_matches in range(1, min(length1, length2) + 1):
        for indices1 in itertools.combinations(range(length1), num_matches):
            for indices2 in itertools.combinations(range(length2), num_matches):
                pairs = tuple(zip(indices1, indices2))
                gap_symbols = sum(
                    indices1[k + 1]
                    - indices1[k]
                    - 1
                    + indices2[k + 1]
                    - indices2[k]
                    - 1
                    for k in range(num_matches - 1)
                )
                path_score = sum(scores[i, j] for i, j in pairs)
                paths.append((path_score + gap * gap_symbols, pairs, gap_symbols))

    logits = np.asarray([path[0] / temp for path in paths], dtype=np.float64)
    max_logit = logits.max()
    weights = np.exp(logits - max_logit)
    weights /= weights.sum()
    value = temp * (max_logit + math.log(np.exp(logits - max_logit).sum()))

    marginals = np.zeros_like(scores, dtype=np.float64)
    for weight, (_, pairs, _) in zip(weights, paths):
        for i, j in pairs:
            marginals[i, j] += weight

    expected_path_score = sum(weight * path[0] for weight, path in zip(weights, paths))
    grad_gap = sum(weight * path[2] for weight, path in zip(weights, paths))
    grad_temp = (value - expected_path_score) / temp
    return value, marginals, grad_gap, grad_temp


def _scores(seed=1234, shape=(1, 4, 5)):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def test_operator_names_resolve():
    assert hasattr(torch.ops.orihime, "sv_linear_forward")
    assert hasattr(torch.ops.orihime, "soft_sv_linear_float")


def test_forward_marginals_and_parameter_grads_match_exhaustive_oracle():
    scores = _scores(shape=(1, 4, 4))
    expected = _canonical_oracle(scores[0].double().numpy())
    value, marginals = ops.sv_linear.forward(scores, GAP, TEMP)
    grad_gap, grad_temp = ops.sv_linear.value_grad_params(scores, GAP, TEMP)

    assert value.item() == pytest.approx(expected[0], abs=1e-4)
    np.testing.assert_allclose(marginals[0].numpy(), expected[1], rtol=2e-4, atol=2e-5)
    assert grad_gap.item() == pytest.approx(expected[2], abs=2e-4)
    assert grad_temp.item() == pytest.approx(expected[3], abs=2e-4)


def test_both_sided_skip_is_not_ordinary_sw():
    scores = torch.full((1, 3, 3), -8.0)
    scores[0, 0, 0] = 4.0
    scores[0, 2, 2] = 4.0
    sv_value = ops.sv_linear.forward(scores, GAP, TEMP)[0]
    sw_value = ops.sw.forward(scores, GAP, TEMP)[0]
    assert (sv_value - sw_value).abs().item() > 0.1


def test_hvp_matches_marginal_directional_finite_difference():
    scores = _scores(shape=(1, 4, 5))
    tangent = _scores(seed=5678, shape=(1, 4, 5))
    actual = ops.sv_linear.marginals_hvp(scores, tangent, GAP, TEMP)
    eps = 2e-3
    plus = ops.sv_linear.forward(scores + eps * tangent, GAP, TEMP)[1]
    minus = ops.sv_linear.forward(scores - eps * tangent, GAP, TEMP)[1]
    expected = (plus - minus) / (2 * eps)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-5)


def test_marginal_parameter_jacobians_match_finite_difference():
    scores = _scores(seed=4321, shape=(1, 4, 5))
    actual_gap = ops.sv_linear.marginals_grad_gap(scores, GAP, TEMP)
    actual_temp = ops.sv_linear.marginals_grad_temp(scores, GAP, TEMP)
    eps = 2e-3

    plus_gap = ops.sv_linear.forward(scores, GAP + eps, TEMP)[1]
    minus_gap = ops.sv_linear.forward(scores, GAP - eps, TEMP)[1]
    plus_temp = ops.sv_linear.forward(scores, GAP, TEMP + eps)[1]
    minus_temp = ops.sv_linear.forward(scores, GAP, TEMP - eps)[1]

    torch.testing.assert_close(
        actual_gap, (plus_gap - minus_gap) / (2 * eps), rtol=3e-3, atol=5e-5
    )
    torch.testing.assert_close(
        actual_temp,
        (plus_temp - minus_temp) / (2 * eps),
        rtol=3e-3,
        atol=5e-5,
    )
