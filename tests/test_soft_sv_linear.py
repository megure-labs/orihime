"""Canonical Saigo-Vert linear-gap source regression tests."""

import itertools
import math

import numpy as np
import pytest
import torch

import orihime
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

sv_ops = orihime.ops._kernels["sv"]
sw_ops = orihime.ops._kernels["sw"]


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
    value, marginals = sv_ops.forward(scores, GAP, TEMP)
    grad_gap, grad_temp = sv_ops.value_grad_params(scores, GAP, TEMP)

    assert value.item() == pytest.approx(expected[0], abs=1e-4)
    np.testing.assert_allclose(marginals[0].numpy(), expected[1], rtol=2e-4, atol=2e-5)
    assert grad_gap.item() == pytest.approx(expected[2], abs=2e-4)
    assert grad_temp.item() == pytest.approx(expected[3], abs=2e-4)


def test_both_sided_skip_is_not_ordinary_sw():
    scores = torch.full((1, 3, 3), -8.0)
    scores[0, 0, 0] = 4.0
    scores[0, 2, 2] = 4.0
    sv_value = sv_ops.forward(scores, GAP, TEMP)[0]
    sw_value = sw_ops.forward(scores, GAP, TEMP)[0]
    assert (sv_value - sw_value).abs().item() > 0.1


def test_hvp_matches_marginal_directional_finite_difference():
    scores = _scores(shape=(1, 4, 5))
    tangent = _scores(seed=5678, shape=(1, 4, 5))
    actual = sv_ops.marginals_hvp(scores, tangent, GAP, TEMP)
    eps = 2e-3
    plus = sv_ops.forward(scores + eps * tangent, GAP, TEMP)[1]
    minus = sv_ops.forward(scores - eps * tangent, GAP, TEMP)[1]
    expected = (plus - minus) / (2 * eps)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-5)


def test_marginal_parameter_jacobians_match_finite_difference():
    scores = _scores(seed=4321, shape=(1, 4, 5))
    actual_gap = sv_ops.marginals_grad_gap(scores, GAP, TEMP)
    actual_temp = sv_ops.marginals_grad_temp(scores, GAP, TEMP)
    eps = 2e-3

    plus_gap = sv_ops.forward(scores, GAP + eps, TEMP)[1]
    minus_gap = sv_ops.forward(scores, GAP - eps, TEMP)[1]
    plus_temp = sv_ops.forward(scores, GAP, TEMP + eps)[1]
    minus_temp = sv_ops.forward(scores, GAP, TEMP - eps)[1]

    torch.testing.assert_close(
        actual_gap, (plus_gap - minus_gap) / (2 * eps), rtol=3e-3, atol=5e-5
    )
    torch.testing.assert_close(
        actual_temp,
        (plus_temp - minus_temp) / (2 * eps),
        rtol=3e-3,
        atol=5e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA or ROCm build",
)
def test_cpu_gpu_parity():
    scores = _scores(seed=20260723, shape=(2, 5, 6))
    lengths = torch.tensor([[5, 6], [3, 4]], dtype=torch.int32)
    value_cpu, marginals_cpu = sv_ops.forward(scores, GAP, TEMP, lengths)
    value_gpu, marginals_gpu = sv_ops.forward(
        scores.cuda(), GAP, TEMP, lengths.cuda()
    )
    torch.testing.assert_close(value_cpu, value_gpu.cpu(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(marginals_cpu, marginals_gpu.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA or ROCm build",
)
def test_complete_gpu_primitive_surface_matches_cpu():
    scores = _scores(seed=20260827, shape=(2, 4, 5))
    tangent = _scores(seed=20260828, shape=(2, 4, 5))
    lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32)
    gap = torch.tensor([GAP], dtype=scores.dtype)
    temp = torch.tensor([TEMP], dtype=scores.dtype)

    cpu_results = (
        sv_ops.forward_t(scores, gap, temp, lengths),
        sv_ops.value_grad_params(scores, GAP, TEMP, lengths),
        sv_ops.marginals_backward(scores, tangent, GAP, TEMP, lengths),
        sv_ops.marginals_hvp(scores, tangent, GAP, TEMP, lengths),
        sv_ops.marginals_grad_gap(scores, GAP, TEMP, lengths),
        sv_ops.marginals_grad_temp(scores, GAP, TEMP, lengths),
    )
    gpu_results = (
        sv_ops.forward_t(
            scores.cuda(), gap.cuda(), temp.cuda(), lengths.cuda()
        ),
        sv_ops.value_grad_params(
            scores.cuda(), GAP, TEMP, lengths.cuda()
        ),
        sv_ops.marginals_backward(
            scores.cuda(), tangent.cuda(), GAP, TEMP, lengths.cuda()
        ),
        sv_ops.marginals_hvp(
            scores.cuda(), tangent.cuda(), GAP, TEMP, lengths.cuda()
        ),
        sv_ops.marginals_grad_gap(
            scores.cuda(), GAP, TEMP, lengths.cuda()
        ),
        sv_ops.marginals_grad_temp(
            scores.cuda(), GAP, TEMP, lengths.cuda()
        ),
    )
    for cpu_result, gpu_result in zip(cpu_results, gpu_results):
        cpu_tensors = (
            cpu_result if isinstance(cpu_result, (tuple, list)) else (cpu_result,)
        )
        gpu_tensors = (
            gpu_result if isinstance(gpu_result, (tuple, list)) else (gpu_result,)
        )
        for cpu_tensor, gpu_tensor in zip(cpu_tensors, gpu_tensors):
            torch.testing.assert_close(
                cpu_tensor,
                gpu_tensor.cpu(),
                rtol=3e-4,
                atol=3e-5,
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA or ROCm build",
)
def test_public_gpu_map_value_entropy_and_backward():
    scores = _scores(seed=20260829, shape=(1, 4, 5)).cuda().requires_grad_(True)
    gap = torch.tensor([GAP], device="cuda", requires_grad=True)
    temp = torch.tensor([TEMP], device="cuda", requires_grad=True)
    alignment = orihime.sv_linear(
        scores,
        gap_score=gap,
        temperature=temp,
    )
    value = orihime.sv_linear_value(
        scores,
        gap_score=gap,
        temperature=temp,
    )
    entropy = orihime.sv_linear_entropy(scores, gap_score=GAP, temperature=TEMP)
    (alignment.square().sum() + value.sum() + entropy.sum()).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert gap.grad is not None and torch.isfinite(gap.grad).all()
    assert temp.grad is not None and torch.isfinite(temp.grad).all()


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
def test_tensor_parameters_reject_wrong_gpu_device():
    scores = torch.randn((1, 3, 4), device="cuda:0")
    gap = torch.tensor([GAP], device="cuda:1")

    with pytest.raises(ValueError, match=r"gap_score tensor must be on the same device"):
        orihime.sv_linear(scores, gap_score=gap)
