"""Canonical Saigo-Vert affine local-alignment regression tests."""

import itertools
import math

import numpy as np
import pytest
import torch

import orihime
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

sv_affine_ops = orihime.ops._kernels["sv_affine"]
sw_affine_ops = orihime.ops._kernels["sw_affine"]


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
    assert hasattr(torch.ops.orihime, "sv_affine_forward")
    assert hasattr(torch.ops.orihime, "soft_sv_affine_float")


def test_forward_matches_canonical_brute_force():
    scores = _scores(shape=(1, 4, 4))
    value, _ = sv_affine_ops.forward(scores, GAP_OPEN, GAP_EXT, TEMP)
    expected = TEMP * math.log(_canonical_z_brute(scores[0].double().numpy()))
    assert value.item() == pytest.approx(expected, abs=1e-4)


def test_marginals_match_value_finite_difference():
    scores = _scores(shape=(1, 3, 4))
    _, marginals = sv_affine_ops.forward(scores, GAP_OPEN, GAP_EXT, TEMP)
    eps = 1e-3
    finite_difference = torch.empty_like(scores)
    for i in range(scores.size(1)):
        for j in range(scores.size(2)):
            plus = scores.clone()
            minus = scores.clone()
            plus[0, i, j] += eps
            minus[0, i, j] -= eps
            value_plus = sv_affine_ops.forward(plus, GAP_OPEN, GAP_EXT, TEMP)[0]
            value_minus = sv_affine_ops.forward(minus, GAP_OPEN, GAP_EXT, TEMP)[0]
            finite_difference[0, i, j] = (value_plus - value_minus) / (2 * eps)
    torch.testing.assert_close(marginals, finite_difference, rtol=2e-3, atol=3e-4)


def test_both_sided_skip_distinguishes_sv_from_sw():
    scores = torch.full((1, 3, 3), -8.0)
    scores[0, 0, 0] = 4.0
    scores[0, 2, 2] = 4.0
    sv_value = sv_affine_ops.forward(scores, GAP_OPEN, GAP_EXT, TEMP)[0]
    sw_value = sw_affine_ops.forward(scores, GAP_OPEN, GAP_EXT, TEMP)[0]
    assert (sv_value - sw_value).abs().item() > 0.1


def test_hvp_matches_marginal_directional_finite_difference():
    scores = _scores(shape=(1, 4, 5))
    tangent = _scores(seed=5678, shape=(1, 4, 5))
    actual = sv_affine_ops.marginals_hvp(
        scores, tangent, GAP_OPEN, GAP_EXT, TEMP
    )
    eps = 2e-3
    plus = sv_affine_ops.forward(
        scores + eps * tangent, GAP_OPEN, GAP_EXT, TEMP
    )[1]
    minus = sv_affine_ops.forward(
        scores - eps * tangent, GAP_OPEN, GAP_EXT, TEMP
    )[1]
    expected = (plus - minus) / (2 * eps)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-5)


def test_parameter_gradients_and_marginal_jacobians_match_finite_difference():
    scores = _scores(seed=4321, shape=(1, 4, 5))
    grad_open, grad_ext, grad_temp = sv_affine_ops.value_grad_params(
        scores, GAP_OPEN, GAP_EXT, TEMP
    )
    jacobians = (
        sv_affine_ops.marginals_grad_gap_open(scores, GAP_OPEN, GAP_EXT, TEMP),
        sv_affine_ops.marginals_grad_gap_ext(scores, GAP_OPEN, GAP_EXT, TEMP),
        sv_affine_ops.marginals_grad_temp(scores, GAP_OPEN, GAP_EXT, TEMP),
    )
    eps = 2e-3
    cases = (
        ((GAP_OPEN + eps, GAP_EXT, TEMP), (GAP_OPEN - eps, GAP_EXT, TEMP), grad_open),
        ((GAP_OPEN, GAP_EXT + eps, TEMP), (GAP_OPEN, GAP_EXT - eps, TEMP), grad_ext),
        ((GAP_OPEN, GAP_EXT, TEMP + eps), (GAP_OPEN, GAP_EXT, TEMP - eps), grad_temp),
    )
    for jacobian, (plus_params, minus_params, value_grad) in zip(jacobians, cases):
        value_plus, marginals_plus = sv_affine_ops.forward(scores, *plus_params)
        value_minus, marginals_minus = sv_affine_ops.forward(scores, *minus_params)
        value_expected = (value_plus - value_minus) / (2 * eps)
        jacobian_expected = (marginals_plus - marginals_minus) / (2 * eps)
        torch.testing.assert_close(value_grad, value_expected, rtol=2e-3, atol=3e-4)
        torch.testing.assert_close(jacobian, jacobian_expected, rtol=3e-3, atol=5e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA or ROCm build",
)
def test_cpu_gpu_parity():
    scores = _scores(seed=20260723, shape=(2, 5, 6))
    lengths = torch.tensor([[5, 6], [3, 4]], dtype=torch.int32)
    value_cpu, marginals_cpu = sv_affine_ops.forward(
        scores, GAP_OPEN, GAP_EXT, TEMP, lengths
    )
    value_gpu, marginals_gpu = sv_affine_ops.forward(
        scores.cuda(), GAP_OPEN, GAP_EXT, TEMP, lengths.cuda()
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
    gap_open = torch.tensor([GAP_OPEN], dtype=scores.dtype)
    gap_ext = torch.tensor([GAP_EXT], dtype=scores.dtype)
    temp = torch.tensor([TEMP], dtype=scores.dtype)

    cpu_results = (
        sv_affine_ops.forward_t(
            scores, gap_open, gap_ext, temp, lengths
        ),
        sv_affine_ops.value_grad_params(
            scores, GAP_OPEN, GAP_EXT, TEMP, lengths
        ),
        sv_affine_ops.marginals_backward(
            scores, tangent, GAP_OPEN, GAP_EXT, TEMP, lengths
        ),
        sv_affine_ops.marginals_hvp(
            scores, tangent, GAP_OPEN, GAP_EXT, TEMP, lengths
        ),
        sv_affine_ops.marginals_grad_gap_open(
            scores, GAP_OPEN, GAP_EXT, TEMP, lengths
        ),
        sv_affine_ops.marginals_grad_gap_ext(
            scores, GAP_OPEN, GAP_EXT, TEMP, lengths
        ),
        sv_affine_ops.marginals_grad_temp(
            scores, GAP_OPEN, GAP_EXT, TEMP, lengths
        ),
    )
    gpu_scores = scores.cuda()
    gpu_tangent = tangent.cuda()
    gpu_lengths = lengths.cuda()
    gpu_results = (
        sv_affine_ops.forward_t(
            gpu_scores,
            gap_open.cuda(),
            gap_ext.cuda(),
            temp.cuda(),
            gpu_lengths,
        ),
        sv_affine_ops.value_grad_params(
            gpu_scores, GAP_OPEN, GAP_EXT, TEMP, gpu_lengths
        ),
        sv_affine_ops.marginals_backward(
            gpu_scores, gpu_tangent, GAP_OPEN, GAP_EXT, TEMP, gpu_lengths
        ),
        sv_affine_ops.marginals_hvp(
            gpu_scores, gpu_tangent, GAP_OPEN, GAP_EXT, TEMP, gpu_lengths
        ),
        sv_affine_ops.marginals_grad_gap_open(
            gpu_scores, GAP_OPEN, GAP_EXT, TEMP, gpu_lengths
        ),
        sv_affine_ops.marginals_grad_gap_ext(
            gpu_scores, GAP_OPEN, GAP_EXT, TEMP, gpu_lengths
        ),
        sv_affine_ops.marginals_grad_temp(
            gpu_scores, GAP_OPEN, GAP_EXT, TEMP, gpu_lengths
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
    gap_open = torch.tensor([GAP_OPEN], device="cuda", requires_grad=True)
    gap_ext = torch.tensor([GAP_EXT], device="cuda", requires_grad=True)
    temp = torch.tensor([TEMP], device="cuda", requires_grad=True)
    alignment = orihime.sv_affine(
        scores,
        gap_open_score=gap_open,
        gap_extend_score=gap_ext,
        temperature=temp,
    )
    value = orihime.sv_affine_value(
        scores,
        gap_open_score=gap_open,
        gap_extend_score=gap_ext,
        temperature=temp,
    )
    entropy = orihime.sv_affine_entropy(
        scores,
        gap_open_score=GAP_OPEN,
        gap_extend_score=GAP_EXT,
        temperature=TEMP,
    )
    (alignment.square().sum() + value.sum() + entropy.sum()).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    for parameter in (gap_open, gap_ext, temp):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
def test_tensor_parameters_reject_wrong_gpu_device():
    scores = torch.randn((1, 3, 4), device="cuda:0")
    gap_open = torch.tensor([GAP_OPEN], device="cuda:1")

    with pytest.raises(
        ValueError,
        match=r"gap_open_score tensor must be on the same device",
    ):
        orihime.sv_affine(scores, gap_open_score=gap_open)
