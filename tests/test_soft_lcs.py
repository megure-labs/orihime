# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft LCS (Longest Common Subsequence).
"""

import contextlib
import importlib
import os
import re
import site
import sys
from pathlib import Path

import pytest
import torch

from reference import lcs_forward_naive, lcs_naive
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED


def _skip_stale_orihime_editable_loader():
    editable_paths = []
    for site_dir in site.getsitepackages():
        loader_path = Path(site_dir) / "_orihime_editable_loader.py"
        if not loader_path.exists():
            continue
        match = re.search(
            r"install\(\s*'orihime',\s*\{'orihime'\},\s*'([^']+)'",
            loader_path.read_text(),
            re.S,
        )
        if match:
            editable_paths.append(match.group(1))
    if not editable_paths:
        return

    existing = [path for path in os.environ.get("MESONPY_EDITABLE_SKIP", "").split(os.pathsep) if path]
    for path in editable_paths:
        if path not in existing:
            existing.append(path)
    os.environ["MESONPY_EDITABLE_SKIP"] = os.pathsep.join(existing)


_skip_stale_orihime_editable_loader()

try:
    import orihime
    ORIHIME_AVAILABLE = True
except ImportError:
    sys.modules.pop("orihime", None)
    source_root = str(Path(__file__).resolve().parents[1])
    sys.path = [path for path in sys.path if path != source_root]
    for site_dir in reversed(site.getsitepackages()):
        if site_dir in sys.path:
            sys.path.remove(site_dir)
        sys.path.insert(0, site_dir)
    try:
        orihime = importlib.import_module("orihime")
        ORIHIME_AVAILABLE = True
    except ImportError:
        ORIHIME_AVAILABLE = False

if ORIHIME_AVAILABLE:
    lcs_ops = orihime.ops._kernels["lcs"]
    from operator_test_utils import lcs_forward_with_grads

CUDA_AVAILABLE = torch.cuda.is_available()
HVP_FINITE_DIFFERENCE_STEP = 5e-3


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def assert_padded_pairwise_region_zero(name, values, lengths):
    for batch_index, (length_1, length_2) in enumerate(lengths.tolist()):
        padded = torch.ones_like(values[batch_index], dtype=torch.bool)
        padded[:length_1, :length_2] = False
        assert torch.count_nonzero(values[batch_index][padded]).item() == 0, (
            f"{name} has non-zero padded cells for batch {batch_index}"
        )


def assert_tensors_on_device(result, device):
    tensors = (result,) if isinstance(result, torch.Tensor) else tuple(result)
    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device == device


def require_raises(fn, match):
    with pytest.raises(RuntimeError, match=match):
        fn()


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(scores, tangent, temperature):
    lcs_score, posteriors, grad_T = lcs_forward_with_grads(scores, temperature, None)
    hvp = lcs_ops.marginals_hvp(scores, tangent, temperature, None)
    dP_dT = lcs_ops.marginals_grad_temp(scores, temperature, None)
    return {
        "lcs_score": lcs_score,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_lcs_correctness(outputs, reference_outputs, thread_count):
    lcs_score_ref = reference_outputs["lcs_score"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(lcs_score_ref, outputs["lcs_score"]), \
        f"{thread_count}-thread score mismatch: max diff = {max_diff(lcs_score_ref, outputs['lcs_score'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dT_ref, outputs["dP_dT"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(dP_dT_ref, outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(8, 10), (16, 16), (5, 20)])
def seq_lengths(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestForward:

    def test_lcs_score(self, batch_size, seq_lengths, temperature, device):
        """Test that LCS scores match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        # Create match score matrix (higher for matches)
        scores = torch.rand(batch_size, L1, L2, device=device)

        score_ref, _ = lcs_forward_naive(scores, temperature)
        score_orihime = lcs_ops.forward(scores, temperature, None)[0]

        assert allclose(score_ref, score_orihime), \
            f"LCS score mismatch: max diff = {max_diff(score_ref, score_orihime)}"

    def test_lcs_score_matches_only(self, batch_size, temperature, device):
        """Test with explicit match/mismatch pattern."""
        L1, L2 = 10, 12

        torch.manual_seed(42)
        # Binary scores: 1 for match, 0 for mismatch
        scores = torch.zeros(batch_size, L1, L2, device=device)
        # Add some matches on diagonal-ish pattern
        for i in range(min(L1, L2)):
            scores[:, i, i] = 1.0

        score_ref, _ = lcs_forward_naive(scores, temperature)
        score_orihime = lcs_ops.forward(scores, temperature, None)[0]

        assert allclose(score_ref, score_orihime), \
            f"LCS score mismatch (matches only): max diff = {max_diff(score_ref, score_orihime)}"

    def test_lcs_known_pattern(self, device):
        """Test on known pattern: identical sequences should have high LCS."""
        B = 2
        L = 5
        temperature = 1.0

        # First batch: matches on diagonal (identical sequences)
        # Second batch: all zeros (different sequences)
        scores = torch.zeros(B, L, L, device=device)
        for i in range(L):
            scores[0, i, i] = 1.0

        score, _ = lcs_ops.forward(scores, temperature, None)

        # First batch (matches on diagonal) should have higher score
        assert score[0] > score[1], "Identical sequences should have higher LCS score"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device)

        posteriors_ref = lcs_naive(scores, temperature)
        posteriors_orihime = lcs_ops.forward(scores, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_orihime, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_orihime)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        posteriors = lcs_ops.forward(scores, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_orihime = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = lcs_naive(scores_ref, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestWithGrads:

    def test_with_grads_returns_temp_grad(self, device):
        """Test that lcs_with_grads returns temperature gradient."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        lcs_score, posteriors, grad_T = lcs_forward_with_grads(
            scores, temperature, None
        )

        # Check shapes
        assert lcs_score.shape == (B,)
        assert posteriors.shape == (B, L1, L2)
        assert grad_T.shape == (B,)

    def test_with_grads_temperature_matches_finite_diff(self, device):
        """The score/value temperature gradient should match finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 0.9
        eps = 1e-4

        torch.manual_seed(321)
        scores = torch.rand(B, L1, L2, device=device)
        _, _, grad_T = lcs_forward_with_grads(scores, temperature, None)

        temperature_reference_scores = scores.double()
        score_plus, _ = lcs_forward_naive(
            temperature_reference_scores, temperature + eps
        )
        score_minus, _ = lcs_forward_naive(
            temperature_reference_scores, temperature - eps
        )
        grad_T_fd = ((score_plus - score_minus) / (2 * eps)).to(scores.dtype)

        assert allclose(grad_T_fd, grad_T, rtol=1e-2, atol=2e-3), (
            "Value temperature gradient mismatch: "
            f"max diff = {max_diff(grad_T_fd, grad_T)}"
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        hvp_fd_step = HVP_FINITE_DIFFERENCE_STEP

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_orihime = lcs_ops.marginals_hvp(scores, V, temperature, None)

        posteriors_plus = lcs_naive(scores + hvp_fd_step * V, temperature)
        posteriors_minus = lcs_naive(scores - hvp_fd_step * V, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * hvp_fd_step)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"

    def test_hvp_various_temps(self, device):
        """Test HVP at various temperatures."""
        B, L1, L2 = 2, 6, 7
        hvp_fd_step = HVP_FINITE_DIFFERENCE_STEP

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores = torch.rand(B, L1, L2, device=device)
            V = torch.randn(B, L1, L2, device=device)

            hvp_orihime = lcs_ops.marginals_hvp(scores, V, temperature, None)

            posteriors_plus = lcs_naive(
                scores + hvp_fd_step * V, temperature
            )
            posteriors_minus = lcs_naive(
                scores - hvp_fd_step * V, temperature
            )
            hvp_fd = (posteriors_plus - posteriors_minus) / (2 * hvp_fd_step)

            # Slightly more relaxed tolerance for finite differences (O(eps^2) error)
            assert allclose(hvp_fd, hvp_orihime, rtol=2e-2, atol=5e-3), \
                f"HVP mismatch at T={temperature}: max diff = {max_diff(hvp_fd, hvp_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        temperature = 1.0
        temperature_fd_step = 1e-4
        hvp_fd_step = HVP_FINITE_DIFFERENCE_STEP
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.rand(B, L1, L2)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            lcs_score_ref, _ = lcs_forward_naive(scores, temperature)
            posteriors_ref = lcs_naive(scores, temperature)
            temperature_reference_scores = scores.double()

            lcs_score_plus, _ = lcs_forward_naive(
                temperature_reference_scores,
                temperature + temperature_fd_step,
            )
            lcs_score_minus, _ = lcs_forward_naive(
                temperature_reference_scores,
                temperature - temperature_fd_step,
            )
            grad_T_ref = (
                (lcs_score_plus - lcs_score_minus) / (2 * temperature_fd_step)
            ).to(scores.dtype)

            posteriors_plus = lcs_naive(
                scores + hvp_fd_step * tangent, temperature
            )
            posteriors_minus = lcs_naive(
                scores - hvp_fd_step * tangent, temperature
            )
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * hvp_fd_step)

            posteriors_temp_plus = lcs_naive(
                temperature_reference_scores,
                temperature + temperature_fd_step,
            )
            posteriors_temp_minus = lcs_naive(
                temperature_reference_scores,
                temperature - temperature_fd_step,
            )
            dP_dT_ref = (
                (posteriors_temp_plus - posteriors_temp_minus)
                / (2 * temperature_fd_step)
            ).to(scores.dtype)

        reference_outputs = {
            "lcs_score": lcs_score_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, temperature)
            assert_threaded_lcs_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, L1, L2 = 2, 8, 10
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.rand(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = lcs_ops.forward(scores_cpu, temperature, None)[1]
        posteriors_cuda = lcs_ops.forward(scores_cuda, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_various_temps(self):
        """Test CPU vs CUDA at various temperatures."""
        B, L1, L2 = 2, 10, 12

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores_cpu = torch.rand(B, L1, L2)
            scores_cuda = scores_cpu.cuda()

            posteriors_cpu = lcs_ops.forward(scores_cpu, temperature, None)[1]
            posteriors_cuda = lcs_ops.forward(scores_cuda, temperature, None)[1]

            assert allclose(posteriors_cpu, posteriors_cuda), \
                f"CPU/CUDA mismatch at T={temperature}: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_score_autograd_matches_cpu(self):
        """CUDA autograd should include the score path, not just the posterior path."""
        B, L1, L2 = 2, 5, 6
        temperature = 0.9

        torch.manual_seed(123)
        scores_cpu = torch.rand(B, L1, L2, dtype=torch.float32, requires_grad=True)
        scores_cuda = scores_cpu.detach().clone().cuda().requires_grad_(True)

        score_cpu = lcs_ops.forward(scores_cpu, temperature, None)[0]
        score_cuda = lcs_ops.forward(scores_cuda, temperature, None)[0]

        score_cpu.sum().backward()
        score_cuda.sum().backward()

        assert allclose(scores_cpu.grad, scores_cuda.grad, rtol=1e-3, atol=1e-4), \
            f"Score-path gradient mismatch: max diff = {max_diff(scores_cpu.grad, scores_cuda.grad)}"

    def test_backward_full_consistency(self):
        """Explicit backward_full should match the CPU dP/dT contraction on CUDA."""
        B, L1, L2 = 2, 6, 7
        temperature = 0.9

        torch.manual_seed(321)
        scores_cpu = torch.rand(B, L1, L2, dtype=torch.float32)
        grad_cpu = torch.randn(B, L1, L2, dtype=torch.float32)
        lengths_cpu = torch.tensor([[6, 7], [4, 5]], dtype=torch.int32)

        scores_cuda = scores_cpu.cuda()
        grad_cuda = grad_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()

        backward_cpu = lcs_ops.marginals_backward(scores_cpu, grad_cpu, temperature, lengths_cpu)
        backward_cuda = lcs_ops.marginals_backward(scores_cuda, grad_cuda, temperature, lengths_cuda)

        assert allclose(backward_cpu[0], backward_cuda[0], rtol=2e-2, atol=5e-3), \
            f"backward_full grad_scores mismatch: max diff = {max_diff(backward_cpu[0], backward_cuda[0])}"
        assert allclose(backward_cpu[1], backward_cuda[1], rtol=2e-2, atol=5e-3), \
            f"backward_full grad_T mismatch: max diff = {max_diff(backward_cpu[1], backward_cuda[1])}"

    def test_derivative_entrypoints_cpu_cuda_parity(self):
        """All LCS value/map derivative entrypoints should agree across backends."""
        B, L1, L2 = 3, 6, 7
        temperature = 0.9

        torch.manual_seed(321)
        scores_cpu = torch.rand(B, L1, L2)
        tangent_cpu = torch.randn_like(scores_cpu)
        lengths_cpu = torch.tensor(
            [[L1, L2], [4, 5], [5, 3]], dtype=torch.int32
        )

        def raw_outputs(scores, tangent, lengths):
            score, posteriors = lcs_ops.forward(
                scores, temperature, lengths
            )
            score_t, posteriors_t = lcs_ops.forward_t(
                scores,
                scores.new_tensor([temperature]),
                lengths,
            )
            grad_temp = lcs_ops.value_grad_params(
                scores, temperature, lengths
            )
            hvp = lcs_ops.marginals_hvp(
                scores, tangent, temperature, lengths
            )
            dP_dT = lcs_ops.marginals_grad_temp(
                scores, temperature, lengths
            )
            full = lcs_ops.marginals_backward(
                scores, tangent, temperature, lengths
            )
            raw_vjp = orihime.raw.lcs.vjp_one(
                scores,
                wrt="temperature",
                cotangent=tangent,
                temperature=temperature,
                lengths=lengths,
            )
            return (
                score,
                posteriors,
                score_t,
                posteriors_t,
                grad_temp,
                hvp,
                dP_dT,
                full[0],
                full[1],
                raw_vjp,
            )

        def public_autograd(scores, lengths):
            map_input = scores.detach().clone().requires_grad_(True)
            map_temperature = map_input.new_tensor(
                [temperature], requires_grad=True
            )
            map_result = orihime.lcs(
                map_input,
                temperature=map_temperature,
                lengths=lengths,
            )
            map_result.square().sum().backward()
            map_grad = map_input.grad.detach().clone()
            map_temp_grad = map_temperature.grad.detach().clone()

            value_input = scores.detach().clone().requires_grad_(True)
            value_temperature = value_input.new_tensor(
                [temperature], requires_grad=True
            )
            value_result = orihime.lcs_value(
                value_input,
                temperature=value_temperature,
                lengths=lengths,
            )
            value_result.sum().backward()
            return (
                map_grad,
                map_temp_grad,
                value_input.grad.detach().clone(),
                value_temperature.grad.detach().clone(),
            )

        scores_cuda = scores_cpu.cuda()
        tangent_cuda = tangent_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()
        cpu_outputs = raw_outputs(scores_cpu, tangent_cpu, lengths_cpu)
        cuda_outputs = raw_outputs(scores_cuda, tangent_cuda, lengths_cuda)

        for index, (cpu, cuda) in enumerate(zip(cpu_outputs, cuda_outputs)):
            tolerance = (1e-2, 5e-3) if index >= 4 else (1e-3, 1e-4)
            assert allclose(
                cpu,
                cuda,
                rtol=tolerance[0],
                atol=tolerance[1],
            ), f"LCS CPU/CUDA derivative mismatch at output {index}"

        cpu_autograd = public_autograd(scores_cpu, lengths_cpu)
        cuda_autograd = public_autograd(scores_cuda, lengths_cuda)
        for index, (cpu, cuda) in enumerate(zip(cpu_autograd, cuda_autograd)):
            assert allclose(
                cpu,
                cuda,
                rtol=1e-2,
                atol=5e-3,
            ), f"LCS CPU/CUDA autograd mismatch at output {index}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, max_L1, max_L2, device=device)

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        lcs_score, posteriors = lcs_ops.forward(scores, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            score_ref, _ = lcs_forward_naive(scores_b, temperature)
            posteriors_ref = lcs_naive(scores_b, temperature)

            # Score should match for this sequence
            assert allclose(score_ref, lcs_score[b:b+1]), \
                f"LCS score mismatch for batch {b}: {score_ref.item()} vs {lcs_score[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"

    def test_padded_regions_stay_zero_for_all_derivative_paths(self, device):
        """Every LCS map/derivative field must preserve zero padded regions."""
        B, max_L1, max_L2 = 3, 6, 7
        temperature = 0.9

        torch.manual_seed(123)
        scores = torch.rand(B, max_L1, max_L2, device=device)
        tangent = torch.randn_like(scores)
        lengths = torch.tensor(
            [[max_L1, max_L2], [4, 5], [5, 3]],
            dtype=torch.int32,
            device=device,
        )

        _, posteriors = lcs_ops.forward(scores, temperature, lengths)
        with_grads = lcs_forward_with_grads(
            scores, temperature, lengths
        )
        hvp = lcs_ops.marginals_hvp(
            scores, tangent, temperature, lengths
        )
        dP_dT = lcs_ops.marginals_grad_temp(
            scores, temperature, lengths
        )
        full = lcs_ops.marginals_backward(
            scores, tangent, temperature, lengths
        )

        assert_padded_pairwise_region_zero("forward posteriors", posteriors, lengths)
        assert_padded_pairwise_region_zero(
            "with_grads posteriors", with_grads[1], lengths
        )
        assert_padded_pairwise_region_zero("HVP", hvp, lengths)
        assert_padded_pairwise_region_zero("temperature Jacobian", dP_dT, lengths)
        assert_padded_pairwise_region_zero(
            "full VJP scores gradient", full[0], lengths
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 score matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        temperature = 1.0

        lcs_score, posteriors = lcs_ops.forward(scores, temperature, None)
        score_ref, _ = lcs_forward_naive(scores, temperature)
        posteriors_ref = lcs_naive(scores, temperature)

        assert allclose(lcs_score, score_ref), f"Single element score wrong: {lcs_score.item()} vs {score_ref.item()}"
        assert allclose(posteriors, posteriors_ref), "Single element posterior wrong"

    def test_row_vector(self, device):
        """Test 1xN score matrix."""
        scores = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        temperature = 1.0

        lcs_score, posteriors = lcs_ops.forward(scores, temperature, None)
        score_ref, _ = lcs_forward_naive(scores, temperature)

        assert allclose(lcs_score, score_ref), "Row vector score mismatch"

    def test_col_vector(self, device):
        """Test Nx1 score matrix."""
        scores = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        temperature = 1.0

        lcs_score, posteriors = lcs_ops.forward(scores, temperature, None)
        score_ref, _ = lcs_forward_naive(scores, temperature)

        assert allclose(lcs_score, score_ref), "Column vector score mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard LCS)."""
        B, L1, L2 = 2, 6, 8
        temperature = 0.01

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        posteriors = lcs_ops.forward(scores, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        temperature = 10.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        posteriors = lcs_ops.forward(scores, temperature, None)[1]
        posteriors_ref = lcs_naive(scores, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_zero_temperature_clamp(self, device):
        """Test that very low temperature doesn't cause NaN."""
        B, L1, L2 = 2, 5, 6
        temperature = 1e-6

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        lcs_score, posteriors = lcs_ops.forward(scores, temperature, None)

        assert not torch.isnan(lcs_score).any(), "LCS score contains NaN"
        assert not torch.isnan(posteriors).any(), "Posteriors contain NaN"
        assert not torch.isinf(lcs_score).any(), "LCS score contains Inf"

    def test_all_matches(self, device):
        """Test with all matches (identical sequences on diagonal)."""
        B, L = 2, 5
        temperature = 1.0

        # High match scores on diagonal
        scores = torch.zeros(B, L, L, device=device)
        for i in range(L):
            scores[:, i, i] = 1.0

        lcs_score, posteriors = lcs_ops.forward(scores, temperature, None)

        # With matches on diagonal, LCS should be close to L for low temperature
        # At T=1.0, it will be somewhat higher due to softmax aggregation
        assert lcs_score.mean() > 0, "LCS score should be positive"

    def test_invalid_lengths_rejected(self, device):
        """Lengths must stay within the padded score tensor bounds."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0

        scores = torch.rand(B, L1, L2, device=device)
        bad_lengths = torch.tensor([[L1 + 1, L2], [L1, L2]], dtype=torch.int32, device=device)

        require_raises(
            lambda: lcs_ops.forward(scores, temperature, bad_lengths),
            rf"lengths\[0,0\] must be between 0 and {L1}",
        )

    def test_hvp_rejects_bad_tangent_shape(self, device):
        """HVP tangents must match the score tensor shape."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0

        scores = torch.rand(B, L1, L2, device=device)
        bad_tangent = torch.rand(B, L1 - 1, L2, device=device)

        require_raises(
            lambda: lcs_ops.marginals_hvp(scores, bad_tangent, temperature, None),
            "tangent must have same shape as scores",
        )

    def test_backward_full_rejects_bad_grad_shape(self, device):
        """Explicit backward inputs must match the posterior tensor shape."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0

        scores = torch.rand(B, L1, L2, device=device)
        bad_grad = torch.rand(B, L1 - 1, L2, device=device)

        require_raises(
            lambda: lcs_ops.marginals_backward(scores, bad_grad, temperature, None),
            "cotangent must have same shape as scores",
        )

    def test_noncontiguous_inputs_have_explicit_contract(self):
        """Raw LCS entrypoints state their contiguous-input behavior."""
        scores = torch.rand(2, 4, 5)
        noncontiguous_scores = torch.rand(2, 4, 5, 2)[..., 0]
        tangent = torch.randn_like(scores)
        noncontiguous_tangent = torch.randn(2, 4, 5, 2)[..., 0]
        noncontiguous_cotangent = torch.randn(2, 4, 5, 2)[..., 0]
        noncontiguous_lengths = torch.tensor(
            [[4, 5], [3, 4]], dtype=torch.int32
        ).t()

        assert not noncontiguous_scores.is_contiguous()
        assert not noncontiguous_tangent.is_contiguous()
        assert not noncontiguous_cotangent.is_contiguous()
        assert not noncontiguous_lengths.is_contiguous()

        with pytest.raises(RuntimeError, match="scores must be contiguous"):
            lcs_ops.forward(noncontiguous_scores, 1.0, None)
        with pytest.raises(RuntimeError, match="tangent must be contiguous"):
            lcs_ops.marginals_hvp(scores, noncontiguous_tangent, 1.0, None)
        with pytest.raises(RuntimeError, match="lengths must be contiguous"):
            lcs_ops.forward(scores, 1.0, noncontiguous_lengths)
        with pytest.raises(RuntimeError, match="cotangent must be contiguous"):
            lcs_ops.marginals_backward(
                scores, noncontiguous_cotangent, 1.0, None
            )

        with pytest.raises(ValueError, match="cotangent.*contiguous"):
            orihime.raw.lcs.vjp_one(
                scores,
                wrt="temperature",
                cotangent=noncontiguous_cotangent,
                temperature=1.0,
            )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_public_autograd_backward_normalizes_noncontiguous_upstream_gradient():
    torch.manual_seed(20260811)
    scores = torch.rand(2, 4, 5)
    upstream = torch.randn(2, 4, 5, 2)[..., 0]
    assert not upstream.is_contiguous()

    scores_with_noncontiguous_upstream = scores.clone().requires_grad_(True)
    map_with_noncontiguous_upstream = orihime.lcs(scores_with_noncontiguous_upstream)
    map_with_noncontiguous_upstream.backward(upstream)

    scores_with_contiguous_upstream = scores.clone().requires_grad_(True)
    map_with_contiguous_upstream = orihime.lcs(scores_with_contiguous_upstream)
    map_with_contiguous_upstream.backward(upstream.contiguous())

    torch.testing.assert_close(
        map_with_noncontiguous_upstream,
        map_with_contiguous_upstream,
    )
    assert scores_with_noncontiguous_upstream.grad is not None
    assert scores_with_contiguous_upstream.grad is not None
    assert torch.isfinite(scores_with_noncontiguous_upstream.grad).all()
    torch.testing.assert_close(
        scores_with_noncontiguous_upstream.grad,
        scores_with_contiguous_upstream.grad,
    )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestParamJacobian:

    def test_param_jacobian_temperature(self, device):
        """Test parameter Jacobian for temperature."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        # dP/dT via param_jacobian
        dP_dT = lcs_ops.marginals_grad_temp(scores, temperature, None)

        # Finite diff
        posteriors_plus = lcs_naive(scores, temperature + eps)
        posteriors_minus = lcs_naive(scores, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    def test_param_jacobian_various_temps(self, device):
        """Test parameter Jacobian at various temperatures."""
        B, L1, L2 = 2, 5, 6
        eps = 1e-4

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores = torch.rand(B, L1, L2, device=device)

            dP_dT = lcs_ops.marginals_grad_temp(scores, temperature, None)

            posteriors_plus = lcs_naive(scores, temperature + eps)
            posteriors_minus = lcs_naive(scores, temperature - eps)
            dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
                f"Param Jacobian (T={temperature}) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"


# --- Memory-safety regression tests (merged from test_lcs_{cpp,cuda}_memsafety.py) ---

OVERSIZED_LCS_SHAPE = (0, 65535, 65535)
OVERSIZED_ERROR = r"LCS workspace too large"


def _oversized_inputs():
    # A zero-batch tensor preserves the dangerous L1 x L2 shape metadata while
    # containing no elements or backing storage. Empty tensors are contiguous,
    # so this reaches the native workspace-size guard without relying on a
    # runner accepting a ~16 GiB allocation or virtual-memory mapping.
    scores = torch.empty(OVERSIZED_LCS_SHAPE, dtype=torch.float32)
    assert scores.is_contiguous()
    assert scores.untyped_storage().nbytes() == 0
    lengths = torch.empty((0, 2), dtype=torch.int32)
    temperature = torch.empty((1,), dtype=torch.float32)
    return scores, scores, lengths, temperature


def _call_lcs(scores, same_shape, lengths, temperature):
    del same_shape
    return lcs_ops.forward_t(scores, temperature, lengths)


def _call_lcs_float(scores, same_shape, lengths, temperature):
    del same_shape, lengths, temperature
    return lcs_ops.forward(scores, 1.0, None)


def _call_lcs_with_grads(scores, same_shape, lengths, temperature):
    del same_shape, lengths, temperature
    return lcs_forward_with_grads(scores, 1.0, None)


def _call_lcs_hvp(scores, same_shape, lengths, temperature):
    del lengths, temperature
    return lcs_ops.marginals_hvp(scores, same_shape, 1.0, None)


def _call_lcs_param_jacobian(scores, same_shape, lengths, temperature):
    del same_shape, lengths, temperature
    return lcs_ops.marginals_grad_temp(scores, 1.0, None)


def _call_lcs_backward_full(scores, same_shape, lengths, temperature):
    del lengths, temperature
    return lcs_ops.marginals_backward(scores, same_shape, 1.0, None)


OVERSIZED_WORKSPACE_CALLS = [
    ("lcs", _call_lcs),
    ("lcs_float", _call_lcs_float),
    ("lcs_with_grads", _call_lcs_with_grads),
    ("lcs_hvp", _call_lcs_hvp),
    ("lcs_param_jacobian", _call_lcs_param_jacobian),
    ("lcs_backward_full", _call_lcs_backward_full),
]


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.parametrize(
    ("name", "call"),
    OVERSIZED_WORKSPACE_CALLS,
    ids=[name for name, _ in OVERSIZED_WORKSPACE_CALLS],
)
def test_oversized_lcs_cpu_workspace_rejected_before_allocation(name, call):
    del name
    scores, same_shape, lengths, temperature = _oversized_inputs()
    with pytest.raises(RuntimeError, match=OVERSIZED_ERROR):
        call(scores, same_shape, lengths, temperature)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_valid_lcs_cpu_outputs_unchanged():
    scores = torch.tensor(
        [
            [
                [0.7, 0.1, 0.2],
                [0.2, 0.9, 0.3],
                [0.4, 0.3, 0.8],
            ]
        ],
        dtype=torch.float32,
    )
    tangent = torch.randn_like(scores)
    temperature = 1.0

    score, posteriors = lcs_ops.forward(scores, temperature, None)
    score_dbg, posteriors_dbg, grad_T = lcs_forward_with_grads(scores, temperature, None)
    hvp = lcs_ops.marginals_hvp(scores, tangent, temperature, None)
    dP_dT = lcs_ops.marginals_grad_temp(scores, temperature, None)
    grad_scores, grad_temperature = lcs_ops.marginals_backward(scores, tangent, temperature, None)

    torch.testing.assert_close(score, score_dbg)
    torch.testing.assert_close(posteriors, posteriors_dbg)
    assert grad_T.shape == (1,)
    assert hvp.shape == scores.shape
    assert dP_dT.shape == scores.shape
    assert grad_scores.shape == scores.shape
    assert grad_temperature.shape == (1,)


def _oversized_logical_scores():
    """A huge logical LCS table with one-float storage.

    The zero strides avoid allocating the 16+ GiB score matrix, while the wrapper
    still sees the dangerous L=65535 dimensions from I-18.
    """
    base = torch.empty((1,), device="cuda", dtype=torch.float32)
    return base.as_strided((1, 65535, 65535), (0, 0, 0))


def _assert_rejects_oversized_dp_table(call):
    scores = _oversized_logical_scores()
    with pytest.raises(RuntimeError, match="LCS CUDA DP table is too large"):
        call(scores)


@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA memory-safety regressions need the CUDA orihime backend",
)
@pytest.mark.parametrize(
    "call",
    [
        lambda scores: lcs_ops.forward_t(scores, torch.tensor(1.0, device=scores.device), None),
        lambda scores: lcs_ops.forward(scores, 1.0, None),
        lambda scores: lcs_forward_with_grads(scores, 1.0, None),
        lambda scores: lcs_ops.marginals_hvp(scores, scores, 1.0, None),
        lambda scores: lcs_ops.marginals_grad_temp(scores, 1.0, None),
        lambda scores: lcs_ops.marginals_backward(scores, scores, 1.0, None),
    ],
)
def test_lcs_cuda_rejects_dp_table_int32_overflow_before_allocation(call):
    _assert_rejects_oversized_dp_table(call)


def _backward_peak_bytes(loss_kind, size=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    scores = torch.randn(1, size, size, device="cuda", requires_grad=True)
    lcs_score, posteriors = lcs_ops.forward(scores, 1.0, None)
    if loss_kind == "score":
        loss = lcs_score.sum()
    else:
        loss = posteriors.square().sum()
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA memory-safety regressions need the CUDA orihime backend",
)
def test_lcs_cuda_score_only_backward_skips_materialized_hvp_path():
    size = 512
    score_only_peak = _backward_peak_bytes("score", size=size)
    posterior_peak = _backward_peak_bytes("posterior", size=size)

    alpha_bytes = (size + 1) * (size + 1) * 4
    assert posterior_peak - score_only_peak > 2 * alpha_bytes


@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA memory-safety regressions need the CUDA orihime backend",
)
def test_lcs_cuda_valid_small_input_still_matches_cpu():
    torch.manual_seed(20260707)
    scores_cpu = torch.rand(2, 8, 9)
    scores_cuda = scores_cpu.cuda()
    lengths_cpu = torch.tensor([[8, 9], [6, 7]], dtype=torch.int32)
    lengths_cuda = lengths_cpu.cuda()

    score_cpu, post_cpu = lcs_ops.forward(scores_cpu, 1.0, lengths_cpu)
    score_cuda, post_cuda = lcs_ops.forward(scores_cuda, 1.0, lengths_cuda)

    assert torch.allclose(score_cpu, score_cuda.cpu(), rtol=1e-4, atol=1e-5)
    assert torch.allclose(post_cpu, post_cuda.cpu(), rtol=1e-3, atol=1e-4)


@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA device guards require a CUDA-enabled orihime build",
)
@TWO_CUDA_DEVICES_REQUIRED
def test_lcs_entrypoints_guard_current_device_for_all_paths():
    target = torch.device("cuda:0")
    original_device = torch.cuda.current_device()

    torch.manual_seed(123)
    scores = torch.rand(2, 4, 5, device=target)
    lengths = torch.tensor(
        [[4, 5], [3, 4]], dtype=torch.int32, device=target
    )
    tangent = torch.randn_like(scores)
    temperature = scores.new_tensor([0.7])

    try:
        torch.cuda.set_device(1)
        assert target.index != torch.cuda.current_device()
        results = (
            lcs_ops.forward_t(scores, temperature, lengths),
            lcs_ops.forward(scores, 0.7, lengths),
            lcs_forward_with_grads(scores, 0.7, lengths),
            lcs_ops.value_grad_params(scores, 0.7, lengths),
            lcs_ops.marginals_hvp(scores, tangent, 0.7, lengths),
            lcs_ops.marginals_grad_temp(scores, 0.7, lengths),
            lcs_ops.marginals_backward(scores, tangent, 0.7, lengths),
            orihime.lcs(
                scores,
                temperature=temperature,
                lengths=lengths,
            ),
            orihime.lcs_value(
                scores,
                temperature=temperature,
                lengths=lengths,
            ),
            orihime.raw.lcs.vjp_one(
                scores,
                wrt="temperature",
                cotangent=tangent,
                temperature=temperature,
                lengths=lengths,
            ),
        )
        for result in results:
            assert_tensors_on_device(result, target)
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA device guards require a CUDA-enabled orihime build",
)
@TWO_CUDA_DEVICES_REQUIRED
def test_lcs_entrypoints_reject_wrong_device_secondary_tensors():
    scores = torch.rand((2, 4, 5), device="cuda:0")
    lengths = torch.tensor(
        [[4, 5], [3, 4]], dtype=torch.int32, device="cuda:0"
    )
    wrong_lengths = lengths.to("cuda:1")
    tangent = torch.randn_like(scores)
    wrong_tangent = tangent.to("cuda:1")
    cotangent = torch.randn_like(scores)
    wrong_cotangent = cotangent.to("cuda:1")
    temperature = scores.new_tensor([0.7])
    wrong_temperature = torch.tensor([0.7], device="cuda:1")
    assert scores.device.index != wrong_lengths.device.index
    assert scores.device.index != wrong_tangent.device.index
    assert scores.device.index != wrong_cotangent.device.index
    assert scores.device.index != wrong_temperature.device.index

    length_calls = (
        lambda: lcs_ops.forward_t(scores, temperature, wrong_lengths),
        lambda: lcs_ops.forward(scores, 0.7, wrong_lengths),
        lambda: lcs_forward_with_grads(scores, 0.7, wrong_lengths),
        lambda: lcs_ops.value_grad_params(scores, 0.7, wrong_lengths),
        lambda: lcs_ops.marginals_hvp(
            scores, tangent, 0.7, wrong_lengths
        ),
        lambda: lcs_ops.marginals_grad_temp(
            scores, 0.7, wrong_lengths
        ),
        lambda: lcs_ops.marginals_backward(
            scores, cotangent, 0.7, wrong_lengths
        ),
    )
    for call in length_calls:
        with pytest.raises(RuntimeError, match="lengths must be on same device"):
            call()

    with pytest.raises(ValueError, match="lengths.*same device"):
        orihime.lcs(scores, temperature=0.7, lengths=wrong_lengths)
    with pytest.raises(ValueError, match="lengths.*same device"):
        orihime.lcs_value(scores, temperature=0.7, lengths=wrong_lengths)
    with pytest.raises(ValueError, match="lengths.*same device"):
        orihime.raw.lcs.vjp_one(
            scores,
            wrt="temperature",
            cotangent=cotangent,
            temperature=0.7,
            lengths=wrong_lengths,
        )
    with pytest.raises(ValueError, match="temperature.*same device"):
        orihime.lcs(
            scores,
            temperature=wrong_temperature,
            lengths=lengths,
        )

    with pytest.raises(RuntimeError, match="tangent must be on same device"):
        lcs_ops.marginals_hvp(scores, wrong_tangent, 0.7, lengths)

    with pytest.raises(RuntimeError, match="cotangent must be on same device"):
        lcs_ops.marginals_backward(
            scores, wrong_cotangent, 0.7, lengths
        )
    with pytest.raises(ValueError, match="cotangent.*same device"):
        orihime.raw.lcs.vjp_one(
            scores,
            wrt="temperature",
            cotangent=wrong_cotangent,
            temperature=0.7,
            lengths=lengths,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
