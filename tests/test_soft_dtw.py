# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft DTW (Dynamic Time Warping).
"""

import contextlib

import pytest
import torch

from reference import dtw_forward_naive, dtw_naive
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

try:
    import orihime
    dtw_ops = orihime.ops._kernels["dtw"]
    from operator_test_utils import dtw_forward_with_grads
    ORIHIME_AVAILABLE = True
except ImportError:
    ORIHIME_AVAILABLE = False

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


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(costs, tangent, temperature, bandwidth):
    score, posteriors, grad_T = dtw_forward_with_grads(costs, temperature, None, bandwidth)
    hvp = dtw_ops.marginals_hvp(costs, tangent, temperature, None, bandwidth)
    dP_dT = dtw_ops.marginals_grad_temp(costs, temperature, None, bandwidth)
    return {
        "score": score,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_dtw_correctness(outputs, reference_outputs, thread_count):
    score_ref = reference_outputs["score"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(score_ref, outputs["score"]), \
        f"{thread_count}-thread score mismatch: max diff = {max_diff(score_ref, outputs['score'])}"
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

    def test_score(self, batch_size, seq_lengths, temperature, device):
        """Test that DTW scores match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        costs = torch.randn(batch_size, L1, L2, device=device).abs()  # Use positive costs

        score_ref, _ = dtw_forward_naive(costs, temperature)
        score_orihime = dtw_ops.forward(costs, temperature, None, -1)[0]

        assert allclose(score_ref, score_orihime), \
            f"Score mismatch: max diff = {max_diff(score_ref, score_orihime)}"

    def test_score_with_bandwidth(self, batch_size, temperature, device):
        """Test DTW scores with Sakoe-Chiba band."""
        L1, L2 = 12, 15
        bandwidth = 3

        torch.manual_seed(42)
        costs = torch.randn(batch_size, L1, L2, device=device).abs()

        score_ref, _ = dtw_forward_naive(costs, temperature, bandwidth=bandwidth)
        score_orihime = dtw_ops.forward(costs, temperature, None, bandwidth)[0]

        assert allclose(score_ref, score_orihime), \
            f"Score mismatch with bandwidth: max diff = {max_diff(score_ref, score_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        costs = torch.randn(batch_size, L1, L2, device=device).abs()

        posteriors_ref = dtw_naive(costs, temperature)
        posteriors_orihime = dtw_ops.forward(costs, temperature, None, -1)[1]

        assert allclose(posteriors_ref, posteriors_orihime, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_orihime)}"

    def test_posteriors_with_bandwidth(self, batch_size, temperature, device):
        """Test posteriors with Sakoe-Chiba band."""
        L1, L2 = 12, 15
        bandwidth = 3

        torch.manual_seed(42)
        costs = torch.randn(batch_size, L1, L2, device=device).abs()

        posteriors_ref = dtw_naive(costs, temperature, bandwidth=bandwidth)
        posteriors_orihime = dtw_ops.forward(costs, temperature, None, bandwidth)[1]

        assert allclose(posteriors_ref, posteriors_orihime, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch with bandwidth: max diff = {max_diff(posteriors_ref, posteriors_orihime)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8

        torch.manual_seed(42)
        # Create tensor, apply abs, then set requires_grad to get a leaf tensor
        costs = torch.randn(batch_size, L1, L2, device=device).abs()
        costs.requires_grad_(True)

        posteriors = dtw_ops.forward(costs, temperature, None, -1)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_orihime = costs.grad.clone()

        costs_ref = costs.detach().clone().requires_grad_(True)
        posteriors_ref = dtw_naive(costs_ref, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = costs_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        hvp_fd_step = HVP_FINITE_DIFFERENCE_STEP

        torch.manual_seed(42)
        costs = torch.randn(B, L1, L2, device=device).abs()
        V = torch.randn(B, L1, L2, device=device)

        hvp_orihime = dtw_ops.marginals_hvp(costs, V, temperature, None, -1)

        posteriors_plus = dtw_naive(costs + hvp_fd_step * V, temperature)
        posteriors_minus = dtw_naive(costs - hvp_fd_step * V, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * hvp_fd_step)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"

    def test_hvp_with_bandwidth(self, device):
        """Test HVP with Sakoe-Chiba band."""
        B, L1, L2 = 2, 8, 10
        temperature = 1.0
        bandwidth = 3
        hvp_fd_step = HVP_FINITE_DIFFERENCE_STEP

        torch.manual_seed(42)
        costs = torch.randn(B, L1, L2, device=device).abs()
        V = torch.randn(B, L1, L2, device=device)

        hvp_orihime = dtw_ops.marginals_hvp(costs, V, temperature, None, bandwidth)

        posteriors_plus = dtw_naive(
            costs + hvp_fd_step * V, temperature, bandwidth=bandwidth
        )
        posteriors_minus = dtw_naive(
            costs - hvp_fd_step * V, temperature, bandwidth=bandwidth
        )
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * hvp_fd_step)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch with bandwidth: max diff = {max_diff(hvp_fd, hvp_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestParamJacobian:

    def test_value_grad_temperature_cpu(self):
        """The cost/value temperature gradient should match finite differences."""
        B, L1, L2 = 2, 6, 7
        temperature = 0.9
        bandwidth = 2
        eps = 1e-4

        torch.manual_seed(321)
        costs = torch.randn(B, L1, L2).abs()
        grad_temp = dtw_ops.value_grad_params(
            costs, temperature, None, bandwidth
        )

        temperature_reference_costs = costs.double()
        score_plus, _ = dtw_forward_naive(
            temperature_reference_costs, temperature + eps, bandwidth=bandwidth
        )
        score_minus, _ = dtw_forward_naive(
            temperature_reference_costs, temperature - eps, bandwidth=bandwidth
        )
        grad_temp_fd = ((score_plus - score_minus) / (2 * eps)).to(costs.dtype)

        assert allclose(grad_temp_fd, grad_temp, rtol=1e-2, atol=2e-3), (
            "Value temperature gradient mismatch: "
            f"max diff = {max_diff(grad_temp_fd, grad_temp)}"
        )

    def test_param_jacobian_temperature_cpu(self):
        """CPU dP/dT should match a finite-difference reference."""
        B, L1, L2 = 2, 6, 7
        temperature = 1.0
        bandwidth = 2
        eps = 1e-4

        torch.manual_seed(123)
        costs = torch.randn(B, L1, L2).abs()

        with torch_num_threads(1):
            dP_dT = dtw_ops.marginals_grad_temp(costs, temperature, None, bandwidth)
            posteriors_plus = dtw_naive(costs, temperature + eps, bandwidth=bandwidth)
            posteriors_minus = dtw_naive(costs, temperature - eps, bandwidth=bandwidth)
            dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_cpu_parity(self):
        """CUDA dP/dT should match CPU dP/dT (r27 regression)."""
        B, L1, L2 = 2, 6, 7
        temperature = 1.0
        bandwidth = 2

        torch.manual_seed(123)
        costs_cpu = torch.randn(B, L1, L2).abs()
        costs_cuda = costs_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = dtw_ops.marginals_grad_temp(costs_cpu, temperature, None, bandwidth)
        dP_dT_cuda = dtw_ops.marginals_grad_temp(costs_cuda, temperature, None, bandwidth)

        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU param Jacobian mismatch: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_finite_diff(self):
        """CUDA dP/dT should match a finite-difference reference (r27 regression)."""
        B, L1, L2 = 2, 6, 7
        temperature = 1.0
        bandwidth = 2
        eps = 1e-4

        torch.manual_seed(123)
        costs = torch.randn(B, L1, L2).abs().cuda()

        dP_dT = dtw_ops.marginals_grad_temp(costs, temperature, None, bandwidth)
        posteriors_plus = dtw_naive(costs, temperature + eps, bandwidth=bandwidth)
        posteriors_minus = dtw_naive(costs, temperature - eps, bandwidth=bandwidth)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"CUDA param Jacobian vs finite diff mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_no_bandwidth(self):
        """CUDA dP/dT without bandwidth should match CPU (r27 regression)."""
        B, L1, L2 = 2, 8, 10
        temperature = 1.0

        torch.manual_seed(42)
        costs_cpu = torch.randn(B, L1, L2).abs()
        costs_cuda = costs_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = dtw_ops.marginals_grad_temp(costs_cpu, temperature, None, -1)
        dP_dT_cuda = dtw_ops.marginals_grad_temp(costs_cuda, temperature, None, -1)

        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU param Jacobian (no bw) mismatch: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        temperature = 1.0
        bandwidth = 2
        temperature_fd_step = 1e-4
        hvp_fd_step = HVP_FINITE_DIFFERENCE_STEP
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        costs = torch.randn(B, L1, L2).abs()
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            score_ref, _ = dtw_forward_naive(costs, temperature, bandwidth=bandwidth)
            posteriors_ref = dtw_naive(costs, temperature, bandwidth=bandwidth)
            temperature_reference_costs = costs.double()

            score_temp_plus, _ = dtw_forward_naive(
                temperature_reference_costs,
                temperature + temperature_fd_step,
                bandwidth=bandwidth,
            )
            score_temp_minus, _ = dtw_forward_naive(
                temperature_reference_costs,
                temperature - temperature_fd_step,
                bandwidth=bandwidth,
            )
            grad_T_ref = (
                (score_temp_plus - score_temp_minus) / (2 * temperature_fd_step)
            ).to(costs.dtype)

            posteriors_plus = dtw_naive(
                costs + hvp_fd_step * tangent, temperature, bandwidth=bandwidth
            )
            posteriors_minus = dtw_naive(
                costs - hvp_fd_step * tangent, temperature, bandwidth=bandwidth
            )
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * hvp_fd_step)

            posteriors_temp_plus = dtw_naive(
                temperature_reference_costs,
                temperature + temperature_fd_step,
                bandwidth=bandwidth,
            )
            posteriors_temp_minus = dtw_naive(
                temperature_reference_costs,
                temperature - temperature_fd_step,
                bandwidth=bandwidth,
            )
            dP_dT_ref = (
                (posteriors_temp_plus - posteriors_temp_minus)
                / (2 * temperature_fd_step)
            ).to(costs.dtype)

        reference_outputs = {
            "score": score_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(costs, tangent, temperature, bandwidth)
            assert_threaded_dtw_correctness(outputs, reference_outputs, thread_count)
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
        costs_cpu = torch.randn(B, L1, L2).abs()
        costs_cuda = costs_cpu.cuda()

        posteriors_cpu = dtw_ops.forward(costs_cpu, temperature, None, -1)[1]
        posteriors_cuda = dtw_ops.forward(costs_cuda, temperature, None, -1)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_with_bandwidth(self):
        """Test CPU vs CUDA with Sakoe-Chiba band."""
        B, L1, L2 = 2, 10, 12
        temperature = 1.0
        bandwidth = 4

        torch.manual_seed(42)
        costs_cpu = torch.randn(B, L1, L2).abs()
        costs_cuda = costs_cpu.cuda()

        posteriors_cpu = dtw_ops.forward(costs_cpu, temperature, None, bandwidth)[1]
        posteriors_cuda = dtw_ops.forward(costs_cuda, temperature, None, bandwidth)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch with bandwidth: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_derivative_entrypoints_cpu_cuda_parity(self):
        """All DTW value/map derivative entrypoints should agree across backends."""
        B, L1, L2 = 3, 6, 7
        temperature = 0.9
        bandwidth = 2

        torch.manual_seed(321)
        costs_cpu = torch.randn(B, L1, L2).abs()
        tangent_cpu = torch.randn_like(costs_cpu)
        lengths_cpu = torch.tensor(
            [[L1, L2], [4, 5], [5, 3]], dtype=torch.int32
        )

        def raw_outputs(costs, tangent, lengths):
            score, posteriors = dtw_ops.forward(
                costs, temperature, lengths, bandwidth
            )
            score_t, posteriors_t = dtw_ops.forward_t(
                costs,
                costs.new_tensor([temperature]),
                lengths,
                bandwidth,
            )
            grad_temp = dtw_ops.value_grad_params(
                costs, temperature, lengths, bandwidth
            )
            hvp = dtw_ops.marginals_hvp(
                costs, tangent, temperature, lengths, bandwidth
            )
            dP_dT = dtw_ops.marginals_grad_temp(
                costs, temperature, lengths, bandwidth
            )
            full = dtw_ops.marginals_backward(
                costs, tangent, temperature, lengths, bandwidth
            )
            raw_vjp = orihime.raw.dtw.vjp_one(
                costs,
                wrt="temperature",
                cotangent=tangent,
                temperature=temperature,
                lengths=lengths,
                bandwidth=bandwidth,
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

        def public_autograd(costs, lengths):
            map_input = costs.detach().clone().requires_grad_(True)
            map_temperature = map_input.new_tensor(
                [temperature], requires_grad=True
            )
            map_result = orihime.dtw(
                map_input,
                temperature=map_temperature,
                lengths=lengths,
                bandwidth=bandwidth,
            )
            map_result.square().sum().backward()
            map_grad = map_input.grad.detach().clone()
            map_temp_grad = map_temperature.grad.detach().clone()

            value_input = costs.detach().clone().requires_grad_(True)
            value_temperature = value_input.new_tensor(
                [temperature], requires_grad=True
            )
            value_result = orihime.dtw_value(
                value_input,
                temperature=value_temperature,
                lengths=lengths,
                bandwidth=bandwidth,
            )
            value_result.sum().backward()
            return (
                map_grad,
                map_temp_grad,
                value_input.grad.detach().clone(),
                value_temperature.grad.detach().clone(),
            )

        costs_cuda = costs_cpu.cuda()
        tangent_cuda = tangent_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()
        cpu_outputs = raw_outputs(costs_cpu, tangent_cpu, lengths_cpu)
        cuda_outputs = raw_outputs(costs_cuda, tangent_cuda, lengths_cuda)

        for index, (cpu, cuda) in enumerate(zip(cpu_outputs, cuda_outputs)):
            tolerance = (1e-2, 5e-3) if index >= 4 else (1e-3, 1e-4)
            assert allclose(
                cpu,
                cuda,
                rtol=tolerance[0],
                atol=tolerance[1],
            ), f"DTW CPU/CUDA derivative mismatch at output {index}"

        cpu_autograd = public_autograd(costs_cpu, lengths_cpu)
        cuda_autograd = public_autograd(costs_cuda, lengths_cuda)
        for index, (cpu, cuda) in enumerate(zip(cpu_autograd, cuda_autograd)):
            assert allclose(
                cpu,
                cuda,
                rtol=1e-2,
                atol=5e-3,
            ), f"DTW CPU/CUDA autograd mismatch at output {index}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestValidation:

    @pytest.mark.parametrize(
        "device_type",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
    )
    def test_all_entrypoints_reject_out_of_bounds_lengths(self, device_type):
        device = torch.device(device_type)
        B, max_L1, max_L2 = 2, 4, 5
        temperature = 0.8

        torch.manual_seed(7)
        costs = torch.randn(B, max_L1, max_L2, device=device).abs()
        tangent = torch.randn_like(costs)
        bad_lengths = torch.tensor([[max_L1 + 1, max_L2], [max_L1, max_L2]], device=device, dtype=torch.int32)

        entrypoints = (
            lambda: dtw_ops.forward(costs, temperature, bad_lengths, -1),
            lambda: dtw_forward_with_grads(costs, temperature, bad_lengths, -1),
            lambda: dtw_ops.marginals_hvp(costs, tangent, temperature, bad_lengths, -1),
            lambda: dtw_ops.marginals_grad_temp(costs, temperature, bad_lengths, -1),
            lambda: dtw_ops.marginals_backward(costs, tangent, temperature, bad_lengths, -1),
        )

        for call in entrypoints:
            with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 0 and 4"):
                call()

    @pytest.mark.parametrize(
        "device_type",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
    )
    def test_hvp_rejects_mismatched_tangent_shape(self, device_type):
        device = torch.device(device_type)
        costs = torch.randn(2, 5, 6, device=device).abs()
        tangent = torch.randn(2, 5, 5, device=device)

        with pytest.raises(RuntimeError, match="tangent must have same shape as costs"):
            dtw_ops.marginals_hvp(costs, tangent, 1.0, None, -1)

    @pytest.mark.parametrize(
        "device_type",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
    )
    def test_backward_full_rejects_mismatched_grad_alignment_shape(self, device_type):
        device = torch.device(device_type)
        costs = torch.randn(2, 5, 6, device=device).abs()
        grad_alignment = torch.randn(2, 5, 5, device=device)

        with pytest.raises(RuntimeError, match="cotangent must have same shape as costs"):
            dtw_ops.marginals_backward(costs, grad_alignment, 1.0, None, -1)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_backward_full_rejects_cpu_grad_alignment_for_cuda_costs(self):
        costs = torch.randn(2, 5, 6, device="cuda").abs()
        grad_alignment = torch.randn(2, 5, 6)

        with pytest.raises(RuntimeError, match="cotangent must be on same device as costs"):
            dtw_ops.marginals_backward(costs, grad_alignment, 1.0, None, -1)

    def test_noncontiguous_inputs_have_explicit_contract(self):
        """Raw DTW entrypoints state their contiguous-input behavior."""
        costs = torch.rand(2, 4, 5)
        noncontiguous_costs = torch.rand(2, 4, 5, 2)[..., 0]
        tangent = torch.randn_like(costs)
        noncontiguous_tangent = torch.randn(2, 4, 5, 2)[..., 0]
        noncontiguous_cotangent = torch.randn(2, 4, 5, 2)[..., 0]
        noncontiguous_lengths = torch.tensor(
            [[4, 5], [3, 4]], dtype=torch.int32
        ).t()

        assert not noncontiguous_costs.is_contiguous()
        assert not noncontiguous_tangent.is_contiguous()
        assert not noncontiguous_cotangent.is_contiguous()
        assert not noncontiguous_lengths.is_contiguous()

        with pytest.raises(RuntimeError, match="costs must be contiguous"):
            dtw_ops.forward(noncontiguous_costs, 1.0, None, -1)
        with pytest.raises(RuntimeError, match="tangent must be contiguous"):
            dtw_ops.marginals_hvp(costs, noncontiguous_tangent, 1.0, None, -1)
        with pytest.raises(RuntimeError, match="lengths must be contiguous"):
            dtw_ops.forward(costs, 1.0, noncontiguous_lengths, -1)

        with pytest.raises(RuntimeError, match="cotangent.*contiguous"):
            dtw_ops.marginals_backward(
                costs, noncontiguous_cotangent, 1.0, None, -1
            )

        with pytest.raises(ValueError, match="cotangent.*contiguous"):
            orihime.raw.dtw.vjp_one(
                costs,
                wrt="temperature",
                cotangent=noncontiguous_cotangent,
                temperature=1.0,
            )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        temperature = 1.0

        torch.manual_seed(42)
        costs = torch.randn(B, max_L1, max_L2, device=device).abs()

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        score, posteriors = dtw_ops.forward(costs, temperature, lengths, -1)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            costs_b = costs[b:b+1, :l1, :l2]

            score_ref, _ = dtw_forward_naive(costs_b, temperature)
            posteriors_ref = dtw_naive(costs_b, temperature)

            # Score should match for this sequence
            assert allclose(score_ref, score[b:b+1]), \
                f"Score mismatch for batch {b}: {score_ref.item()} vs {score[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"

            if l1 < max_L1:
                assert torch.count_nonzero(posteriors[b, l1:, :]).item() == 0, \
                    f"Padded rows should stay zero for batch {b}"
            if l2 < max_L2:
                assert torch.count_nonzero(posteriors[b, :, l2:]).item() == 0, \
                    f"Padded columns should stay zero for batch {b}"

    def test_padded_regions_stay_zero_for_all_derivative_paths(self, device):
        """Every DTW map/derivative field must preserve zero padded regions."""
        B, max_L1, max_L2 = 3, 6, 7
        temperature = 0.9
        bandwidth = 2

        torch.manual_seed(123)
        costs = torch.randn(B, max_L1, max_L2, device=device).abs()
        tangent = torch.randn_like(costs)
        lengths = torch.tensor(
            [[max_L1, max_L2], [4, 5], [5, 3]],
            dtype=torch.int32,
            device=device,
        )

        _, posteriors = dtw_ops.forward(
            costs, temperature, lengths, bandwidth
        )
        with_grads = dtw_forward_with_grads(
            costs, temperature, lengths, bandwidth
        )
        hvp = dtw_ops.marginals_hvp(
            costs, tangent, temperature, lengths, bandwidth
        )
        dP_dT = dtw_ops.marginals_grad_temp(
            costs, temperature, lengths, bandwidth
        )
        full = dtw_ops.marginals_backward(
            costs, tangent, temperature, lengths, bandwidth
        )

        assert_padded_pairwise_region_zero("forward posteriors", posteriors, lengths)
        assert_padded_pairwise_region_zero(
            "with_grads posteriors", with_grads[1], lengths
        )
        assert_padded_pairwise_region_zero("HVP", hvp, lengths)
        assert_padded_pairwise_region_zero("temperature Jacobian", dP_dT, lengths)
        assert_padded_pairwise_region_zero(
            "full VJP costs gradient", full[0], lengths
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 cost matrix."""
        costs = torch.tensor([[[0.5]]], device=device)
        temperature = 1.0

        score, posteriors = dtw_ops.forward(costs, temperature, None, -1)

        # Single element: score = cost, posterior = 1.0
        assert allclose(score, torch.tensor([0.5], device=device)), "Single element score wrong"
        assert allclose(posteriors, torch.ones_like(costs)), "Single element posterior wrong"

    def test_row_vector(self, device):
        """Test 1xN cost matrix."""
        costs = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        temperature = 1.0

        score, posteriors = dtw_ops.forward(costs, temperature, None, -1)
        score_ref, _ = dtw_forward_naive(costs, temperature)

        assert allclose(score, score_ref), "Row vector score mismatch"
        # All posteriors should be 1 (only one path)
        assert allclose(posteriors, torch.ones_like(costs)), "Row vector posteriors wrong"

    def test_col_vector(self, device):
        """Test Nx1 cost matrix."""
        costs = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        temperature = 1.0

        score, posteriors = dtw_ops.forward(costs, temperature, None, -1)
        score_ref, _ = dtw_forward_naive(costs, temperature)

        assert allclose(score, score_ref), "Column vector score mismatch"
        # All posteriors should be 1 (only one path)
        assert allclose(posteriors, torch.ones_like(costs)), "Column vector posteriors wrong"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard DTW)."""
        B, L1, L2 = 2, 6, 8
        temperature = 0.01

        torch.manual_seed(42)
        costs = torch.randn(B, L1, L2, device=device).abs()

        posteriors = dtw_ops.forward(costs, temperature, None, -1)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        temperature = 10.0

        torch.manual_seed(42)
        costs = torch.randn(B, L1, L2, device=device).abs()

        posteriors = dtw_ops.forward(costs, temperature, None, -1)[1]
        posteriors_ref = dtw_naive(costs, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"


# --- Memory-safety regression tests (merged from test_dtw_{cpp,cuda}_memsafety.py) ---

OVERSIZED_L = 65_536
OVERSIZED_ERROR = "DTW CPU .*supported int32 index range"


def _oversized_contiguous_costs():
    # A zero-batch tensor preserves the dangerous L1 x L2 shape metadata while
    # containing no elements or backing storage. Empty tensors are contiguous,
    # so this reaches the native int32 index-range guard without relying on a
    # runner accepting a ~16 GiB allocation or virtual-memory mapping.
    costs = torch.empty((0, OVERSIZED_L, OVERSIZED_L), dtype=torch.float32)
    assert costs.is_contiguous()
    assert costs.untyped_storage().nbytes() == 0
    return costs


def _oversized_lengths():
    return torch.empty((0, 2), dtype=torch.int32)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.parametrize(
    "call",
    [
        lambda costs: dtw_ops.forward_t(
            costs, torch.tensor([1.0], dtype=torch.float32), _oversized_lengths(), -1
        ),
        lambda costs: dtw_ops.forward(costs, 1.0, None, -1),
        lambda costs: dtw_forward_with_grads(costs, 1.0, None, -1),
        lambda costs: dtw_ops.marginals_hvp(costs, costs, 1.0, None, -1),
        lambda costs: dtw_ops.marginals_grad_temp(costs, 1.0, None, -1),
        lambda costs: dtw_ops.marginals_backward(costs, costs, 1.0, None, -1),
    ],
)
def test_oversized_cpu_dtw_rejects_before_workspace_allocation(call):
    costs = _oversized_contiguous_costs()
    with pytest.raises(RuntimeError, match=OVERSIZED_ERROR):
        call(costs)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_valid_cpu_dtw_outputs_match_reference():
    torch.manual_seed(20260707)
    costs = torch.randn(2, 5, 6, dtype=torch.float32).abs()
    temperature = 0.7
    bandwidth = 2

    expected_score, _ = dtw_forward_naive(costs, temperature, bandwidth)
    expected_alignment = dtw_naive(costs, temperature, bandwidth)

    score, alignment = dtw_ops.forward(
        costs, temperature, None, bandwidth
    )

    assert torch.allclose(score, expected_score, rtol=1e-4, atol=1e-5)
    assert torch.allclose(alignment, expected_alignment, rtol=1e-3, atol=1e-4)


def cuda_kernel_names(call):
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        pytest.skip(f"CUDA profiler unavailable: {exc}")

    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
    except RuntimeError as exc:
        message = str(exc).lower()
        if "cupti" in message or "kineto" in message or "profiler" in message:
            pytest.skip(f"CUDA profiler failed: {exc}")
        raise

    return {
        getattr(event, "name", "") or getattr(event, "key", "")
        for event in prof.events()
    }


@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="DTW CUDA memsafety regressions require a CUDA-enabled orihime build",
)
def test_all_entrypoints_reject_alpha_size_overflow_before_launch():
    costs = torch.empty((0, 46341, 46341), device="cuda")
    tangent = torch.empty_like(costs)
    temperature = torch.tensor([1.0], device="cuda")
    lengths = torch.empty((0, 2), dtype=torch.int32, device="cuda")

    entrypoints = (
        lambda: dtw_ops.forward_t(costs, temperature, lengths, -1),
        lambda: dtw_ops.forward(costs, 1.0, None, -1),
        lambda: dtw_forward_with_grads(costs, 1.0, None, -1),
        lambda: dtw_ops.marginals_hvp(costs, tangent, 1.0, None, -1),
        lambda: dtw_ops.marginals_grad_temp(costs, 1.0, None, -1),
        lambda: dtw_ops.marginals_backward(costs, tangent, 1.0, None, -1),
    )

    for call in entrypoints:
        with pytest.raises(RuntimeError, match="DTW alpha table is too large"):
            call()


@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="DTW CUDA memsafety regressions require a CUDA-enabled orihime build",
)
@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
def test_entrypoints_guard_costs_device_when_current_device_differs():
    costs = torch.rand((2, 4, 5), device="cuda:0")
    tangent = torch.randn_like(costs)
    temperature = torch.tensor([0.7], device="cuda:0")
    lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32, device="cuda:0")

    with torch.cuda.device(1):
        assert costs.device.index != torch.cuda.current_device()
        score, posteriors = dtw_ops.forward_t(costs, temperature, lengths, -1)
        score_f, posteriors_f = dtw_ops.forward(costs, 0.7, lengths, -1)
        score_g, posteriors_g, grad_T = dtw_forward_with_grads(costs, 0.7, lengths, -1)
        hvp = dtw_ops.marginals_hvp(costs, tangent, 0.7, lengths, -1)
        dP_dT = dtw_ops.marginals_grad_temp(costs, 0.7, lengths, -1)
        grad_costs, grad_temp = dtw_ops.marginals_backward(costs, tangent, 0.7, lengths, -1)

    for tensor in (
        score,
        posteriors,
        score_f,
        posteriors_f,
        score_g,
        posteriors_g,
        grad_T,
        hvp,
        dP_dT,
        grad_costs,
        grad_temp,
    ):
        assert tensor.device == costs.device


@pytest.mark.multi_gpu
@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="DTW CUDA device guards require a CUDA-enabled orihime build",
)
@TWO_CUDA_DEVICES_REQUIRED
def test_entrypoints_reject_wrong_device_secondary_tensors():
    costs = torch.rand((2, 4, 5), device="cuda:0")
    lengths = torch.tensor(
        [[4, 5], [3, 4]], dtype=torch.int32, device="cuda:0"
    )
    wrong_lengths = lengths.to("cuda:1")
    tangent = torch.randn_like(costs)
    wrong_tangent = tangent.to("cuda:1")
    cotangent = torch.randn_like(costs)
    wrong_cotangent = cotangent.to("cuda:1")
    temperature = costs.new_tensor([0.7])
    wrong_temperature = torch.tensor([0.7], device="cuda:1")
    assert costs.device.index != wrong_lengths.device.index
    assert costs.device.index != wrong_tangent.device.index
    assert costs.device.index != wrong_cotangent.device.index
    assert costs.device.index != wrong_temperature.device.index

    with pytest.raises(RuntimeError, match="temperature must be on same device as costs"):
        dtw_ops.forward_t(costs, wrong_temperature, lengths, -1)

    length_calls = (
        lambda: dtw_ops.forward_t(costs, temperature, wrong_lengths, -1),
        lambda: dtw_ops.forward(costs, 0.7, wrong_lengths, -1),
        lambda: dtw_forward_with_grads(costs, 0.7, wrong_lengths, -1),
        lambda: dtw_ops.marginals_hvp(
            costs, tangent, 0.7, wrong_lengths, -1
        ),
        lambda: dtw_ops.value_grad_params(
            costs, 0.7, wrong_lengths, -1
        ),
        lambda: dtw_ops.marginals_grad_temp(
            costs, 0.7, wrong_lengths, -1
        ),
        lambda: dtw_ops.marginals_backward(
            costs, cotangent, 0.7, wrong_lengths, -1
        ),
    )
    for call in length_calls:
        with pytest.raises(RuntimeError, match="lengths must be on same device"):
            call()

    with pytest.raises(ValueError, match="lengths.*same device"):
        orihime.dtw(
            costs,
            temperature=0.7,
            lengths=wrong_lengths,
            bandwidth=None,
        )
    with pytest.raises(ValueError, match="lengths.*same device"):
        orihime.raw.dtw.vjp_one(
            costs,
            wrt="temperature",
            cotangent=cotangent,
            temperature=0.7,
            lengths=wrong_lengths,
        )

    with pytest.raises(ValueError, match="temperature.*same device"):
        orihime.dtw(
            costs,
            temperature=wrong_temperature,
            lengths=lengths,
            bandwidth=None,
        )
    with pytest.raises(ValueError, match="temperature.*same device"):
        orihime.raw.dtw.vjp_one(
            costs,
            wrt="temperature",
            cotangent=cotangent,
            temperature=wrong_temperature,
            lengths=lengths,
            bandwidth=None,
        )

    with pytest.raises(RuntimeError, match="tangent must be on same device"):
        dtw_ops.marginals_hvp(costs, wrong_tangent, 0.7, lengths, -1)

    with pytest.raises(RuntimeError, match="cotangent must be on same device"):
        dtw_ops.marginals_backward(
            costs, wrong_cotangent, 0.7, lengths, -1
        )
    with pytest.raises(ValueError, match="cotangent.*same device"):
        orihime.raw.dtw.vjp_one(
            costs,
            wrt="temperature",
            cotangent=wrong_cotangent,
            temperature=0.7,
            lengths=lengths,
        )


@pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="DTW CUDA memsafety regressions require a CUDA-enabled orihime build",
)
def test_score_only_backward_skips_zero_posterior_hvp_and_param_grad_kernels():
    costs = torch.rand((2, 5, 6), device="cuda", requires_grad=True)
    temperature = torch.tensor([0.9], device="cuda", requires_grad=True)
    lengths = torch.tensor([[5, 6], [4, 5]], dtype=torch.int32, device="cuda")

    score_ref, posteriors_ref, grad_T_ref = dtw_forward_with_grads(
        costs.detach(), 0.9, lengths, -1
    )
    assert score_ref.shape == (2,)

    def score_only_backward():
        costs.grad = None
        temperature.grad = None
        score, _ = dtw_ops.forward_t(costs, temperature, lengths, -1)
        score.sum().backward()

    kernel_names = cuda_kernel_names(score_only_backward)
    unexpected = sorted(
        name
        for name in kernel_names
        if "dtw_hvp_" in name or "dtw_param_grad_" in name
    )
    assert not unexpected, f"score-only backward launched zero-contribution kernels: {unexpected}"

    assert torch.allclose(costs.grad, posteriors_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(temperature.grad, grad_T_ref.sum().reshape_as(temperature), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
