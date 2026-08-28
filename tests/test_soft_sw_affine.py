# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft Smith-Waterman (affine gap).
"""

import contextlib

import pytest
import torch

from reference import sw_affine_forward_naive, sw_affine_naive
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

try:
    import orihime
    sw_affine_ops = orihime.ops._kernels["sw_affine"]
    from operator_test_utils import (
        sw_affine_forward_with_grads,
        sw_affine_param_field,
    )
    ORIHIME_AVAILABLE = True
except ImportError:
    ORIHIME_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()
HVP_FINITE_DIFF_STEP = 5e-3


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def seeded_randn(shape, seed, device=None):
    generator = torch.Generator()
    generator.manual_seed(seed)
    values = torch.randn(*shape, generator=generator)
    return values if device is None else values.to(device)


def assert_padded_region_zero(name, tensor, lengths):
    tensor_cpu = tensor.cpu()
    lengths_cpu = lengths.cpu()
    max_L1 = tensor_cpu.size(1)
    max_L2 = tensor_cpu.size(2)

    for batch, (l1, l2) in enumerate(lengths_cpu.tolist()):
        if l1 < max_L1:
            assert torch.count_nonzero(tensor_cpu[batch, l1:, :]).item() == 0, \
                f"{name} batch {batch} wrote past active L1 length"
        if l2 < max_L2:
            assert torch.count_nonzero(tensor_cpu[batch, :, l2:]).item() == 0, \
                f"{name} batch {batch} wrote past active L2 length"


def finite_diff_param_jacobian(scores, param_type, gap_open, gap_ext, temperature, lengths, eps):
    if param_type == 0:
        plus = sw_affine_ops.forward(scores, gap_open + eps, gap_ext, temperature, lengths)[1]
        minus = sw_affine_ops.forward(scores, gap_open - eps, gap_ext, temperature, lengths)[1]
    elif param_type == 1:
        plus = sw_affine_ops.forward(scores, gap_open, gap_ext + eps, temperature, lengths)[1]
        minus = sw_affine_ops.forward(scores, gap_open, gap_ext - eps, temperature, lengths)[1]
    else:
        plus = sw_affine_ops.forward(scores, gap_open, gap_ext, temperature + eps, lengths)[1]
        minus = sw_affine_ops.forward(scores, gap_open, gap_ext, temperature - eps, lengths)[1]
    return (plus - minus) / (2 * eps)


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(scores, tangent, gap_open, gap_ext, temperature):
    partition, posteriors, grad_open, grad_ext, grad_T = sw_affine_forward_with_grads(
        scores, gap_open, gap_ext, temperature, None
    )
    hvp = sw_affine_ops.marginals_hvp(scores, tangent, gap_open, gap_ext, temperature, None)
    dP_dgap_open = sw_affine_param_field(scores, 0, gap_open, gap_ext, temperature, None)
    dP_dgap_ext = sw_affine_param_field(scores, 1, gap_open, gap_ext, temperature, None)
    dP_dT = sw_affine_param_field(scores, 2, gap_open, gap_ext, temperature, None)
    return {
        "partition": partition,
        "posteriors": posteriors,
        "grad_open": grad_open,
        "grad_ext": grad_ext,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dgap_open": dP_dgap_open,
        "dP_dgap_ext": dP_dgap_ext,
        "dP_dT": dP_dT,
    }


def assert_threaded_sw_affine_correctness(outputs, reference_outputs, thread_count):
    partition_ref = reference_outputs["partition"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_open_ref = reference_outputs["grad_open"]
    grad_ext_ref = reference_outputs["grad_ext"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]

    assert allclose(partition_ref, outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(partition_ref, outputs['partition'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_open_ref, outputs["grad_open"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_open mismatch: max diff = {max_diff(grad_open_ref, outputs['grad_open'])}"
    assert allclose(grad_ext_ref, outputs["grad_ext"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_ext mismatch: max diff = {max_diff(grad_ext_ref, outputs['grad_ext'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=1e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"


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

    def test_partition(self, batch_size, seq_lengths, temperature, device):
        """Test that partition functions match."""
        L1, L2 = seq_lengths
        gap_open = -2.0
        gap_ext = -0.5

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        partition_ref, _, _, _ = sw_affine_forward_naive(
            scores, gap_open, gap_ext, temperature
        )
        partition_orihime = sw_affine_ops.forward(
            scores, gap_open, gap_ext, temperature, None
        )[0]

        assert allclose(partition_ref, partition_orihime), \
            f"Partition mismatch: max diff = {max_diff(partition_ref, partition_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths
        gap_open = -2.0
        gap_ext = -0.5

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        posteriors_ref = sw_affine_naive(
            scores, gap_open, gap_ext, temperature
        )
        posteriors_orihime = sw_affine_ops.forward(
            scores, gap_open, gap_ext, temperature, None
        )[1]

        assert allclose(posteriors_ref, posteriors_orihime, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_orihime)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        gap_open = -2.0
        gap_ext = -0.5

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device, requires_grad=True)

        posteriors = sw_affine_ops.forward(scores, gap_open, gap_ext, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_orihime = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = sw_affine_naive(scores_ref, gap_open, gap_ext, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"

    def test_gap_parameter_gradients_respect_local_restart_boundary(self):
        """Local SW derivatives must match the restart-aware recurrence at boundaries."""
        B, L1, L2 = 2, 5, 7
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0
        eps = 1e-4

        scores = seeded_randn((B, L1, L2), 7)
        scores[:, 0, :] -= 2.5
        scores[:, :, 0] -= 2.5

        _, _, grad_open, grad_ext, grad_T = sw_affine_forward_with_grads(
            scores, gap_open, gap_ext, temperature, None
        )

        partition_open_plus, _, _, _ = sw_affine_forward_naive(
            scores, gap_open + eps, gap_ext, temperature
        )
        partition_open_minus, _, _, _ = sw_affine_forward_naive(
            scores, gap_open - eps, gap_ext, temperature
        )
        grad_open_ref = (partition_open_plus - partition_open_minus) / (2 * eps)

        partition_ext_plus, _, _, _ = sw_affine_forward_naive(
            scores, gap_open, gap_ext + eps, temperature
        )
        partition_ext_minus, _, _, _ = sw_affine_forward_naive(
            scores, gap_open, gap_ext - eps, temperature
        )
        grad_ext_ref = (partition_ext_plus - partition_ext_minus) / (2 * eps)

        partition_T_plus, _, _, _ = sw_affine_forward_naive(
            scores, gap_open, gap_ext, temperature + eps
        )
        partition_T_minus, _, _, _ = sw_affine_forward_naive(
            scores, gap_open, gap_ext, temperature - eps
        )
        grad_T_ref = (partition_T_plus - partition_T_minus) / (2 * eps)

        assert allclose(grad_open_ref, grad_open, rtol=1e-2, atol=2e-3), \
            f"local grad_open boundary mismatch: max diff = {max_diff(grad_open_ref, grad_open)}"
        assert allclose(grad_ext_ref, grad_ext, rtol=1e-2, atol=2e-3), \
            f"local grad_ext boundary mismatch: max diff = {max_diff(grad_ext_ref, grad_ext)}"
        assert allclose(grad_T_ref, grad_T, rtol=1e-2, atol=2e-3), \
            f"local grad_T boundary mismatch: max diff = {max_diff(grad_T_ref, grad_T)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        gap_open = -2.0
        gap_ext = -0.5
        temperature = 1.0
        eps = HVP_FINITE_DIFF_STEP

        scores = seeded_randn((B, L1, L2), 20260813, device)
        V = seeded_randn((B, L1, L2), 20260814, device)

        hvp_orihime = sw_affine_ops.marginals_hvp(scores, V, gap_open, gap_ext, temperature, None)

        posteriors_plus = sw_affine_naive(
            scores + eps * V, gap_open, gap_ext, temperature
        )
        posteriors_minus = sw_affine_naive(
            scores - eps * V, gap_open, gap_ext, temperature
        )
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=1e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestEdgeCases:

    @staticmethod
    def assert_matches_reference(scores, gap_open, gap_ext, temperature):
        partition, posteriors = sw_affine_ops.forward(
            scores, gap_open, gap_ext, temperature, None
        )
        partition_ref, _, _, _ = sw_affine_forward_naive(
            scores, gap_open, gap_ext, temperature
        )
        posteriors_ref = sw_affine_naive(
            scores, gap_open, gap_ext, temperature
        )
        assert allclose(partition, partition_ref), "edge-case partition mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3, atol=1e-4), \
            "edge-case posterior mismatch"

    def test_single_element(self, device):
        self.assert_matches_reference(
            torch.tensor([[[0.1]]], device=device), -2.0, -0.5, 1.0
        )

    def test_row_vector(self, device):
        self.assert_matches_reference(
            torch.tensor([[[0.1, 0.2, 0.3]]], device=device),
            -2.0,
            -0.5,
            1.0,
        )

    def test_col_vector(self, device):
        self.assert_matches_reference(
            torch.tensor([[[0.1], [0.2], [0.3]]], device=device),
            -2.0,
            -0.5,
            1.0,
        )

    def test_low_temperature(self, device):
        scores = seeded_randn((2, 6, 8), 42, device)
        self.assert_matches_reference(scores, -2.0, -0.5, 0.01)

    def test_high_temperature(self, device):
        scores = seeded_randn((2, 6, 8), 42, device)
        self.assert_matches_reference(scores, -2.0, -0.5, 10.0)

    @pytest.mark.parametrize(
        ("gap_open", "gap_ext"),
        [(-1.0, -1.0), (-2.0, 0.0), (0.5, 0.2)],
    )
    def test_gap_penalty_edges(self, device, gap_open, gap_ext):
        scores = seeded_randn((2, 6, 6), 42, device)
        self.assert_matches_reference(scores, gap_open, gap_ext, 1.0)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        gap_open = -2.0
        gap_ext = -0.5
        temperature = 1.0
        eps = 1e-4
        hvp_eps = HVP_FINITE_DIFF_STEP
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.randn(B, L1, L2)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            partition_ref, _, _, _ = sw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
            posteriors_ref = sw_affine_naive(scores, gap_open, gap_ext, temperature)

            partition_open_plus, _, _, _ = sw_affine_forward_naive(
                scores, gap_open + eps, gap_ext, temperature
            )
            partition_open_minus, _, _, _ = sw_affine_forward_naive(
                scores, gap_open - eps, gap_ext, temperature
            )
            grad_open_ref = (partition_open_plus - partition_open_minus) / (2 * eps)

            partition_ext_plus, _, _, _ = sw_affine_forward_naive(
                scores, gap_open, gap_ext + eps, temperature
            )
            partition_ext_minus, _, _, _ = sw_affine_forward_naive(
                scores, gap_open, gap_ext - eps, temperature
            )
            grad_ext_ref = (partition_ext_plus - partition_ext_minus) / (2 * eps)

            partition_temp_plus, _, _, _ = sw_affine_forward_naive(
                scores, gap_open, gap_ext, temperature + eps
            )
            partition_temp_minus, _, _, _ = sw_affine_forward_naive(
                scores, gap_open, gap_ext, temperature - eps
            )
            grad_T_ref = (partition_temp_plus - partition_temp_minus) / (2 * eps)

            posteriors_plus = sw_affine_naive(
                scores + hvp_eps * tangent, gap_open, gap_ext, temperature
            )
            posteriors_minus = sw_affine_naive(
                scores - hvp_eps * tangent, gap_open, gap_ext, temperature
            )
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * hvp_eps)

        reference_outputs = {
            "partition": partition_ref,
            "posteriors": posteriors_ref,
            "grad_open": grad_open_ref,
            "grad_ext": grad_ext_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, gap_open, gap_ext, temperature)
            assert_threaded_sw_affine_correctness(outputs, reference_outputs, thread_count)
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
        gap_open = -2.0
        gap_ext = -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = sw_affine_ops.forward(scores_cpu, gap_open, gap_ext, temperature, None)[1]
        posteriors_cuda = sw_affine_ops.forward(scores_cuda, gap_open, gap_ext, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_derivative_entrypoints_cpu_cuda_parity(self):
        """CPU/CUDA agree for HVP, full VJP, sensitivities, and autograd."""
        B, L1, L2 = 2, 6, 7
        gap_open, gap_ext, temperature = -1.7, -0.4, 0.9

        scores_cpu = seeded_randn((B, L1, L2), 20260810)
        lengths_cpu = torch.tensor([[6, 7], [4, 5]], dtype=torch.int32)
        tangent_cpu = seeded_randn((B, L1, L2), 20260811)
        cotangent_cpu = seeded_randn((B, L1, L2), 20260812)

        def run_backend(scores, lengths, tangent, cotangent):
            partition, posteriors, grad_open, grad_ext, grad_T = (
                sw_affine_forward_with_grads(
                    scores, gap_open, gap_ext, temperature, lengths
                )
            )
            hvp = sw_affine_ops.marginals_hvp(
                scores, tangent, gap_open, gap_ext, temperature, lengths
            )
            sensitivities = tuple(
                sw_affine_param_field(
                    scores, index, gap_open, gap_ext, temperature, lengths
                )
                for index in range(3)
            )
            full_vjp = sw_affine_ops.marginals_backward(
                scores,
                cotangent,
                gap_open,
                gap_ext,
                temperature,
                lengths,
            )
            scores_with_grad = scores.detach().clone().requires_grad_(True)
            score_autograd, map_autograd = sw_affine_ops.forward(
                scores_with_grad,
                gap_open,
                gap_ext,
                temperature,
                lengths,
            )
            grad_autograd = torch.autograd.grad(
                score_autograd.sum() + 0.25 * map_autograd.sum(),
                scores_with_grad,
            )[0]
            return (
                partition,
                posteriors,
                grad_open,
                grad_ext,
                grad_T,
                hvp,
                *sensitivities,
                *full_vjp,
                grad_autograd,
            )

        cpu_outputs = run_backend(
            scores_cpu, lengths_cpu, tangent_cpu, cotangent_cpu
        )
        cuda_outputs = run_backend(
            scores_cpu.cuda(),
            lengths_cpu.cuda(),
            tangent_cpu.cuda(),
            cotangent_cpu.cuda(),
        )

        output_names = (
            "partition",
            "posteriors",
            "grad_open",
            "grad_ext",
            "grad_temperature",
            "hvp",
            "param_field_gap_open",
            "param_field_gap_ext",
            "param_field_temperature",
            "full_vjp_scores",
            "full_vjp_gap_open",
            "full_vjp_gap_ext",
            "full_vjp_temperature",
            "autograd_scores",
        )
        scalar_parameter_gradients = {"grad_open", "grad_ext", "grad_temperature"}

        assert len(output_names) == len(cpu_outputs) == len(cuda_outputs) == 14
        for name, cpu, cuda in zip(output_names, cpu_outputs, cuda_outputs):
            if name in {"partition", "posteriors"}:
                assert allclose(cpu, cuda), \
                    f"CPU/CUDA {name} mismatch: max diff = {max_diff(cpu, cuda)}"
                continue

            if name in scalar_parameter_gradients:
                rtol, atol = 1e-2, 2e-3
            else:
                rtol, atol = 1e-2, 5e-3
            assert allclose(cpu, cuda, rtol=rtol, atol=atol), \
                f"CPU/CUDA {name} mismatch: max diff = {max_diff(cpu, cuda)}"

    def test_param_jacobian_variable_lengths_matches_finite_diff(self):
        """CPU and CUDA param_jacobians must match finite differences on masked batches."""
        B, max_L1, max_L2 = 3, 6, 7
        gap_open, gap_ext, temperature = -1.7, -0.4, 0.9
        eps = 1e-3

        torch.manual_seed(20260409)
        scores_cpu = torch.randn(B, max_L1, max_L2)
        scores_cpu[0, 0, :] -= 2.5
        scores_cpu[1, :, 0] -= 2.0
        scores_cpu[2, 0, 0] -= 3.0
        lengths_cpu = torch.tensor([[6, 7], [4, 5], [5, 3]], dtype=torch.int32)

        scores_cuda = scores_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()

        for backend, scores, lengths in (
            ("cpu", scores_cpu, lengths_cpu),
            ("cuda", scores_cuda, lengths_cuda),
        ):
            for param_type, name in ((0, "gap_open"), (1, "gap_ext"), (2, "temperature")):
                actual = sw_affine_param_field(
                    scores, param_type, gap_open, gap_ext, temperature, lengths
                )
                fd = finite_diff_param_jacobian(
                    scores, param_type, gap_open, gap_ext, temperature, lengths, eps
                )
                assert allclose(fd, actual, rtol=2e-2, atol=5e-3), \
                    f"{backend} dP/d{name} mismatch vs finite diff: max diff = {max_diff(fd, actual)}"
                assert_padded_region_zero(f"{backend} dP/d{name}", actual, lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Variable-length batches should match the naive per-example result."""
        B = 4
        max_L1, max_L2 = 10, 12
        gap_open = -2.0
        gap_ext = -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, max_L1, max_L2, device=device)
        lengths = torch.tensor(
            [[8, 10], [10, 12], [6, 8], [9, 11]],
            device=device,
            dtype=torch.int32,
        )

        partition, posteriors = sw_affine_ops.forward(
            scores, gap_open, gap_ext, temperature, lengths
        )
        assert_padded_region_zero("variable_length_posteriors", posteriors, lengths)

        _, posteriors_with_grads, grad_open, grad_ext, grad_T = (
            sw_affine_forward_with_grads(
                scores, gap_open, gap_ext, temperature, lengths
            )
        )
        assert_padded_region_zero(
            "variable_length_posteriors_with_grads",
            posteriors_with_grads,
            lengths,
        )
        for name, field in (
            ("grad_gap_open", sw_affine_param_field(
                scores, 0, gap_open, gap_ext, temperature, lengths
            )),
            ("grad_gap_ext", sw_affine_param_field(
                scores, 1, gap_open, gap_ext, temperature, lengths
            )),
            ("grad_temperature", sw_affine_param_field(
                scores, 2, gap_open, gap_ext, temperature, lengths
            )),
        ):
            assert_padded_region_zero(name, field, lengths)
        assert grad_open.shape == (B,)
        assert grad_ext.shape == (B,)
        assert grad_T.shape == (B,)

        tangent = seeded_randn(tuple(scores.shape), 751, scores.device)
        hvp = sw_affine_ops.marginals_hvp(
            scores, tangent, gap_open, gap_ext, temperature, lengths
        )
        assert_padded_region_zero("hvp", hvp, lengths)

        cotangent = seeded_randn(tuple(scores.shape), 752, scores.device)
        grad_scores, grad_open_vjp, grad_ext_vjp, grad_T_vjp = (
            sw_affine_ops.marginals_backward(
                scores,
                cotangent,
                gap_open,
                gap_ext,
                temperature,
                lengths,
            )
        )
        assert_padded_region_zero("full_vjp_grad_scores", grad_scores, lengths)
        assert grad_open_vjp.shape == (1,)
        assert grad_ext_vjp.shape == (1,)
        assert grad_T_vjp.shape == (1,)

        scores_with_grad = scores.detach().clone().requires_grad_(True)
        score_autograd, map_autograd = sw_affine_ops.forward(
            scores_with_grad,
            gap_open,
            gap_ext,
            temperature,
            lengths,
        )
        grad_autograd = torch.autograd.grad(
            score_autograd.sum() + map_autograd.sum(),
            scores_with_grad,
        )[0]
        assert_padded_region_zero("autograd_grad_scores", grad_autograd, lengths)

        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            partition_ref, _, _, _ = sw_affine_forward_naive(
                scores_b, gap_open, gap_ext, temperature
            )
            posteriors_ref = sw_affine_naive(scores_b, gap_open, gap_ext, temperature)

            assert allclose(partition_ref, partition[b:b+1]), \
                f"Partition mismatch for batch {b}: {partition_ref.item()} vs {partition[b].item()}"
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestLengthValidation:

    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [[-1, 5]], r"lengths\[0,0\] must be between 0 and 4"),
            ("cpu", [[4, 6]], r"lengths\[0,1\] must be between 0 and 5"),
            pytest.param(
                "cuda",
                [[-1, 5]],
                r"lengths\[0,0\] must be between 0 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[4, 6]],
                r"lengths\[0,1\] must be between 0 and 5",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_invalid_lengths_raise(self, device_type, lengths, match):
        """Explicit lengths must stay within the padded score shape."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            sw_affine_ops.forward(scores, -2.0, -0.5, 1.0, lengths_t)

    def test_hvp_rejects_non_3d_scores_cpu(self):
        """CPU HVP scores must be rank-3 before sizes are read."""
        scores = torch.randn(4, 5)
        tangent = torch.randn(4, 5)

        with pytest.raises(RuntimeError, match=r"scores must be 3D"):
            sw_affine_ops.marginals_hvp(scores, tangent, -2.0, -0.5, 1.0, None)

    def test_hvp_rejects_mismatched_tangent_shape_cpu(self):
        """CPU HVP tangents must match the padded score tensor shape."""
        scores = torch.randn(1, 4, 5)
        tangent = torch.randn(1, 4, 4)

        with pytest.raises(RuntimeError, match=r"tangent must have same shape as scores"):
            sw_affine_ops.marginals_hvp(scores, tangent, -2.0, -0.5, 1.0, None)

    def test_forward_rejects_noncontiguous_scores_cpu(self):
        scores = seeded_randn((1, 5, 4), 761).transpose(1, 2)
        assert scores.shape == (1, 4, 5)
        assert not scores.is_contiguous()

        with pytest.raises(RuntimeError, match=r"scores must be contiguous"):
            sw_affine_ops.forward(scores, -2.0, -0.5, 1.0, None)

    def test_hvp_rejects_noncontiguous_tangent_cpu(self):
        scores = seeded_randn((1, 4, 5), 762)
        tangent = seeded_randn((1, 5, 4), 763).transpose(1, 2)
        assert not tangent.is_contiguous()

        with pytest.raises(RuntimeError, match=r"tangent must be contiguous"):
            sw_affine_ops.marginals_hvp(
                scores, tangent, -2.0, -0.5, 1.0, None
            )

    def test_backward_full_rejects_noncontiguous_cotangent_cpu(self):
        scores = seeded_randn((1, 4, 5), 764)
        cotangent = seeded_randn((1, 5, 4), 765).transpose(1, 2)
        assert cotangent.shape == scores.shape
        assert not cotangent.is_contiguous()

        with pytest.raises(RuntimeError, match=r"cotangent must be contiguous"):
            sw_affine_ops.marginals_backward(
                scores, cotangent, -2.0, -0.5, 1.0, None
            )

    @pytest.mark.multi_gpu
    @TWO_CUDA_DEVICES_REQUIRED
    def test_cuda_lengths_must_match_scores_device(self):
        """Explicit CUDA lengths must live on the same device as scores."""
        scores = torch.randn(1, 4, 5, device="cuda:0")
        lengths_t = torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:1")
        assert scores.device.index != lengths_t.device.index

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            sw_affine_ops.forward(scores, -2.0, -0.5, 1.0, lengths_t)

    @pytest.mark.multi_gpu
    @TWO_CUDA_DEVICES_REQUIRED
    def test_hvp_rejects_tangent_on_wrong_cuda_device(self):
        scores = seeded_randn((1, 4, 5), 766, "cuda:0")
        tangent = seeded_randn((1, 4, 5), 767, "cuda:1")
        assert scores.device.index != tangent.device.index

        with pytest.raises(RuntimeError, match=r"tangent must be on same device as scores"):
            sw_affine_ops.marginals_hvp(
                scores, tangent, -2.0, -0.5, 1.0, None
            )


# --- Memory-safety regression tests (merged from test_sw_affine_{cpp,cuda}_memsafety.py) ---

# CPU memory-safety regression tests
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_rejects_dp_table_size_overflow_before_allocation():
    huge_l2 = 715_827_882
    scores = torch.empty((1, 0, huge_l2), dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"sw_affine CPU DP table is too large"):
        sw_affine_ops.forward(scores, -2.0, -0.5, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_hvp_rejects_mismatched_tangent_shape():
    scores = torch.randn(1, 4, 5)
    tangent = torch.randn(1, 4, 4)

    with pytest.raises(RuntimeError, match=r"tangent must have same shape as scores"):
        sw_affine_ops.marginals_hvp(scores, tangent, -2.0, -0.5, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_backward_full_rejects_mismatched_grad_alignment_shape():
    scores = torch.randn(1, 4, 5)
    grad_alignment = torch.randn(1, 4, 4)

    with pytest.raises(RuntimeError, match=r"cotangent must have same shape as scores"):
        sw_affine_ops.marginals_backward(scores, grad_alignment, -2.0, -0.5, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_backward_full_rejects_non_float_grad_alignment():
    scores = torch.randn(1, 4, 5)
    grad_alignment = torch.randn(1, 4, 5, dtype=torch.float64)

    with pytest.raises(RuntimeError, match=r"cotangent must have dtype torch\.float32"):
        sw_affine_ops.marginals_backward(scores, grad_alignment, -2.0, -0.5, 1.0, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_backward_full_rejects_cuda_grad_alignment():
    scores = torch.randn(1, 4, 5)
    grad_alignment = torch.randn(1, 4, 5, device="cuda")

    with pytest.raises(RuntimeError, match=r"cotangent must be on same device as scores"):
        sw_affine_ops.marginals_backward(scores, grad_alignment, -2.0, -0.5, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_backward_full_valid_input_still_runs():
    torch.manual_seed(123)
    scores = torch.randn(2, 3, 4)
    grad_alignment = torch.randn_like(scores)

    grad_scores, grad_open, grad_ext, grad_temp = sw_affine_ops.marginals_backward(
        scores, grad_alignment, -2.0, -0.5, 1.0, None
    )

    assert grad_scores.shape == scores.shape
    assert grad_open.shape == (1,)
    assert grad_ext.shape == (1,)
    assert grad_temp.shape == (1,)


# CUDA memory-safety regression helpers/constants
GAP_OPEN = -2.0
GAP_EXT = -0.5
TEMPERATURE = 1.0


def _small_cuda_inputs(device="cuda"):
    scores = seeded_randn((2, 3, 4), 770, device)
    tangent = seeded_randn(tuple(scores.shape), 771, device)
    grad_alignment = seeded_randn(tuple(scores.shape), 772, device)
    lengths = torch.tensor([[3, 4], [2, 3]], dtype=torch.int32, device=device)
    return scores, tangent, grad_alignment, lengths


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_tensors(item)


def cuda_kernel_names(call):
    try:
        from torch.profiler import ProfilerActivity, profile
    except (ImportError, AttributeError) as exc:
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


def assert_cuda_kernel_seen(kernel_names, token):
    if any(token in name for name in kernel_names):
        return
    relevant = sorted(name for name in kernel_names if "sw_affine_" in name)
    raise AssertionError(
        f"CUDA profiler did not capture {token}; sw_affine kernels seen: {relevant[:20]}"
    )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_oversized_alpha_table_rejected_before_allocation():
    too_wide = torch.iinfo(torch.int32).max // 3
    scores = torch.empty((1, 0, too_wide), device="cuda")

    with pytest.raises(RuntimeError, match="sw_affine DP table too large"):
        sw_affine_forward_with_grads(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_full_rejects_grad_alignment_shape_mismatch():
    scores, _, _, lengths = _small_cuda_inputs()
    bad_grad_alignment = torch.randn(2, 3, 3, device=scores.device)

    with pytest.raises(RuntimeError, match="cotangent must have same shape as scores"):
        sw_affine_ops.marginals_backward(
            scores, bad_grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_full_rejects_cpu_grad_alignment_for_cuda_scores():
    scores, _, _, lengths = _small_cuda_inputs()
    bad_grad_alignment = torch.randn(scores.shape)

    with pytest.raises(RuntimeError, match="cotangent must be on same device as scores"):
        sw_affine_ops.marginals_backward(
            scores, bad_grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
        )


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_full_rejects_cross_device_grad_alignment():
    scores, _, _, lengths = _small_cuda_inputs("cuda:0")
    bad_grad_alignment = seeded_randn(tuple(scores.shape), 773, "cuda:1")
    assert scores.device.index != bad_grad_alignment.device.index

    with pytest.raises(RuntimeError, match="cotangent must be on same device as scores"):
        sw_affine_ops.marginals_backward(
            scores, bad_grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
        )


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.parametrize(
    "entry",
    [
        "sw_affine_float",
        "sw_affine_with_grads",
        "sw_affine_hvp",
        "sw_affine_param_jacobian",
        "sw_affine_backward_full",
        "sw_affine_forward",
        "sw_affine_marginals_hvp",
        "sw_affine_marginals_backward",
    ],
)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_entries_run_on_scores_device_when_current_device_differs(entry):
    original_device = torch.cuda.current_device()
    try:
        scores, tangent, grad_alignment, lengths = _small_cuda_inputs("cuda:0")
        torch.cuda.set_device(1)
        assert scores.device.index == 0
        assert scores.device.index != torch.cuda.current_device()

        if entry == "sw_affine_float":
            result = sw_affine_ops.forward(
                scores, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_with_grads":
            result = sw_affine_forward_with_grads(
                scores, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_hvp":
            result = sw_affine_ops.marginals_hvp(
                scores, tangent, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_param_jacobian":
            result = sw_affine_param_field(
                scores, 0, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_backward_full":
            result = sw_affine_ops.marginals_backward(
                scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_forward":
            result = sw_affine_ops.forward(
                scores, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_marginals_hvp":
            result = sw_affine_ops.marginals_hvp(
                scores, tangent, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        elif entry == "sw_affine_marginals_backward":
            result = sw_affine_ops.marginals_backward(
                scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
            )
        else:
            raise AssertionError(f"unhandled entry {entry}")

        torch.cuda.synchronize(torch.device("cuda:0"))
        assert torch.cuda.current_device() == 1
        for tensor in _flatten_tensors(result):
            assert tensor.device == scores.device
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_tensor_parameters_use_scores_device_when_current_device_differs():
    original_device = torch.cuda.current_device()
    try:
        scores = seeded_randn((2, 3, 4), 774, "cuda:0").requires_grad_(True)
        lengths = torch.tensor(
            [[3, 4], [2, 3]], dtype=torch.int32, device="cuda:0"
        )
        gap_open = torch.tensor([-2.0], device="cuda:0", requires_grad=True)
        gap_ext = torch.tensor([-0.5], device="cuda:0", requires_grad=True)
        temperature = torch.tensor([1.0], device="cuda:0", requires_grad=True)
        torch.cuda.set_device(1)

        assert scores.device.index == 0
        assert torch.cuda.current_device() == 1
        map_result = orihime.sw_affine(
            scores,
            gap_open_score=gap_open,
            gap_extend_score=gap_ext,
            temperature=temperature,
            lengths=lengths,
        )
        value_result = orihime.sw_affine_value(
            scores,
            gap_open_score=gap_open,
            gap_extend_score=gap_ext,
            temperature=temperature,
            lengths=lengths,
        )
        (map_result.sum() + value_result.sum()).backward()
        torch.cuda.synchronize(scores.device)

        assert map_result.device == scores.device
        assert value_result.device == scores.device
        assert scores.grad is not None
        assert scores.grad.device == scores.device
        for parameter in (gap_open, gap_ext, temperature):
            assert parameter.grad is not None
            assert parameter.grad.device == scores.device
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.parametrize(
    "parameter_name",
    ["gap_open_score", "gap_extend_score", "temperature"],
)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_tensor_parameters_reject_wrong_cuda_device(parameter_name):
    scores = seeded_randn((1, 3, 4), 775, "cuda:0")
    lengths = torch.tensor([[3, 4]], dtype=torch.int32, device="cuda:0")
    kwargs = {
        "gap_open_score": -2.0,
        "gap_extend_score": -0.5,
        "temperature": 1.0,
    }
    kwargs[parameter_name] = torch.tensor([kwargs[parameter_name]], device="cuda:1")
    assert scores.device.index != kwargs[parameter_name].device.index

    with pytest.raises(
        ValueError,
        match=rf"{parameter_name} tensor must be on the same device",
    ):
        orihime.sw_affine(scores, lengths=lengths, **kwargs)


@pytest.mark.skipif(not (ORIHIME_AVAILABLE and CUDA_AVAILABLE), reason="CUDA orihime build required")
def test_score_backward_skips_unused_posterior_hvp():
    def score_only_backward():
        scores = seeded_randn((1, 8, 9), 776, "cuda").requires_grad_(True)
        score, _ = sw_affine_ops.forward(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )
        score.sum().backward()

    def explicit_zero_posterior_backward():
        scores = seeded_randn((1, 8, 9), 777, "cuda").requires_grad_(True)
        score, posteriors = sw_affine_ops.forward(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )
        (score.sum() + 0.0 * posteriors.sum()).backward()

    score_only_kernels = cuda_kernel_names(score_only_backward)
    assert_cuda_kernel_seen(score_only_kernels, "sw_affine_forward_diag_kernel")
    assert not any("sw_affine_hvp_" in name for name in score_only_kernels)
    assert not any("sw_affine_param_grad_" in name for name in score_only_kernels)

    explicit_zero_kernels = cuda_kernel_names(explicit_zero_posterior_backward)
    assert_cuda_kernel_seen(
        explicit_zero_kernels, "sw_affine_hvp_forward_diag_kernel"
    )
    assert_cuda_kernel_seen(
        explicit_zero_kernels, "sw_affine_param_grad_forward_diag_kernel"
    )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_valid_backward_full_cuda_still_runs():
    scores, _, grad_alignment, lengths = _small_cuda_inputs()

    grad_scores, grad_open, grad_ext, grad_temp = sw_affine_ops.marginals_backward(
        scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
    )

    assert grad_scores.shape == scores.shape
    assert grad_open.shape == (1,)
    assert grad_ext.shape == (1,)
    assert grad_temp.shape == (1,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
