# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft Eisner (Projective Dependency Parsing).
"""

import contextlib
import tempfile

import pytest
import torch

from reference import eisner_forward_naive, eisner_naive
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

try:
    import orihime
    eisner_ops = orihime.ops._kernels["eisner"]
    from operator_test_utils import (
        eisner_forward_with_grads,
        eisner_full_outputs,
    )
    ORIHIME_AVAILABLE = True
except ImportError:
    ORIHIME_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def assert_padded_chart_zero(name, tensor, lengths):
    tensor_cpu = tensor.detach().cpu()
    lengths_cpu = lengths.detach().cpu()
    for batch, sequence_length in enumerate(lengths_cpu.tolist()):
        if sequence_length < tensor_cpu.size(1):
            trailing_rows = tensor_cpu[batch, sequence_length:, :]
            trailing_cols = tensor_cpu[batch, :sequence_length, sequence_length:]
            assert torch.count_nonzero(trailing_rows).item() == 0, \
                f"{name} batch {batch} wrote past length={sequence_length} in rows"
            assert torch.count_nonzero(trailing_cols).item() == 0, \
                f"{name} batch {batch} wrote past length={sequence_length} in columns"


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(arc_scores, tangent, temperature):
    partition, marginals, grad_T = eisner_full_outputs(arc_scores, temperature, None)
    hvp = eisner_ops.marginals_hvp(arc_scores, tangent, temperature, None)
    dP_dT = eisner_ops.marginals_grad_temp(arc_scores, temperature, None)
    return {
        "partition": partition,
        "marginals": marginals,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_eisner_correctness(outputs, reference_outputs, thread_count):
    partition_ref = reference_outputs["partition"]
    marginals_ref = reference_outputs["marginals"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(partition_ref, outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(partition_ref, outputs['partition'])}"
    assert allclose(marginals_ref, outputs["marginals"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread marginals mismatch: max diff = {max_diff(marginals_ref, outputs['marginals'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dT_ref, outputs["dP_dT"], rtol=1e-2, atol=1e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(dP_dT_ref, outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 8, 12])
def seq_length(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestForward:

    def test_partition(self, batch_size, seq_length, temperature, device):
        """Test that partition functions match."""
        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)

        partition_ref, _, _ = eisner_forward_naive(arc_scores, temperature)
        partition_orihime, _ = eisner_forward_with_grads(arc_scores, temperature, None)

        assert allclose(partition_ref, partition_orihime), \
            f"Partition mismatch: max diff = {max_diff(partition_ref, partition_orihime)}"

    def test_partition_positive_scores(self, batch_size, device):
        """Test with positive arc scores."""
        seq_length = 6
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device).abs()

        partition_ref, _, _ = eisner_forward_naive(arc_scores, temperature)
        partition_orihime, _ = eisner_forward_with_grads(arc_scores, temperature, None)

        assert allclose(partition_ref, partition_orihime), \
            f"Partition mismatch with positive scores: max diff = {max_diff(partition_ref, partition_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_arc_marginals(self, batch_size, seq_length, temperature, device):
        """Test that arc marginals match."""
        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)

        marginals_ref = eisner_naive(arc_scores, temperature)
        _, marginals_orihime = eisner_forward_with_grads(arc_scores, temperature, None)

        assert allclose(marginals_ref, marginals_orihime, rtol=1e-3, atol=1e-4), \
            f"Arc marginals mismatch: max diff = {max_diff(marginals_ref, marginals_orihime)}"

    def test_gradients(self, batch_size, device):
        """Test gradients through the soft Eisner."""
        seq_length = 6
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)
        arc_scores.requires_grad_(True)

        partition = orihime.eisner_value(
            arc_scores,
            temperature=temperature,
        )
        loss = partition.sum()
        loss.backward()
        grad_orihime = arc_scores.grad.clone()

        arc_scores_ref = arc_scores.detach().clone().requires_grad_(True)
        partition_ref, _, _ = eisner_forward_naive(arc_scores_ref, temperature)
        loss_ref = partition_ref.sum()
        loss_ref.backward()
        grad_ref = arc_scores_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"

    def test_marginal_sum(self, batch_size, seq_length, device):
        """Test that arc marginals for each dependent sum to ~1."""
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)

        _, marginals = eisner_forward_with_grads(arc_scores, temperature, None)

        # For each word j (except root at 0), sum of incoming arcs should be ~1
        # marginals[:, :, j].sum(dim=1) ~= 1 for j > 0
        for j in range(1, seq_length):
            incoming_sum = marginals[:, :, j].sum(dim=1)
            expected = torch.ones(batch_size, device=device)
            assert allclose(incoming_sum, expected, rtol=1e-2, atol=1e-2), \
                f"Incoming arc sum for word {j} should be ~1, got {incoming_sum}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, n = 2, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)
        V = torch.randn(B, n, n, device=device)

        hvp_orihime = eisner_ops.marginals_hvp(arc_scores, V, temperature, None)

        marginals_plus = eisner_naive(arc_scores + eps * V, temperature)
        marginals_minus = eisner_naive(arc_scores - eps * V, temperature)
        hvp_fd = (marginals_plus - marginals_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"

    def test_hvp_various_temps(self, device):
        """Test HVP with various temperatures."""
        B, n = 2, 6
        eps = 1e-4

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            arc_scores = torch.randn(B, n, n, device=device)
            V = torch.randn(B, n, n, device=device)

            hvp_orihime = eisner_ops.marginals_hvp(arc_scores, V, temperature, None)

            marginals_plus = eisner_naive(arc_scores + eps * V, temperature)
            marginals_minus = eisner_naive(arc_scores - eps * V, temperature)
            hvp_fd = (marginals_plus - marginals_minus) / (2 * eps)

            assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=5e-3), \
                f"HVP mismatch for temp={temperature}: max diff = {max_diff(hvp_fd, hvp_orihime)}"

    def test_full_vjp_matches_reference(self):
        batch_size, seq_length = 2, 5
        temperature = 0.85

        torch.manual_seed(8042)
        arc_scores = torch.randn(batch_size, seq_length, seq_length)
        grad_marginals = torch.randn_like(arc_scores)

        grad_arc_scores, grad_temp = eisner_ops.marginals_backward(
            arc_scores,
            grad_marginals,
            temperature,
            None,
        )

        arc_scores_ref = arc_scores.detach().clone().requires_grad_(True)
        temp_ref = torch.tensor([temperature], requires_grad=True)
        marginals_ref = eisner_naive(arc_scores_ref, temp_ref)
        grad_arc_ref, grad_temp_ref = torch.autograd.grad(
            (marginals_ref * grad_marginals).sum(),
            (arc_scores_ref, temp_ref),
        )

        assert allclose(grad_arc_ref, grad_arc_scores, rtol=2e-2, atol=4e-3), \
            f"Full VJP arc mismatch: max diff = {max_diff(grad_arc_ref, grad_arc_scores)}"
        assert allclose(grad_temp_ref, grad_temp, rtol=2e-2, atol=4e-3), \
            f"Full VJP temperature mismatch: max diff = {max_diff(grad_temp_ref, grad_temp)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, n = 8, 7
        temperature = 1.0
        hvp_eps = 1e-4
        param_eps = 1e-3
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        arc_scores = torch.randn(B, n, n)
        tangent = torch.randn(B, n, n)

        with torch_num_threads(1):
            partition_ref, _, _ = eisner_forward_naive(arc_scores, temperature)
            marginals_ref = eisner_naive(arc_scores, temperature)

            partition_plus, _, _ = eisner_forward_naive(arc_scores, temperature + hvp_eps)
            partition_minus, _, _ = eisner_forward_naive(arc_scores, temperature - hvp_eps)
            grad_T_ref = (partition_plus - partition_minus) / (2 * hvp_eps)

            marginals_plus = eisner_naive(arc_scores + hvp_eps * tangent, temperature)
            marginals_minus = eisner_naive(arc_scores - hvp_eps * tangent, temperature)
            hvp_ref = (marginals_plus - marginals_minus) / (2 * hvp_eps)

            marginals_temp_plus = eisner_naive(arc_scores, temperature + param_eps)
            marginals_temp_minus = eisner_naive(arc_scores, temperature - param_eps)
            dP_dT_ref = (marginals_temp_plus - marginals_temp_minus) / (2 * param_eps)

        reference_outputs = {
            "partition": partition_ref,
            "marginals": marginals_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(arc_scores, tangent, temperature)
            assert_threaded_eisner_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, n = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores_cpu = torch.randn(B, n, n)
        arc_scores_cuda = arc_scores_cpu.cuda()

        _, marginals_cpu = eisner_forward_with_grads(arc_scores_cpu, temperature, None)
        _, marginals_cuda = eisner_forward_with_grads(arc_scores_cuda, temperature, None)

        assert allclose(marginals_cpu, marginals_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(marginals_cpu, marginals_cuda)}"

    def test_consistency_partition(self):
        """Test CPU vs CUDA partition functions match."""
        B, n = 2, 10
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores_cpu = torch.randn(B, n, n)
        arc_scores_cuda = arc_scores_cpu.cuda()

        partition_cpu, _ = eisner_forward_with_grads(arc_scores_cpu, temperature, None)
        partition_cuda, _ = eisner_forward_with_grads(arc_scores_cuda, temperature, None)

        assert allclose(partition_cpu, partition_cuda), \
            f"CPU/CUDA partition mismatch: max diff = {max_diff(partition_cpu, partition_cuda)}"

    def test_derivative_consistency(self):
        B, n = 2, 7
        temperature = 0.9

        torch.manual_seed(5109)
        arc_scores_cpu = torch.randn(B, n, n)
        tangent_cpu = torch.randn_like(arc_scores_cpu)
        grad_marginals_cpu = torch.randn_like(arc_scores_cpu)

        def raw_outputs(device):
            arc_scores = arc_scores_cpu.to(device)
            tangent = tangent_cpu.to(device)
            grad_marginals = grad_marginals_cpu.to(device)
            value, marginals = eisner_forward_with_grads(
                arc_scores, temperature, None
            )
            grad_temp = eisner_ops.value_grad_params(
                arc_scores, temperature, None
            )
            hvp = eisner_ops.marginals_hvp(
                arc_scores, tangent, temperature, None
            )
            dP_dT = eisner_ops.marginals_grad_temp(
                arc_scores, temperature, None
            )
            full_vjp = eisner_ops.marginals_backward(
                arc_scores, grad_marginals, temperature, None
            )
            return (value, marginals, grad_temp, hvp, dP_dT, *full_vjp)

        cpu_outputs = raw_outputs("cpu")
        cuda_outputs = raw_outputs("cuda")
        for cpu_output, cuda_output in zip(cpu_outputs, cuda_outputs):
            assert allclose(cpu_output, cuda_output, rtol=2e-3, atol=2e-4), \
                f"CPU/CUDA derivative mismatch: max diff = {max_diff(cpu_output, cuda_output)}"

        def autograd_outputs(device):
            arc_scores = arc_scores_cpu.to(device).requires_grad_(True)
            temp = torch.tensor([temperature], device=device, requires_grad=True)
            value, marginals = eisner_ops.forward_t(arc_scores, temp, None)
            loss = value.sum() + 0.25 * marginals.square().sum()
            return torch.autograd.grad(loss, (arc_scores, temp))

        cpu_autograd = autograd_outputs("cpu")
        cuda_autograd = autograd_outputs("cuda")
        for cpu_output, cuda_output in zip(cpu_autograd, cuda_autograd):
            assert allclose(cpu_output, cuda_output, rtol=2e-3, atol=2e-4), \
                f"CPU/CUDA autograd mismatch: max diff = {max_diff(cpu_output, cuda_output)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sentence lengths in batch."""
        B = 4
        max_n = 10
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, max_n, max_n, device=device)

        # Variable lengths
        lengths = torch.tensor([8, 10, 6, 9], device=device, dtype=torch.int32)

        partition, marginals = eisner_forward_with_grads(arc_scores, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            n = lengths[b].item()
            arc_scores_b = arc_scores[b:b+1, :n, :n]

            partition_ref, _, _ = eisner_forward_naive(arc_scores_b, temperature)
            marginals_ref = eisner_naive(arc_scores_b, temperature)

            assert allclose(partition_ref, partition[b:b+1]), \
                f"Partition mismatch for batch {b}: {partition_ref.item()} vs {partition[b].item()}"

            assert allclose(marginals_ref, marginals[b:b+1, :n, :n], rtol=1e-3, atol=1e-4), \
                f"Marginals mismatch for batch {b}"

    def test_variable_lengths_zero_padded_outputs_and_derivatives(self, device):
        B, max_n = 3, 7
        temperature = 0.9
        lengths = torch.tensor([7, 4, 2], dtype=torch.int32, device=device)

        torch.manual_seed(8043)
        arc_scores = torch.randn(B, max_n, max_n, device=device)
        tangent = torch.randn_like(arc_scores)
        grad_marginals = torch.randn_like(arc_scores)

        partition, marginals = eisner_ops.forward(
            arc_scores, temperature, lengths
        )
        hvp = eisner_ops.marginals_hvp(
            arc_scores, tangent, temperature, lengths
        )
        grad_arc_scores, _ = eisner_ops.marginals_backward(
            arc_scores, grad_marginals, temperature, lengths
        )
        dP_dT = eisner_ops.marginals_grad_temp(
            arc_scores, temperature, lengths
        )
        value_grad_temp = eisner_ops.value_grad_params(
            arc_scores, temperature, lengths
        )

        assert partition.shape == (B,)
        assert torch.isfinite(partition).all()
        for name, tensor in (
            ("marginals", marginals),
            ("HVP", hvp),
            ("full VJP arc gradient", grad_arc_scores),
            ("temperature sensitivity", dP_dT),
        ):
            assert torch.isfinite(tensor).all()
            assert_padded_chart_zero(name, tensor, lengths)
        assert torch.isfinite(value_grad_temp).all()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestValidation:

    @pytest.mark.parametrize(
        "device_type",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_non_square_arc_scores_raise(self, device_type):
        """Eisner charts must stay square."""
        device = torch.device(device_type)
        arc_scores = torch.randn(1, 4, 5, device=device)

        with pytest.raises(RuntimeError, match=r"arc_scores must be \[B, n, n\]"):
            eisner_ops.forward(arc_scores, 1.0, None)

    @pytest.mark.parametrize(
        "device_type",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_zero_length_chart_raises(self, device_type):
        """An empty Eisner chart must be rejected before kernel launch."""
        device = torch.device(device_type)
        arc_scores = torch.empty(1, 0, 0, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError, match=r"eisner requires n > 0"):
            eisner_ops.forward(arc_scores, 1.0, None)

    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [0], r"lengths\[0\] must be between 1 and 4"),
            ("cpu", [5], r"lengths\[0\] must be between 1 and 4"),
            pytest.param(
                "cuda",
                [0],
                r"lengths\[0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [5],
                r"lengths\[0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_invalid_lengths_raise(self, device_type, lengths, match):
        """Explicit lengths must stay within the padded chart shape."""
        device = torch.device(device_type)
        arc_scores = torch.randn(1, 4, 4, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            eisner_ops.forward(arc_scores, 1.0, lengths_t)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_lengths_must_match_scores_device(self):
        """Lengths must stay on the same device as the score chart."""
        cuda_scores = torch.randn(1, 4, 4, device="cuda")
        cpu_lengths = torch.tensor([4], dtype=torch.int32)

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            eisner_ops.forward(cuda_scores, 1.0, cpu_lengths)

    @pytest.mark.parametrize(
        "device_type",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_hvp_shape_mismatch_raises(self, device_type):
        """The HVP tangent must stay shape-aligned with arc_scores."""
        device = torch.device(device_type)
        arc_scores = torch.randn(1, 4, 4, device=device)
        tangent = torch.randn(1, 4, 5, device=device)

        with pytest.raises(RuntimeError, match=r"tangent must have same shape as arc_scores"):
            eisner_ops.marginals_hvp(arc_scores, tangent, 1.0, None)

    @pytest.mark.parametrize(
        "bad_input",
        ("arc_scores", "lengths", "tangent", "grad_marginals"),
    )
    @pytest.mark.parametrize(
        "device_type",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_noncontiguous_chart_and_derivative_inputs_reject_cleanly(
        self, device_type, bad_input
    ):
        device = torch.device(device_type)
        arc_scores = torch.randn(2, 4, 4, device=device)
        lengths = torch.tensor([4, 4], dtype=torch.int32, device=device)
        tangent = torch.randn_like(arc_scores)
        grad_marginals = torch.randn_like(arc_scores)

        if bad_input == "arc_scores":
            bad_arc_scores = arc_scores.transpose(1, 2)
            call = lambda: eisner_ops.forward(bad_arc_scores, 1.0, None)
        elif bad_input == "lengths":
            length_storage = torch.tensor([4, 4, 4, 4], dtype=torch.int32, device=device)
            bad_lengths = length_storage[::2]
            call = lambda: eisner_ops.forward(arc_scores, 1.0, bad_lengths)
        elif bad_input == "tangent":
            bad_tangent = tangent.transpose(1, 2)
            call = lambda: eisner_ops.marginals_hvp(
                arc_scores, bad_tangent, 1.0, None
            )
        else:
            bad_grad_marginals = grad_marginals.transpose(1, 2)
            call = lambda: eisner_ops.marginals_backward(
                arc_scores, bad_grad_marginals, 1.0, None
            )

        expected_match = {
            "arc_scores": r"arc_scores must be contiguous",
            "lengths": r"lengths must be contiguous",
            "tangent": r"tangent must be contiguous",
            "grad_marginals": r"cotangent must be contiguous",
        }[bad_input]
        with pytest.raises(RuntimeError, match=expected_match):
            call()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestEdgeCases:

    def test_two_words(self, device):
        """Test with 2-word sentence (simplest non-trivial case)."""
        arc_scores = torch.tensor([[[0.0, 0.5], [-0.3, 0.0]]], device=device)
        temperature = 1.0

        partition, marginals = eisner_forward_with_grads(arc_scores, temperature, None)
        partition_ref, _, _ = eisner_forward_naive(arc_scores, temperature)
        marginals_ref = eisner_naive(arc_scores, temperature)

        assert allclose(partition, partition_ref), \
            f"Two words partition wrong: {partition.item()} vs {partition_ref.item()}"
        assert allclose(marginals, marginals_ref), \
            "Two words marginals mismatch"

    def test_three_words(self, device):
        """Test with 3-word sentence."""
        torch.manual_seed(42)
        arc_scores = torch.randn(1, 3, 3, device=device)
        temperature = 1.0

        partition, marginals = eisner_forward_with_grads(arc_scores, temperature, None)
        partition_ref, _, _ = eisner_forward_naive(arc_scores, temperature)
        marginals_ref = eisner_naive(arc_scores, temperature)

        assert allclose(partition, partition_ref), \
            "Three words partition mismatch"
        assert allclose(marginals, marginals_ref, rtol=1e-3), \
            "Three words marginals mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard parsing)."""
        B, n = 2, 6
        temperature = 0.01

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        _, marginals = eisner_forward_with_grads(arc_scores, temperature, None)

        # With low temperature, marginals should be close to 0 or 1
        assert marginals.min() >= -0.1, "Low temp marginals should be >= 0"
        assert marginals.max() <= 1.1, "Low temp marginals should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, n = 2, 6
        temperature = 10.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        _, marginals = eisner_forward_with_grads(arc_scores, temperature, None)
        marginals_ref = eisner_naive(arc_scores, temperature)

        assert allclose(marginals_ref, marginals, rtol=1e-3, atol=1e-4), \
            "High temperature marginals mismatch"

    def test_diagonal_zeros(self, device):
        """Test that self-loop marginals are zero."""
        B, n = 2, 6
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        _, marginals = eisner_forward_with_grads(arc_scores, temperature, None)

        # Self-loops (diagonal) should have zero probability
        diag_marginals = torch.diagonal(marginals, dim1=1, dim2=2)
        assert allclose(diag_marginals, torch.zeros_like(diag_marginals), atol=1e-5), \
            f"Self-loop marginals should be 0, got max {diag_marginals.abs().max().item()}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestWithGrads:

    def test_with_grads_output(self, device):
        """Test eisner_with_grads returns correct values."""
        B, n = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        partition, marginals = eisner_forward_with_grads(arc_scores, temperature, None)

        # Compare with reference
        partition_ref, _, _ = eisner_forward_naive(arc_scores, temperature)
        marginals_ref = eisner_naive(arc_scores, temperature)

        assert allclose(partition, partition_ref), "with_grads partition mismatch"
        assert allclose(marginals, marginals_ref, rtol=1e-3), "with_grads marginals mismatch"

    def test_backward_full_output(self, device):
        """Test eisner_backward_full returns correct values."""
        B, n = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        partition, marginals, grad_T = eisner_full_outputs(arc_scores, temperature, None)

        # Compare with eisner_with_grads
        partition_ref, marginals_ref = eisner_forward_with_grads(arc_scores, temperature, None)

        assert allclose(partition, partition_ref), "backward_full partition mismatch"
        assert allclose(marginals, marginals_ref), "backward_full marginals mismatch"
        assert grad_T.shape == (B,), f"grad_T shape wrong: {grad_T.shape}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestParamJacobian:

    def test_value_temperature_gradient_matches_finite_difference(self):
        B, n = 2, 6
        temperature = 0.9
        eps = 1e-3

        torch.manual_seed(8045)
        arc_scores = torch.randn(B, n, n)
        grad_temp = eisner_ops.value_grad_params(
            arc_scores, temperature, None
        )
        partition_plus = eisner_ops.forward(
            arc_scores, temperature + eps, None
        )[0]
        partition_minus = eisner_ops.forward(
            arc_scores, temperature - eps, None
        )[0]
        grad_temp_fd = (partition_plus - partition_minus) / (2 * eps)

        assert allclose(grad_temp_fd, grad_temp, rtol=2e-2, atol=2e-3), \
            f"Value temperature gradient mismatch: max diff = {max_diff(grad_temp_fd, grad_temp)}"

    def test_param_jacobian_temperature(self):
        """Test parameter Jacobian for temperature against finite differences (CPU)."""
        B, n = 2, 6
        temperature = 1.0
        eps = 1e-3

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n)

        # dP/dT via param_jacobian
        dP_dT = eisner_ops.marginals_grad_temp(arc_scores, temperature, None)

        # Finite diff validation
        marginals_plus = eisner_naive(arc_scores, temperature + eps)
        marginals_minus = eisner_naive(arc_scores, temperature - eps)
        dP_dT_fd = (marginals_plus - marginals_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=1e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    def test_param_jacobian_various_temps(self):
        """Test parameter Jacobian at various temperatures (CPU)."""
        B, n = 2, 6
        eps = 1e-3

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            arc_scores = torch.randn(B, n, n)

            dP_dT = eisner_ops.marginals_grad_temp(arc_scores, temperature, None)

            marginals_plus = eisner_naive(arc_scores, temperature + eps)
            marginals_minus = eisner_naive(arc_scores, temperature - eps)
            dP_dT_fd = (marginals_plus - marginals_minus) / (2 * eps)

            assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=1e-3), \
                f"Param Jacobian mismatch for temp={temperature}: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    def test_param_jacobian_shape(self):
        """Test that param_jacobian output shape matches arc_scores."""
        B, n = 3, 8
        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n)

        dP_dT = eisner_ops.marginals_grad_temp(arc_scores, 1.0, None)
        assert dP_dT.shape == arc_scores.shape, \
            f"Shape mismatch: {dP_dT.shape} vs {arc_scores.shape}"


# --- Memory-safety regression tests (merged from test_eisner_{cpp,cuda}_memsafety.py) ---

import re
from pathlib import Path


BAD_N = 46341
MAX_SAFE_N = 46340
OVERFLOW_MATCH = rf"eisner requires n <= {MAX_SAFE_N}"


def _huge_logical_cpu_scores():
    """A logical [1, 46341, 46341] tensor backed by one float of storage."""
    storage = torch.empty(1, dtype=torch.float32)
    scores = torch.as_strided(storage, (1, BAD_N, BAD_N), (0, 0, 0))
    assert scores.numel() > 2**31
    assert storage.numel() == 1
    return scores


@contextlib.contextmanager
def _file_backed_contiguous_cpu_tangent():
    """Yield a contiguous oversized tangent without materializing its values."""
    nbytes = BAD_N * BAD_N * 4
    backing_file = tempfile.NamedTemporaryFile()
    try:
        backing_file.truncate(nbytes)
        storage = torch.UntypedStorage.from_file(
            backing_file.name,
            shared=False,
            nbytes=nbytes,
        )
        tangent = torch.empty(0, dtype=torch.float32).set_(
            storage,
            0,
            (1, BAD_N, BAD_N),
            (BAD_N * BAD_N, BAD_N, 1),
        )
        assert tangent.is_contiguous()
        yield tangent
    finally:
        backing_file.close()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.parametrize(
    "entrypoint",
    [
        lambda scores: eisner_ops.forward_t(
            scores, torch.tensor([1.0]), None
        ),
        lambda scores: eisner_ops.forward(scores, 1.0, None),
        lambda scores: eisner_forward_with_grads(scores, 1.0, None),
        lambda scores: eisner_full_outputs(scores, 1.0, None),
        lambda scores: eisner_ops.marginals_grad_temp(scores, 1.0, None),
    ],
)
def test_cpu_entrypoints_reject_int_overflow_chart_before_kernel(entrypoint):
    scores = _huge_logical_cpu_scores()

    with pytest.raises(RuntimeError, match=OVERFLOW_MATCH):
        entrypoint(scores)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_hvp_rejects_int_overflow_chart_before_kernel():
    scores = _huge_logical_cpu_scores()

    with _file_backed_contiguous_cpu_tangent() as tangent:
        with pytest.raises(RuntimeError, match=OVERFLOW_MATCH):
            eisner_ops.marginals_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_valid_cpu_inputs_still_run():
    torch.manual_seed(20260707)
    scores = torch.randn(2, 5, 5)
    tangent = torch.randn_like(scores)
    lengths = torch.tensor([5, 3], dtype=torch.int32)

    partition_float = eisner_ops.forward(scores, 1.0, lengths)[0]
    partition, marginals = eisner_forward_with_grads(scores, 1.0, lengths)
    partition_full, marginals_full, grad_T = eisner_full_outputs(scores, 1.0, lengths)
    hvp = eisner_ops.marginals_hvp(scores, tangent, 1.0, lengths)
    dP_dT = eisner_ops.marginals_grad_temp(scores, 1.0, lengths)

    assert torch.allclose(partition_float, partition)
    assert torch.allclose(partition, partition_full)
    assert torch.allclose(marginals, marginals_full)
    assert partition.shape == (2,)
    assert marginals.shape == scores.shape
    assert grad_T.shape == (2,)
    assert hvp.shape == scores.shape
    assert dP_dT.shape == scores.shape


def test_eisner_cpu_kernel_cell_offsets_are_widened():
    source = Path(__file__).resolve().parents[1] / "src" / "eisner" / "kernels_cpu.cpp"
    text = source.read_text()

    for forbidden in ("i * n +", "j * n +", "k * n +", "(k + 1) * n +", "0 * n +"):
        assert forbidden not in text
    assert "static_cast<size_t>(row) * static_cast<size_t>(n)" in text


CUDA_RUNTIME_REQUIRED = pytest.mark.skipif(
    not (ORIHIME_AVAILABLE and CUDA_AVAILABLE),
    reason="Eisner CUDA memory-safety regression needs the CUDA orihime backend",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_CUDA_CPP = REPO_ROOT / "src" / "eisner" / "torch_cuda.cpp"
KERNELS_CU = REPO_ROOT / "src" / "eisner" / "kernels_gpu.cu"


def _read_source(path):
    return path.read_text(encoding="utf-8")


def _call_eisner_entrypoint(name, scores, lengths=None):
    tangent = torch.randn_like(scores)
    grad_marginals = torch.randn_like(scores)
    temp_t = torch.tensor([1.0], dtype=torch.float32, device=scores.device)
    calls = {
        "eisner": lambda: eisner_ops.forward_t(scores, temp_t, lengths),
        "eisner_float": lambda: eisner_ops.forward(scores, 1.0, lengths),
        "eisner_with_grads": lambda: eisner_forward_with_grads(scores, 1.0, lengths),
        "eisner_hvp": lambda: eisner_ops.marginals_hvp(scores, tangent, 1.0, lengths),
        "eisner_marginals_backward": lambda: eisner_ops.marginals_backward(
            scores, grad_marginals, 1.0, lengths
        ),
        "eisner_backward_full": lambda: eisner_full_outputs(scores, 1.0, lengths),
        "eisner_value_grad_params": lambda: eisner_ops.value_grad_params(
            scores, 1.0, lengths
        ),
        "eisner_param_jacobian": lambda: eisner_ops.marginals_grad_temp(scores, 1.0, lengths),
    }
    return calls[name]()


def _flatten_tensors(result):
    if isinstance(result, (tuple, list)):
        return result
    return (result,)


def test_cuda_autograd_disables_materialized_marginals_grad():
    source = _read_source(TORCH_CUDA_CPP)

    assert "ctx->set_materialize_grads(false);" in source


def test_cuda_wrappers_guard_arc_scores_device():
    source = _read_source(TORCH_CUDA_CPP)

    # forward, backward, and the five direct CUDA entrypoints all need a guard
    # whose lifetime covers allocation, recordStream, and kernel dispatch.
    assert source.count("ORIHIME_CUDA_GUARD(arc_scores);") >= 7


def test_cuda_chart_indices_use_size_t_helper():
    source = _read_source(KERNELS_CU)

    assert "__device__ __forceinline__ size_t chart_index" in source
    assert "return (size_t)row * (size_t)n + (size_t)col;" in source
    assert not re.search(r"\b[ijk]\s*\*\s*n\s*\+", source)
    assert not re.search(r"\(k\s*\+\s*1\)\s*\*\s*n\s*\+", source)


@CUDA_RUNTIME_REQUIRED
@pytest.mark.parametrize(
    "entrypoint",
    [
        "eisner",
        "eisner_float",
        "eisner_with_grads",
        "eisner_hvp",
        "eisner_marginals_backward",
        "eisner_backward_full",
        "eisner_value_grad_params",
        "eisner_param_jacobian",
    ],
)
def test_empty_cuda_batch_rejected_before_launch(entrypoint):
    scores = torch.empty((0, 4, 4), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match=r"eisner requires B > 0"):
        _call_eisner_entrypoint(entrypoint, scores)


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@CUDA_RUNTIME_REQUIRED
@pytest.mark.parametrize(
    "entrypoint",
    [
        "eisner",
        "eisner_float",
        "eisner_with_grads",
        "eisner_hvp",
        "eisner_marginals_backward",
        "eisner_backward_full",
        "eisner_value_grad_params",
        "eisner_param_jacobian",
    ],
)
def test_cuda_entrypoints_follow_arc_scores_device(entrypoint):
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(1, 4, 4, dtype=torch.float32, device="cuda:1")
        lengths = torch.tensor([4], dtype=torch.int32, device="cuda:1")
        assert scores.device.index != torch.cuda.current_device()

        result = _call_eisner_entrypoint(entrypoint, scores, lengths)

        for tensor in _flatten_tensors(result):
            assert tensor.device == scores.device
    finally:
        torch.cuda.set_device(previous_device)


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@CUDA_RUNTIME_REQUIRED
def test_cuda_autograd_backward_follows_arc_scores_device():
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(1, 4, 4, dtype=torch.float32, device="cuda:1", requires_grad=True)
        lengths = torch.tensor([4], dtype=torch.int32, device="cuda:1")
        temp = torch.tensor([1.0], dtype=torch.float32, device="cuda:1")
        assert scores.device.index != torch.cuda.current_device()

        partition, _ = eisner_ops.forward_t(scores, temp, lengths)
        partition.sum().backward()

        assert scores.grad is not None
        assert scores.grad.device == scores.device
    finally:
        torch.cuda.set_device(previous_device)


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@CUDA_RUNTIME_REQUIRED
def test_cuda_secondary_tensors_must_match_arc_scores_device():
    scores = torch.randn(1, 4, 4, dtype=torch.float32, device="cuda:0")
    other_device = torch.device("cuda:1")
    lengths = torch.tensor([4], dtype=torch.int32, device=other_device)
    tangent = torch.randn_like(scores, device=other_device)
    grad_marginals = torch.randn_like(scores, device=other_device)
    assert scores.device.index != other_device.index

    with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
        eisner_ops.forward(scores, 1.0, lengths)
    with pytest.raises(RuntimeError, match=r"tangent must be on same device as arc_scores"):
        eisner_ops.marginals_hvp(scores, tangent, 1.0, None)
    with pytest.raises(RuntimeError, match=r"cotangent must be on same device as arc_scores"):
        eisner_ops.marginals_backward(scores, grad_marginals, 1.0, None)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
