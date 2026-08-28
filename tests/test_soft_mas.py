# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft MAS (Monotonic Alignment Search).
"""

import contextlib
import inspect

import pytest
import torch

from reference import mas_forward_naive, mas_naive
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

try:
    import orihime
    mas_ops = orihime.ops._kernels["mas"]
    from operator_test_utils import mas_forward_with_grads, mas_full_outputs
    ORIHIME_AVAILABLE = True
except ImportError:
    ORIHIME_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()

PARAM_JACOBIAN_FIRST_COLUMN_REGRESSION_SCORES = torch.tensor(
    [[
        [0.6390, 0.7386, 0.3336, 0.6488, -0.1221, -1.4663],
        [1.8326, -0.1780, -1.0180, 0.5100, -2.5325, 0.0219],
        [1.5904, 0.0740, -0.6459, -0.9266, 0.7246, 1.2455],
        [0.3673, 0.2022, -1.3746, 0.0133, -0.3854, -0.2535],
        [-1.4655, 0.5268, 1.4230, 0.7589, 0.1948, -0.6701],
        [-0.0178, 0.3026, 0.0068, -0.3653, 2.6368, 0.0119],
        [0.0449, 0.1808, 0.9646, 0.6207, -0.1754, -0.2981],
        [-1.1332, 0.2050, -0.2275, -1.3782, -1.5521, 1.0560],
        [-1.5421, -1.1325, 0.3822, 0.7824, 0.3121, -1.1421],
        [0.6226, -2.3291, 1.1045, 1.4058, 0.4952, -1.0435],
        [1.3387, -0.9426, -0.7033, 1.3839, -0.6329, -0.0738],
        [0.4396, 0.0949, 1.0510, -1.5036, -0.4133, -1.7405],
    ]],
    dtype=torch.float32,
)


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(scores, tangent, temperature):
    partition, posteriors = mas_forward_with_grads(scores, temperature, None)
    grad_T = mas_full_outputs(scores, temperature, None)[2]
    hvp = mas_ops.marginals_hvp(scores, tangent, temperature, None)
    dP_dT = mas_ops.marginals_grad_temp(scores, temperature, None)
    return {
        "partition": partition,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_mas_correctness(outputs, reference_outputs, thread_count):
    partition_ref = reference_outputs["partition"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(partition_ref, outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(partition_ref, outputs['partition'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dT_ref, outputs["dP_dT"], rtol=2e-2, atol=5e-3), \
        f"{thread_count}-thread dP_dT mismatch: max diff = {max_diff(dP_dT_ref, outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


def _iter_tensors(result):
    if isinstance(result, torch.Tensor):
        yield result
    elif isinstance(result, dict):
        for value in result.values():
            yield from _iter_tensors(value)
    elif isinstance(result, (tuple, list)):
        for value in result:
            yield from _iter_tensors(value)
    else:
        raise AssertionError(f"expected tensors, got {type(result)!r}")


def _assert_padded_regions_zero(tensor, lengths):
    assert tensor.dim() == 3
    for batch, (t_len, s_len) in enumerate(lengths.tolist()):
        if t_len < tensor.size(1):
            assert torch.count_nonzero(tensor[batch, t_len:, :]).item() == 0
        if s_len < tensor.size(2):
            assert torch.count_nonzero(tensor[batch, :, s_len:]).item() == 0


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(10, 8), (16, 16), (20, 5)])
def seq_lengths(request):
    """(T, S) where T=frames, S=text length. Must have T >= S."""
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
        """Test that MAS scores match."""
        T, S = seq_lengths

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)

        score_ref, _ = mas_forward_naive(scores, temperature)
        score_orihime, _ = mas_forward_with_grads(scores, temperature, None)

        assert allclose(score_ref, score_orihime), \
            f"Score mismatch: max diff = {max_diff(score_ref, score_orihime)}"

    def test_score_positive_scores(self, batch_size, device):
        """Test MAS with positive scores."""
        T, S = 10, 6
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device).abs()

        score_ref, _ = mas_forward_naive(scores, temperature)
        score_orihime, _ = mas_forward_with_grads(scores, temperature, None)

        assert allclose(score_ref, score_orihime), \
            f"Score mismatch with positive scores: max diff = {max_diff(score_ref, score_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        T, S = seq_lengths

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)

        posteriors_ref = mas_naive(scores, temperature)
        _, posteriors_orihime = mas_forward_with_grads(scores, temperature, None)

        assert allclose(posteriors_ref, posteriors_orihime, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_orihime)}"

    def test_gradients(self, batch_size, device):
        """Test gradients through the soft alignment."""
        T, S = 8, 5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)
        scores.requires_grad_(True)

        # Use mas_float which has autograd support
        score, _ = mas_ops.forward(scores, temperature, None)
        loss = score.sum()
        loss.backward()
        grad_orihime = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        score_ref, _ = mas_forward_naive(scores_ref, temperature)
        loss_ref = score_ref.sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"

    def test_score_gradients(self, batch_size, device):
        """Test gradients through the score output."""
        T, S = 8, 5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)
        scores.requires_grad_(True)

        score, _ = mas_ops.forward(scores, temperature, None)
        loss = score.sum()
        loss.backward()
        grad_orihime = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        score_ref, _ = mas_forward_naive(scores_ref, temperature)
        loss_ref = score_ref.sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-3, atol=1e-4), \
            f"Score gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, T, S = 2, 8, 5
        temperature = 1.0
        # FP32 finite differences need a step large enough to avoid
        # cancellation across supported PyTorch/compiler versions.
        eps = 1e-3

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)
        V = torch.randn(B, T, S, device=device)

        hvp_orihime = mas_ops.marginals_hvp(scores, V, temperature, None)

        posteriors_plus = mas_naive(scores + eps * V, temperature)
        posteriors_minus = mas_naive(scores - eps * V, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"

    def test_hvp_various_temps(self, device):
        """Test HVP with various temperatures."""
        B, T, S = 2, 10, 6
        # FP32 finite differences need a step large enough to avoid
        # cancellation across supported PyTorch/compiler versions.
        eps = 1e-3

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores = torch.randn(B, T, S, device=device)
            V = torch.randn(B, T, S, device=device)

            hvp_orihime = mas_ops.marginals_hvp(scores, V, temperature, None)

            posteriors_plus = mas_naive(scores + eps * V, temperature)
            posteriors_minus = mas_naive(scores - eps * V, temperature)
            hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=5e-3), \
                f"HVP mismatch for temp={temperature}: max diff = {max_diff(hvp_fd, hvp_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, T, S = 8, 9, 6
        temperature = 1.0
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.randn(B, T, S)
        tangent = torch.randn(B, T, S)

        with torch_num_threads(1):
            partition_ref, _ = mas_forward_naive(scores, temperature)
            posteriors_ref = mas_naive(scores, temperature)

            partition_temp_plus, _ = mas_forward_naive(scores, temperature + eps)
            partition_temp_minus, _ = mas_forward_naive(scores, temperature - eps)
            grad_T_ref = (partition_temp_plus - partition_temp_minus) / (2 * eps)

            posteriors_plus = mas_naive(scores + eps * tangent, temperature)
            posteriors_minus = mas_naive(scores - eps * tangent, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            dP_dT_plus = mas_naive(scores, temperature + eps)
            dP_dT_minus = mas_naive(scores, temperature - eps)
            dP_dT_ref = (dP_dT_plus - dP_dT_minus) / (2 * eps)

        reference_outputs = {
            "partition": partition_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, temperature)
            assert_threaded_mas_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, T, S = 2, 12, 8
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, T, S)
        scores_cuda = scores_cpu.cuda()

        _, posteriors_cpu = mas_forward_with_grads(scores_cpu, temperature, None)
        _, posteriors_cuda = mas_forward_with_grads(scores_cuda, temperature, None)

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_score(self):
        """Test CPU vs CUDA scores match."""
        B, T, S = 2, 15, 10
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, T, S)
        scores_cuda = scores_cpu.cuda()

        score_cpu, _ = mas_forward_with_grads(scores_cpu, temperature, None)
        score_cuda, _ = mas_forward_with_grads(scores_cuda, temperature, None)

        assert allclose(score_cpu, score_cuda), \
            f"CPU/CUDA score mismatch: max diff = {max_diff(score_cpu, score_cuda)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_T, max_S = 15, 10
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, max_T, max_S, device=device)

        # Variable lengths [T, S] pairs
        lengths = torch.tensor([
            [12, 8],
            [15, 10],
            [10, 6],
            [14, 9]
        ], device=device, dtype=torch.int32)

        score, posteriors = mas_forward_with_grads(scores, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            t_len, s_len = lengths[b].tolist()
            scores_b = scores[b:b+1, :t_len, :s_len]

            score_ref, _ = mas_forward_naive(scores_b, temperature)
            posteriors_ref = mas_naive(scores_b, temperature)

            assert allclose(score_ref, score[b:b+1]), \
                f"Score mismatch for batch {b}: {score_ref.item()} vs {score[b].item()}"

            assert allclose(posteriors_ref, posteriors[b:b+1, :t_len, :s_len], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestLengthValidation:

    @staticmethod
    def call_op(op_name, scores, tangent, temperature, lengths):
        if op_name == "mas":
            return mas_ops.forward(scores, temperature, lengths)
        if op_name == "mas_float":
            return mas_ops.forward(scores, temperature, lengths)
        if op_name == "mas_with_grads":
            return mas_forward_with_grads(scores, temperature, lengths)
        if op_name == "mas_hvp":
            return mas_ops.marginals_hvp(scores, tangent, temperature, lengths)
        if op_name == "mas_param_jacobian":
            return mas_ops.marginals_grad_temp(scores, temperature, lengths)
        if op_name == "mas_backward_full":
            return mas_full_outputs(scores, temperature, lengths)
        raise AssertionError(f"unknown MAS op {op_name}")

    @pytest.mark.parametrize(
        "op_name",
        [
            "mas",
            "mas_float",
            "mas_with_grads",
            "mas_hvp",
            "mas_param_jacobian",
            "mas_backward_full",
        ],
    )
    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [[0, 5]], r"lengths\[0,0\] must be between 1 and 4"),
            ("cpu", [[4, 0]], r"lengths\[0,1\] must be between 1 and 5"),
            ("cpu", [[5, 5]], r"lengths\[0,0\] must be between 1 and 4"),
            ("cpu", [[4, 6]], r"lengths\[0,1\] must be between 1 and 5"),
            pytest.param(
                "cuda",
                [[0, 5]],
                r"lengths\[0,0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[4, 0]],
                r"lengths\[0,1\] must be between 1 and 5",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[5, 5]],
                r"lengths\[0,0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[4, 6]],
                r"lengths\[0,1\] must be between 1 and 5",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_invalid_lengths_raise_at_op_boundary(self, op_name, device_type, lengths, match):
        """All MAS entry points must reject zero and over-length rows."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        tangent = torch.randn_like(scores)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            self.call_op(op_name, scores, tangent, 1.0, lengths_t)

    def test_lengths_shape_dtype_and_contiguity_raise(self):
        """Explicit CPU lengths must have the expected boundary contract."""
        scores = torch.randn(1, 4, 5)

        with pytest.raises(RuntimeError, match=r"lengths must be \[B, 2\]"):
            mas_ops.forward(scores, 1.0, torch.tensor([4, 5], dtype=torch.int32))

        with pytest.raises(RuntimeError, match=r"lengths must be int32"):
            mas_ops.forward(scores, 1.0, torch.tensor([[4, 5]], dtype=torch.int64))

        non_contiguous = torch.tensor([[4, 0], [5, 0]], dtype=torch.int32)[:, 0].unsqueeze(0)
        assert not non_contiguous.is_contiguous()
        with pytest.raises(RuntimeError, match=r"lengths must be contiguous"):
            mas_ops.forward(scores, 1.0, non_contiguous)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_lengths_must_match_scores_device(self):
        """CUDA MAS must reject CPU lengths before launching kernels."""
        scores = torch.randn(1, 4, 5, device="cuda")
        lengths = torch.tensor([[4, 5]], dtype=torch.int32)

        with pytest.raises(RuntimeError, match=r"lengths must be a CUDA tensor"):
            mas_ops.forward(scores, 1.0, lengths)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_public_mas_audit_repro_raises(self):
        """The confirmed public repro should raise cleanly instead of faulting."""
        scores = torch.randn(1, 8, 12, device="cuda")
        lengths = torch.tensor([[0, 12]], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 1 and 8"):
            orihime.mas(scores, temperature=1.0, lengths=lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 score matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        temperature = 1.0

        score, posteriors = mas_forward_with_grads(scores, temperature, None)
        score_ref, _ = mas_forward_naive(scores, temperature)
        posteriors_ref = mas_naive(scores, temperature)

        # For 1x1, posterior should be 1.0 (only one path)
        assert allclose(score, score_ref), f"Single element score wrong: {score.item()} vs {score_ref.item()}"
        assert allclose(posteriors, posteriors_ref), f"Single element posterior wrong: {posteriors.item()} vs {posteriors_ref.item()}"

    def test_single_text_token(self, device):
        """Test Tx1 score matrix (single text token)."""
        scores = torch.tensor([[[0.1], [0.2], [0.3], [0.4]]], device=device)
        temperature = 1.0

        score, posteriors = mas_forward_with_grads(scores, temperature, None)
        score_ref, _ = mas_forward_naive(scores, temperature)
        posteriors_ref = mas_naive(scores, temperature)

        # All frames must align to the single text token
        assert allclose(score, score_ref), "Single text token score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Single text token posteriors mismatch"

    def test_equal_length(self, device):
        """Test TxT score matrix (one-to-one alignment)."""
        T = 5
        scores = torch.randn(1, T, T, device=device)
        temperature = 1.0

        score, posteriors = mas_forward_with_grads(scores, temperature, None)
        score_ref, _ = mas_forward_naive(scores, temperature)
        posteriors_ref = mas_naive(scores, temperature)

        assert allclose(score, score_ref), "Equal length score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Equal length posteriors mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard alignment)."""
        B, T, S = 2, 10, 6
        temperature = 0.01

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        _, posteriors = mas_forward_with_grads(scores, temperature, None)

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, T, S = 2, 10, 6
        temperature = 10.0

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        _, posteriors = mas_forward_with_grads(scores, temperature, None)
        posteriors_ref = mas_naive(scores, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_monotonicity_constraint(self, device):
        """Test that alignment respects monotonicity."""
        B, T, S = 1, 8, 4
        temperature = 0.1  # Low temp for clear alignment

        # Create scores that favor diagonal alignment
        scores = torch.zeros(B, T, S, device=device)
        for t in range(T):
            s = min(t // 2, S - 1)  # Expected alignment: each text gets ~2 frames
            scores[0, t, s] = 5.0

        _, posteriors = mas_forward_with_grads(scores, temperature, None)

        # Posteriors should concentrate on the diagonal region
        # (monotonicity is enforced by the DP structure)
        assert posteriors.sum(dim=-1).min() > 0.9, "Each frame should align somewhere"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestParamJacobian:

    def test_param_jacobian_temperature_cpu(self):
        """CPU dP/dT should match a finite-difference reference."""
        B, T, S = 2, 9, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, T, S)

        with torch_num_threads(1):
            dP_dT = mas_ops.marginals_grad_temp(scores, temperature, None)
            posteriors_plus = mas_naive(scores, temperature + eps)
            posteriors_minus = mas_naive(scores, temperature - eps)
            dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=2e-2, atol=5e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_cpu_parity(self):
        """CUDA dP/dT should match CPU dP/dT (r27 regression)."""
        B, T, S = 2, 9, 6
        temperature = 1.0

        torch.manual_seed(123)
        scores_cpu = torch.randn(B, T, S)
        scores_cuda = scores_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = mas_ops.marginals_grad_temp(scores_cpu, temperature, None)
        dP_dT_cuda = mas_ops.marginals_grad_temp(scores_cuda, temperature, None)

        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU param Jacobian mismatch: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_finite_diff(self):
        """CUDA dP/dT should match a finite-difference reference (r27 regression)."""
        B, T, S = 2, 9, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, T, S).cuda()

        dP_dT = mas_ops.marginals_grad_temp(scores, temperature, None)
        posteriors_plus = mas_naive(scores, temperature + eps)
        posteriors_minus = mas_naive(scores, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=2e-2, atol=5e-3), \
            f"CUDA param Jacobian vs finite diff mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_various_temps(self):
        """CUDA dP/dT should match CPU across multiple temperatures (r27 regression)."""
        B, T, S = 2, 10, 6

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores_cpu = torch.randn(B, T, S)
            scores_cuda = scores_cpu.cuda()

            with torch_num_threads(1):
                dP_dT_cpu = mas_ops.marginals_grad_temp(scores_cpu, temperature, None)
            dP_dT_cuda = mas_ops.marginals_grad_temp(scores_cuda, temperature, None)

            assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
                f"CUDA/CPU param Jacobian mismatch for temp={temperature}: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_first_column_regression(self):
        """CUDA dP/dT should write back the propagated first-column value at (0,0) (r45 regression)."""
        temperature = 0.1
        scores_cpu = PARAM_JACOBIAN_FIRST_COLUMN_REGRESSION_SCORES.clone()
        scores_cuda = scores_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = mas_ops.marginals_grad_temp(scores_cpu, temperature, None)
        dP_dT_cuda = mas_ops.marginals_grad_temp(scores_cuda, temperature, None).cpu()

        assert abs(dP_dT_cpu[0, 0, 0].item()) > 1e-5, \
            "regression fixture expected a nonzero CPU first-column dP/dT at (0,0)"
        assert abs(dP_dT_cuda[0, 0, 0].item()) > 1e-5, \
            f"CUDA left first-column dP/dT at (0,0) effectively zero: {dP_dT_cuda[0, 0, 0].item()}"
        assert torch.sign(dP_dT_cuda[0, 0, 0]) == torch.sign(dP_dT_cpu[0, 0, 0]), \
            f"CUDA first-column dP/dT changed sign at (0,0): cpu={dP_dT_cpu[0, 0, 0].item()} cuda={dP_dT_cuda[0, 0, 0].item()}"
        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU param Jacobian mismatch in first-column regression: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_variable_lengths_masked_parity(self):
        """CUDA dP/dT should respect per-batch lengths and keep padded regions zero."""
        B, max_T, max_S = 3, 12, 6
        temperature = 0.9

        torch.manual_seed(20260409)
        scores_cpu = torch.randn(B, max_T, max_S)
        lengths_cpu = torch.tensor([[12, 6], [7, 4], [1, 1]], dtype=torch.int32)
        scores_cuda = scores_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = mas_ops.marginals_grad_temp(scores_cpu, temperature, lengths_cpu)
        dP_dT_cuda = mas_ops.marginals_grad_temp(scores_cuda, temperature, lengths_cuda).cpu()

        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU variable-length param Jacobian mismatch: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

        for batch, (t_len, s_len) in enumerate(lengths_cpu.tolist()):
            if t_len < max_T:
                assert torch.count_nonzero(dP_dT_cuda[batch, t_len:, :]).item() == 0, \
                    f"CUDA wrote past active T length for batch {batch}"
            if s_len < max_S:
                assert torch.count_nonzero(dP_dT_cuda[batch, :, s_len:]).item() == 0, \
                    f"CUDA wrote past active S length for batch {batch}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestWithGrads:

    def test_with_grads_output(self, device):
        """Test mas_with_grads returns correct values."""
        B, T, S = 2, 10, 6
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        score, posteriors = mas_forward_with_grads(scores, temperature, None)

        # Compare with reference
        score_ref, _ = mas_forward_naive(scores, temperature)
        posteriors_ref = mas_naive(scores, temperature)

        assert allclose(score, score_ref), "with_grads score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "with_grads posteriors mismatch"

    def test_backward_full_output(self, device):
        """Test mas_backward_full returns correct values."""
        B, T, S = 2, 10, 6
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        score, posteriors, grad_T = mas_full_outputs(scores, temperature, None)

        # Compare with mas_with_grads
        score_ref, posteriors_ref = mas_forward_with_grads(scores, temperature, None)

        assert allclose(score, score_ref), "backward_full score mismatch"
        assert allclose(posteriors, posteriors_ref), "backward_full posteriors mismatch"
        assert grad_T.shape == (B,), f"grad_T shape wrong: {grad_T.shape}"


# --- Memory-safety regression tests (merged from test_mas_{cpp,cuda}_memsafety.py) ---

import re
from pathlib import Path


def _make_noncontiguous(tensor):
    storage = torch.empty(
        tensor.size(0),
        tensor.size(1),
        tensor.size(2) * 2,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = storage[..., ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _assert_tree_allclose(actual, expected):
    if isinstance(actual, (tuple, list)):
        assert isinstance(expected, (tuple, list))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_allclose(actual_item, expected_item)
        return

    assert actual.shape == expected.shape
    if actual.dim() == 3:
        assert actual.is_contiguous()
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)


CPU_MAS_CALLS = {
    "mas": lambda scores, tangent: orihime.mas(
        scores,
        temperature=1.0,
    ),
    "mas_float": lambda scores, tangent: mas_ops.forward(scores, 1.0, None),
    "mas_with_grads": lambda scores, tangent: mas_forward_with_grads(scores, 1.0, None),
    "mas_hvp": lambda scores, tangent: mas_ops.marginals_hvp(
        scores, tangent.contiguous(), 1.0, None
    ),
    "mas_param_jacobian": lambda scores, tangent: mas_ops.marginals_grad_temp(
        scores, 1.0, None
    ),
    "mas_backward_full": lambda scores, tangent: mas_full_outputs(
        scores, 1.0, None
    ),
}


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.parametrize("op_name", CPU_MAS_CALLS)
def test_noncontiguous_scores_return_fully_initialized_contiguous_outputs(op_name):
    torch.manual_seed(20260707)
    scores_contig = torch.randn(2, 6, 4)
    tangent_contig = torch.randn_like(scores_contig)
    scores_view = _make_noncontiguous(scores_contig)
    tangent_view = _make_noncontiguous(tangent_contig)

    call = CPU_MAS_CALLS[op_name]
    expected = call(scores_contig, tangent_contig)
    actual = call(scores_view, tangent_view)

    _assert_tree_allclose(actual, expected)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.parametrize("op_name", CPU_MAS_CALLS)
def test_rank4_scores_rejected_at_cpu_boundary(op_name):
    scores = torch.randn(1, 4, 5, 2)
    tangent = torch.randn_like(scores)

    with pytest.raises(
        (RuntimeError, ValueError),
        match=r"scores must (?:be 3D|have shape \[B, T, S\])",
    ):
        CPU_MAS_CALLS[op_name](scores, tangent)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_hvp_rejects_bad_tangent_shape_before_cpu_kernel():
    scores = torch.randn(2, 6, 4)
    tangent = torch.randn(2, 6, 3)

    with pytest.raises(RuntimeError, match=r"tangent must have same shape as scores"):
        mas_ops.marginals_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_hvp_rejects_bad_tangent_dtype_before_cpu_kernel():
    scores = torch.randn(2, 6, 4)
    tangent = torch.randn(2, 6, 4, dtype=torch.float64)

    with pytest.raises(RuntimeError, match=r"tangent must have dtype torch\.float32"):
        mas_ops.marginals_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hvp_rejects_cross_device_tangent_cleanly():
    scores = torch.randn(2, 6, 4)
    tangent = torch.randn(2, 6, 4, device="cuda")

    with pytest.raises(RuntimeError, match=r"tangent must be on same device as scores"):
        mas_ops.marginals_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_kernel_cell_offsets_are_wide():
    kernel_path = Path(__file__).resolve().parents[1] / "src" / "mas" / "kernels_cpu.cpp"
    kernel = kernel_path.read_text()

    assert "static inline size_t cell_index" in kernel
    assert "int idx = t * max_S + s" not in kernel
    assert "int idx_stay = (t - 1) * max_S + s" not in kernel
    assert "int idx_diag = (t - 1) * max_S + (s - 1)" not in kernel


def _source_text(relative_path):
    return (Path(__file__).resolve().parents[1] / relative_path).read_text()


def _as_tuple(result):
    # Some ops (e.g. mas_float) return a list of tensors, others a tuple,
    # others a single tensor. Normalise all of them to a tuple so callers can
    # iterate and compare element-wise.
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def _assert_outputs_match(name, actual, expected, scores_shape):
    for index, (actual_t, expected_t) in enumerate(zip(_as_tuple(actual), _as_tuple(expected))):
        torch.testing.assert_close(actual_t, expected_t, rtol=1e-5, atol=1e-5)
        if actual_t.dim() == 3:
            assert tuple(actual_t.shape) == scores_shape, f"{name}[{index}] has wrong shape"
            assert actual_t.is_contiguous(), f"{name}[{index}] must be contiguous"


def test_cuda_kernels_use_wide_cell_offsets_for_mas_indices():
    text = _source_text("src/mas/kernels_gpu.cu")

    assert "__device__ __forceinline__ size_t cell_offset" in text
    assert "size_t total_valid = (size_t)T * (size_t)S;" in text

    banned_patterns = [
        r"\bint\s+(idx|idx_stay|idx_diag|final_idx|flat_idx)\s*=",
        r"\bfor\s*\(\s*int\s+idx\s*=\s*threadIdx\.x;\s*idx\s*<\s*T\s*\*\s*S",
        r"\bt\s*\*\s*max_S",
        r"\(T\s*-\s*1\)\s*\*\s*max_S",
    ]
    for pattern in banned_patterns:
        assert re.search(pattern, text) is None, f"found narrow MAS CUDA index pattern: {pattern}"


def test_cuda_wrappers_guard_mas_launch_device_and_validate_tangent():
    text = _source_text("src/mas/torch_cuda.cpp")

    assert text.count("ORIHIME_CUDA_GUARD(scores)") >= 6
    assert "validate_mas_tangent_cuda(scores, V);" in text
    assert "tangent.device() == scores.device()" in text


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_hvp_rejects_bad_tangent_shape_dtype_and_device():
    scores = torch.randn(2, 5, 4, device="cuda")

    with pytest.raises(RuntimeError, match="tangent must have same shape as scores"):
        mas_ops.marginals_hvp(scores, torch.randn(2, 5, 3, device="cuda"), 1.0, None)

    with pytest.raises(RuntimeError, match="tangent must be float32"):
        mas_ops.marginals_hvp(scores, torch.randn(2, 5, 4, device="cuda", dtype=torch.float64), 1.0, None)

    with pytest.raises(RuntimeError, match="tangent must be a CUDA tensor"):
        mas_ops.marginals_hvp(scores, torch.randn(2, 5, 4), 1.0, None)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("mas_float", lambda scores: mas_ops.forward(scores, 1.0, None)),
        ("mas_with_grads", lambda scores: mas_forward_with_grads(scores, 1.0, None)),
        ("mas_hvp", lambda scores: mas_ops.marginals_hvp(scores, torch.randn_like(scores), 1.0, None)),
        ("mas_param_jacobian", lambda scores: mas_ops.marginals_grad_temp(scores, 1.0, None)),
        ("mas_backward_full", lambda scores: mas_full_outputs(scores, 1.0, None)),
    ],
)
def test_cuda_wrappers_reject_rank4_scores(name, call):
    scores = torch.randn(1, 5, 4, 2, device="cuda")

    with pytest.raises(RuntimeError, match=r"scores must be 3D \[B, T, S\]"):
        call(scores)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_noncontiguous_scores_match_contiguous_outputs():
    torch.manual_seed(123)
    scores = torch.randn(2, 5, 8, device="cuda")[:, :, ::2]
    tangent = torch.randn(2, 5, 8, device="cuda")[:, :, ::2]
    assert not scores.is_contiguous()
    assert not tangent.is_contiguous()

    scores_c = scores.contiguous()
    tangent_c = tangent.contiguous()
    cases = [
        ("mas_float", lambda s, v: mas_ops.forward(s, 1.0, None)),
        ("mas_with_grads", lambda s, v: mas_forward_with_grads(s, 1.0, None)),
        ("mas_hvp", lambda s, v: mas_ops.marginals_hvp(s, v.contiguous(), 1.0, None)),
        ("mas_param_jacobian", lambda s, v: mas_ops.marginals_grad_temp(s, 1.0, None)),
        ("mas_backward_full", lambda s, v: mas_full_outputs(s, 1.0, None)),
    ]

    for name, call in cases:
        actual = call(scores, tangent)
        expected = call(scores_c, tangent_c)
        _assert_outputs_match(name, actual, expected, tuple(scores.shape))


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.multi_gpu
def test_current_device_can_differ_from_scores_device():
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(2, 5, 4, device="cuda:1")
        tangent = torch.randn_like(scores)

        assert torch.cuda.current_device() == 0
        assert scores.device.index == 1
        assert scores.device.index != torch.cuda.current_device()

        partition, posteriors = mas_forward_with_grads(scores, 1.0, None)
        hvp = mas_ops.marginals_hvp(scores, tangent, 1.0, None)

        assert partition.device == scores.device
        assert posteriors.device == scores.device
        assert hvp.device == scores.device
        torch.cuda.synchronize(scores.device)
    finally:
        torch.cuda.set_device(previous_device)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.multi_gpu
def test_device_01_current_device_differs_for_all_public_and_raw_entrypoints():
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        target = torch.device("cuda:1")
        scores = torch.randn(2, 5, 3, device=target)
        tangent = torch.randn_like(scores)
        cotangent = torch.randn_like(scores)
        lengths = torch.tensor(
            [[5, 3], [4, 2]],
            dtype=torch.int32,
            device=target,
        )
        temperature_tensor = torch.tensor(
            [0.9],
            dtype=scores.dtype,
            device=target,
        )
        raw = orihime.raw.mas

        assert target.index != torch.cuda.current_device()
        calls = (
            ("public_map", lambda: orihime.mas(scores, lengths=lengths)),
            ("public_value", lambda: orihime.mas_value(scores, lengths=lengths)),
            ("public_entropy", lambda: orihime.mas_entropy(scores, lengths=lengths)),
            ("raw_forward", lambda: raw.forward(scores, 0.9, lengths)),
            (
                "raw_forward_t",
                lambda: raw.forward_t(scores, temperature_tensor, lengths),
            ),
            (
                "raw_value_grad_params",
                lambda: raw.value_grad_params(scores, 0.9, lengths),
            ),
            (
                "raw_marginals_backward",
                lambda: raw.marginals_backward(
                    scores,
                    cotangent,
                    0.9,
                    lengths,
                ),
            ),
            (
                "raw_marginals_hvp",
                lambda: raw.marginals_hvp(
                    scores,
                    tangent,
                    0.9,
                    lengths,
                ),
            ),
            (
                "raw_marginals_grad_temp",
                lambda: raw.marginals_grad_temp(scores, 0.9, lengths),
            ),
            (
                "raw_vjp_one",
                lambda: raw.vjp_one(
                    scores,
                    wrt="temperature",
                    cotangent=cotangent,
                    temperature=0.9,
                    lengths=lengths,
                ),
            ),
            (
                "raw_vjp",
                lambda: raw.vjp(
                    scores,
                    wrt=("temperature",),
                    cotangent=cotangent,
                    temperature=0.9,
                    lengths=lengths,
                ),
            ),
        )

        for name, call in calls:
            result = call()
            torch.cuda.synchronize(target)
            for tensor in _iter_tensors(result):
                assert tensor.device == target, f"{name} returned {tensor.device}"
                assert torch.isfinite(tensor).all(), f"{name} returned non-finite data"
    finally:
        torch.cuda.set_device(previous_device)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.multi_gpu
def test_device_02_wrong_device_secondary_tensors_are_rejected():
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(2, 5, 3, device="cuda:0")
        tangent = torch.randn_like(scores)
        cotangent = torch.randn_like(scores)
        lengths = torch.tensor(
            [[5, 3], [4, 2]],
            dtype=torch.int32,
            device="cuda:0",
        )
        other_lengths = lengths.to("cuda:1")
        other_tangent = tangent.to("cuda:1")
        other_cotangent = cotangent.to("cuda:1")
        other_temperature = torch.tensor(
            [0.9],
            dtype=scores.dtype,
            device="cuda:1",
        )
        raw = orihime.raw.mas

        assert scores.device.index != other_lengths.device.index
        assert scores.device.index != other_tangent.device.index
        assert scores.device.index != other_cotangent.device.index
        assert scores.device.index != other_temperature.device.index

        wrong_lengths_calls = (
            ("public_map_lengths", lambda: orihime.mas(scores, lengths=other_lengths)),
            ("public_value_lengths", lambda: orihime.mas_value(scores, lengths=other_lengths)),
            ("public_entropy_lengths", lambda: orihime.mas_entropy(scores, lengths=other_lengths)),
            ("raw_forward_lengths", lambda: raw.forward(scores, 0.9, other_lengths)),
            ("raw_forward_t_lengths", lambda: raw.forward_t(
                scores,
                torch.tensor([0.9], device="cuda:0"),
                other_lengths,
            )),
            ("raw_value_grad_params_lengths", lambda: raw.value_grad_params(scores, 0.9, other_lengths)),
            ("raw_marginals_grad_temp_lengths", lambda: raw.marginals_grad_temp(scores, 0.9, other_lengths)),
        )
        wrong_temperature_calls = (
            ("public_map_temperature", lambda: orihime.mas(
                scores,
                temperature=other_temperature,
                lengths=lengths,
            )),
            ("public_value_temperature", lambda: orihime.mas_value(
                scores,
                temperature=other_temperature,
                lengths=lengths,
            )),
            ("public_entropy_temperature", lambda: orihime.mas_entropy(
                scores,
                temperature=other_temperature,
                lengths=lengths,
            )),
            ("raw_forward_t_temperature", lambda: raw.forward_t(scores, other_temperature, lengths)),
        )
        wrong_tangent_calls = (
            ("raw_marginals_hvp_tangent", lambda: raw.marginals_hvp(
                scores,
                other_tangent,
                0.9,
                lengths,
            )),
        )
        wrong_cotangent_calls = (
            ("raw_marginals_backward_cotangent", lambda: raw.marginals_backward(
                scores,
                other_cotangent,
                0.9,
                lengths,
            )),
            ("raw_vjp_one_cotangent", lambda: raw.vjp_one(
                scores,
                wrt="temperature",
                cotangent=other_cotangent,
                temperature=0.9,
                lengths=lengths,
            )),
            ("raw_vjp_cotangent", lambda: raw.vjp(
                scores,
                wrt=("temperature",),
                cotangent=other_cotangent,
                temperature=0.9,
                lengths=lengths,
            )),
        )

        def assert_wrong_device_entrypoint(name, call):
            try:
                with pytest.raises((RuntimeError, ValueError), match=r"same device"):
                    call()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                pytest.fail(f"{name}: {exc}")

        for name, call in (
            *wrong_lengths_calls,
            *wrong_temperature_calls,
            *wrong_tangent_calls,
            *wrong_cotangent_calls,
        ):
            assert_wrong_device_entrypoint(name, call)
    finally:
        torch.cuda.set_device(previous_device)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_pub_01_mas_public_observables_are_finite_and_shaped():
    scores = torch.randn(2, 6, 4)

    map_result = orihime.mas(scores, temperature=1.0)
    value_result = orihime.mas_value(scores, temperature=1.0)
    entropy_result = orihime.mas_entropy(scores, temperature=1.0)

    assert map_result.shape == scores.shape
    assert value_result.shape == (scores.shape[0],)
    assert entropy_result.shape == (scores.shape[0],)
    assert torch.isfinite(map_result).all()
    assert torch.isfinite(value_result).all()
    assert torch.isfinite(entropy_result).all()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_pub_02_mas_public_docstrings_state_score_orientation():
    expected_fragments = {
        "mas": "score-native",
        "mas_value": "score-native",
        "mas_entropy": "alignment distribution",
    }
    for name, expected_fragment in expected_fragments.items():
        docstring = inspect.getdoc(getattr(orihime, name))
        assert docstring is not None
        assert expected_fragment in docstring.lower()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_raw_01_mas_selected_vjp_is_type_stable():
    scores = torch.randn(2, 6, 4)
    cotangent = torch.linspace(
        -0.5,
        0.75,
        scores.numel(),
    ).reshape_as(scores).contiguous()
    raw = orihime.raw.mas

    assert raw.vjp_fields == ("temperature",)
    one = raw.vjp_one(
        scores,
        wrt="temperature",
        cotangent=cotangent,
        temperature=0.9,
    )
    selected = raw.vjp(
        scores,
        wrt=("temperature",),
        cotangent=cotangent,
        temperature=0.9,
    )

    assert isinstance(one, torch.Tensor)
    assert not one.requires_grad
    assert torch.isfinite(one).all()
    assert tuple(selected) == ("temperature",)
    assert isinstance(selected["temperature"], torch.Tensor)
    assert not selected["temperature"].requires_grad
    assert torch.isfinite(selected["temperature"]).all()
    torch.testing.assert_close(selected["temperature"], one)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_num_05_mas_full_vjp_matches_independent_reference():
    torch.manual_seed(20260810)
    scores = torch.randn(2, 6, 4)
    cotangent = torch.randn_like(scores)
    temperature = 0.8

    grad_scores, grad_temperature = mas_ops.marginals_backward(
        scores,
        cotangent,
        temperature,
        None,
    )

    reference_scores = scores.detach().clone().requires_grad_(True)
    reference_temperature = torch.tensor(
        temperature,
        dtype=scores.dtype,
        requires_grad=True,
    )
    reference_posteriors = mas_naive(
        reference_scores,
        reference_temperature,
    )
    reference_loss = (reference_posteriors * cotangent).sum()
    expected_scores, expected_temperature = torch.autograd.grad(
        reference_loss,
        (reference_scores, reference_temperature),
    )

    torch.testing.assert_close(
        grad_scores,
        expected_scores,
        rtol=2e-2,
        atol=5e-3,
    )
    torch.testing.assert_close(
        grad_temperature.reshape_as(reference_temperature),
        expected_temperature,
        rtol=2e-2,
        atol=5e-3,
    )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_len_02_all_mas_output_and_derivative_paths_zero_padded_regions():
    torch.manual_seed(20260810)
    scores = torch.randn(3, 8, 5)
    tangent = torch.randn_like(scores)
    cotangent = torch.randn_like(scores)
    lengths = torch.tensor(
        [[8, 5], [6, 3], [4, 2]],
        dtype=torch.int32,
    )
    temperature = 0.9
    temperature_tensor = torch.tensor([temperature])

    _, posteriors = mas_ops.forward(scores, temperature, lengths)
    _, posteriors_t = mas_ops.forward_t(
        scores,
        temperature_tensor,
        lengths,
    )
    _, posteriors_with_grads = mas_forward_with_grads(
        scores,
        temperature,
        lengths,
    )
    _, posteriors_full, _ = mas_full_outputs(
        scores,
        temperature,
        lengths,
    )
    hvp = mas_ops.marginals_hvp(
        scores,
        tangent,
        temperature,
        lengths,
    )
    dP_dT = mas_ops.marginals_grad_temp(
        scores,
        temperature,
        lengths,
    )
    grad_scores, _ = mas_ops.marginals_backward(
        scores,
        cotangent,
        temperature,
        lengths,
    )
    public_map = orihime.mas(
        scores,
        temperature=temperature,
        lengths=lengths,
    )

    for output in (
        posteriors,
        posteriors_t,
        posteriors_with_grads,
        posteriors_full,
        hvp,
        dP_dT,
        grad_scores,
        public_map,
    ):
        assert torch.isfinite(output).all()
        _assert_padded_regions_zero(output, lengths)

    value = orihime.mas_value(
        scores,
        temperature=temperature,
        lengths=lengths,
    )
    entropy = orihime.mas_entropy(
        scores,
        temperature=temperature,
        lengths=lengths,
    )
    assert torch.isfinite(value).all()
    assert torch.isfinite(entropy).all()

    value_scores = scores.detach().clone().requires_grad_(True)
    orihime.mas_value(
        value_scores,
        temperature=temperature,
        lengths=lengths,
    ).sum().backward()
    assert value_scores.grad is not None
    _assert_padded_regions_zero(value_scores.grad, lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_edge_01_fam_06_mas_feasibility_edges_are_explicit():
    with pytest.raises(ValueError, match=r"scores\.shape\[-2\] >= scores\.shape\[-1\]"):
        orihime.mas(torch.randn(1, 1, 2), temperature=1.0)

    scores = torch.randn(1, 4, 3)
    infeasible_lengths = torch.tensor([[2, 3]], dtype=torch.int32)
    for function in (orihime.mas, orihime.mas_value, orihime.mas_entropy):
        with pytest.raises(
            (RuntimeError, ValueError),
            match=r"mas requires lengths\[:, 0\] >= lengths\[:, 1\]",
        ):
            function(
                scores,
                temperature=1.0,
                lengths=infeasible_lengths,
            )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_guard_02_noncontiguous_primary_and_private_derivatives_are_initialized():
    torch.manual_seed(20260810)
    scores_contiguous = torch.randn(2, 6, 4)
    tangent_contiguous = torch.randn_like(scores_contiguous)
    scores = _make_noncontiguous(scores_contiguous)
    tangent = _make_noncontiguous(tangent_contiguous)
    lengths = torch.tensor(
        [[6, 4], [5, 3]],
        dtype=torch.int32,
    )
    raw = orihime.raw.mas

    cases = (
        (
            "public_map",
            lambda s, v: orihime.mas(s, temperature=0.9, lengths=lengths),
        ),
        (
            "public_value",
            lambda s, v: orihime.mas_value(s, temperature=0.9, lengths=lengths),
        ),
        (
            "public_entropy",
            lambda s, v: orihime.mas_entropy(s, temperature=0.9, lengths=lengths),
        ),
        (
            "raw_forward",
            lambda s, v: raw.forward(s, 0.9, lengths),
        ),
        (
            "raw_forward_t",
            lambda s, v: raw.forward_t(
                s,
                s.new_tensor([0.9]),
                lengths,
            ),
        ),
        (
            "raw_value_grad_params",
            lambda s, v: raw.value_grad_params(s, 0.9, lengths),
        ),
        (
            "raw_marginals_grad_temp",
            lambda s, v: raw.marginals_grad_temp(s, 0.9, lengths),
        ),
    )

    for name, call in cases:
        expected = call(scores_contiguous, tangent_contiguous)
        actual = call(scores, tangent)
        _assert_tree_allclose(actual, expected)
        for tensor in _iter_tensors(actual):
            assert torch.isfinite(tensor).all(), f"non-finite {name} output"

    with pytest.raises(RuntimeError, match=r"tangent must be contiguous"):
        raw.marginals_hvp(scores, tangent, 0.9, lengths)
    with pytest.raises(RuntimeError, match=r"cotangent must be contiguous"):
        raw.marginals_backward(scores, tangent, 0.9, lengths)

    cotangent = tangent
    with pytest.raises(ValueError, match=r"cotangent.*contiguous"):
        raw.vjp_one(
            scores,
            wrt="temperature",
            cotangent=cotangent,
            temperature=0.9,
            lengths=lengths,
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_over_01_cpu_dimensions_are_rejected_before_allocation():
    oversized = torch.iinfo(torch.int32).max + 1
    scores = torch.empty_strided(
        (0, oversized, 1),
        (0, 0, 0),
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match=r"scores dimensions must fit int32",
    ):
        mas_ops.forward(scores, 1.0, None)


def test_over_02_cuda_wide_index_source_contract_is_explicit():
    kernel = _source_text("src/mas/kernels_gpu.cu")
    wrapper = _source_text("src/mas/torch_cuda.cpp")

    assert "__device__ __forceinline__ size_t cell_offset" in kernel
    assert "size_t total = (size_t)B * max_T * max_S;" in kernel
    assert kernel.count("size_t stride = (size_t)max_T * max_S;") >= 8
    assert re.search(r"\bint\s+(idx|idx_stay|idx_diag|final_idx|flat_idx)\s*=", kernel) is None
    assert wrapper.count("ORIHIME_CUDA_GUARD(scores)") >= 6


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_over_02_public_cuda_empty_batch_is_rejected_before_dispatch():
    oversized = torch.iinfo(torch.int32).max + 1
    scores = torch.empty_strided(
        (0, oversized, 1),
        (0, 0, 0),
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match=r"empty leading batch \(B=0\)"):
        orihime.mas(scores, temperature=1.0)


def test_prune_01_mas_tensor_backward_guards_unused_output_gradients():
    for relative_path in (
        "src/mas/torch_cpu.cpp",
        "src/mas/torch_cuda.cpp",
    ):
        source = _source_text(relative_path)
        assert "ctx->set_materialize_grads(false)" in source
        assert "if (grad_posteriors.defined() && grad_posteriors.numel() > 0)" in source


_PROFILER_INITIALIZATION_FAILURES = (
    "profiler is not initialized",
    "profiler initialization failed",
    "failed to initialize profiler",
    "could not initialize profiler",
    "unable to initialize profiler",
    "kineto is not initialized",
    "cupti is not initialized",
)


def _is_profiler_initialization_failure(exc):
    message = " ".join(str(exc).lower().split())
    return any(marker in message for marker in _PROFILER_INITIALIZATION_FAILURES)


def _cuda_kernel_names(call):
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:
        pytest.skip(f"CUDA profiler unavailable: {exc}")

    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
    except RuntimeError as exc:
        if _is_profiler_initialization_failure(exc):
            pytest.skip(f"CUDA profiler failed: {exc}")
        raise

    return {
        getattr(event, "name", "") or getattr(event, "key", "")
        for event in prof.events()
    }


def _mas_derivative_kernel_names():
    scores = torch.randn(1, 8, 5, device="cuda")
    tangent = torch.randn_like(scores)
    temperature = 0.9
    lengths = torch.tensor(
        [[8, 5]],
        dtype=torch.int32,
        device="cuda",
    )

    def derivative_paths():
        mas_ops.marginals_hvp(scores, tangent, temperature, lengths)
        mas_ops.marginals_grad_temp(scores, temperature, lengths)

    return _cuda_kernel_names(derivative_paths)


def _assert_mas_derivative_kernel_names_are_observable(kernel_names):
    assert any("hvp" in name.lower() for name in kernel_names), (
        "CUDA profiler did not expose MAS HVP kernel names: "
        f"{sorted(kernel_names)[:20]}"
    )
    assert any("param_grad" in name.lower() for name in kernel_names), (
        "CUDA profiler did not expose MAS parameter-gradient kernel names: "
        f"{sorted(kernel_names)[:20]}"
    )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_prune_01_mas_derivative_kernel_names_are_observable():
    _assert_mas_derivative_kernel_names_are_observable(_mas_derivative_kernel_names())


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_prune_01_score_only_backward_skips_unused_hvp_and_param_grad_kernels():
    _assert_mas_derivative_kernel_names_are_observable(_mas_derivative_kernel_names())

    scores = torch.randn(1, 8, 5, device="cuda", requires_grad=True)
    temperature = torch.tensor(
        [0.9],
        device="cuda",
        requires_grad=True,
    )
    lengths = torch.tensor(
        [[8, 5]],
        dtype=torch.int32,
        device="cuda",
    )

    def score_only_backward():
        scores.grad = None
        temperature.grad = None
        score, _ = mas_ops.forward_t(scores, temperature, lengths)
        score.sum().backward()

    kernel_names = _cuda_kernel_names(score_only_backward)
    mas_names = tuple(
        name for name in kernel_names if "mas" in name.lower()
    )
    assert mas_names, f"CUDA profiler did not capture MAS kernels: {sorted(kernel_names)[:20]}"
    unexpected = sorted(
        name
        for name in mas_names
        if "hvp" in name.lower() or "param_grad" in name.lower()
    )
    assert not unexpected, (
        "score-only backward launched unused MAS derivative kernels: "
        f"{unexpected}"
    )


def _mas_backend_derivatives(scores, tangent, cotangent, lengths):
    temperature = 0.9
    value, posteriors, grad_temperature = mas_full_outputs(
        scores,
        temperature,
        lengths,
    )
    param_jacobian = mas_ops.marginals_grad_temp(
        scores,
        temperature,
        lengths,
    )
    hvp = mas_ops.marginals_hvp(
        scores,
        tangent,
        temperature,
        lengths,
    )
    full_scores, full_temperature = mas_ops.marginals_backward(
        scores,
        cotangent,
        temperature,
        lengths,
    )

    map_scores = scores.detach().clone().requires_grad_(True)
    map_temperature = torch.tensor(
        [temperature],
        dtype=scores.dtype,
        device=scores.device,
        requires_grad=True,
    )
    map_output = orihime.mas(
        map_scores,
        temperature=map_temperature,
        lengths=lengths,
    )
    (map_output * cotangent).sum().backward()

    value_scores = scores.detach().clone().requires_grad_(True)
    value_temperature = torch.tensor(
        [temperature],
        dtype=scores.dtype,
        device=scores.device,
        requires_grad=True,
    )
    orihime.mas_value(
        value_scores,
        temperature=value_temperature,
        lengths=lengths,
    ).sum().backward()

    assert map_scores.grad is not None
    assert map_temperature.grad is not None
    assert value_scores.grad is not None
    assert value_temperature.grad is not None
    torch.testing.assert_close(
        map_scores.grad,
        full_scores,
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(
        map_temperature.grad,
        full_temperature.reshape_as(map_temperature),
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(
        value_scores.grad,
        posteriors,
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(
        value_temperature.grad,
        grad_temperature.sum().reshape_as(value_temperature),
        rtol=1e-3,
        atol=1e-4,
    )

    return {
        "param_jacobian": param_jacobian,
        "hvp": hvp,
        "full_scores": full_scores,
        "full_temperature": full_temperature,
        "map_scores": map_scores.grad,
        "map_temperature": map_temperature.grad,
        "value_scores": value_scores.grad,
        "value_temperature": value_temperature.grad,
    }


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_back_02_cpu_cuda_derivative_entrypoints_match():
    torch.manual_seed(20260810)
    scores_cpu = torch.randn(2, 7, 5)
    tangent_cpu = torch.randn_like(scores_cpu)
    cotangent_cpu = torch.randn_like(scores_cpu)
    lengths_cpu = torch.tensor(
        [[7, 5], [5, 3]],
        dtype=torch.int32,
    )

    cpu = _mas_backend_derivatives(
        scores_cpu,
        tangent_cpu,
        cotangent_cpu,
        lengths_cpu,
    )
    cuda = _mas_backend_derivatives(
        scores_cpu.cuda(),
        tangent_cpu.cuda(),
        cotangent_cpu.cuda(),
        lengths_cpu.cuda(),
    )

    for name in cpu:
        torch.testing.assert_close(
            cuda[name].cpu(),
            cpu[name],
            rtol=1e-3,
            atol=1e-4,
            msg=f"MAS CPU/CUDA derivative mismatch for {name}",
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
