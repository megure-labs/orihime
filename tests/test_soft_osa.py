# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft OSA (Optimal String Alignment / Restricted Damerau-Levenshtein).
"""

import contextlib

import pytest
import torch

from reference import osa_forward_naive, osa_naive
from operator_test_prerequisites import TWO_CUDA_DEVICES_REQUIRED

try:
    import orihime
    from orihime.ops import osa as osa_ops
    from operator_test_utils import osa_forward_with_grads, osa_param_field
    ORIHIME_AVAILABLE = True
except ImportError:
    ORIHIME_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def assert_padded_region_zero(name, tensor, lengths):
    tensor_cpu = tensor.cpu()
    lengths_cpu = lengths.cpu()
    for batch, (l1, l2) in enumerate(lengths_cpu.tolist()):
        if l1 < tensor_cpu.size(1):
            trailing_rows = tensor_cpu[batch, l1:, :]
            assert torch.count_nonzero(trailing_rows).item() == 0, \
                f"{name} batch {batch} wrote past L1={l1}"
        if l2 < tensor_cpu.size(2):
            trailing_cols = tensor_cpu[batch, :, l2:]
            assert torch.count_nonzero(trailing_cols).item() == 0, \
                f"{name} batch {batch} wrote past L2={l2}"


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


def assert_cuda_kernel_seen(kernel_names, token):
    if any(token in name for name in kernel_names):
        return
    relevant = sorted(name for name in kernel_names if "osa_" in name)
    raise AssertionError(
        f"CUDA profiler did not capture {token}; OSA kernels seen: {relevant[:20]}"
    )


def _assert_tensors_on_device(result, device):
    if isinstance(result, torch.Tensor):
        tensors = (result,)
    else:
        tensors = tuple(tensor for tensor in result if isinstance(tensor, torch.Tensor))

    for tensor in tensors:
        assert tensor.device == device


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(sub_costs, trans_mask, tangent, ins_cost, del_cost, trans_cost, temperature):
    distance, posteriors, grad_T, grad_ins, grad_del, grad_trans = osa_forward_with_grads(
        sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None
    )
    hvp = osa_ops.marginals_hvp(sub_costs, trans_mask, tangent, ins_cost, del_cost, trans_cost, temperature, None)
    dP_dIns = osa_param_field(sub_costs, trans_mask, 0, ins_cost, del_cost, trans_cost, temperature, None)
    dP_dDel = osa_param_field(sub_costs, trans_mask, 1, ins_cost, del_cost, trans_cost, temperature, None)
    dP_dTrans = osa_param_field(sub_costs, trans_mask, 2, ins_cost, del_cost, trans_cost, temperature, None)
    dP_dT = osa_param_field(sub_costs, trans_mask, 3, ins_cost, del_cost, trans_cost, temperature, None)
    return {
        "distance": distance,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "grad_ins": grad_ins,
        "grad_del": grad_del,
        "grad_trans": grad_trans,
        "hvp": hvp,
        "dP_dIns": dP_dIns,
        "dP_dDel": dP_dDel,
        "dP_dTrans": dP_dTrans,
        "dP_dT": dP_dT,
    }


def assert_threaded_osa_correctness(outputs, reference_outputs, thread_count):
    assert allclose(reference_outputs["distance"], outputs["distance"]), \
        f"{thread_count}-thread distance mismatch: max diff = {max_diff(reference_outputs['distance'], outputs['distance'])}"
    assert allclose(reference_outputs["posteriors"], outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(reference_outputs['posteriors'], outputs['posteriors'])}"
    assert allclose(reference_outputs["grad_T"], outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(reference_outputs['grad_T'], outputs['grad_T'])}"
    assert allclose(reference_outputs["grad_ins"], outputs["grad_ins"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_ins mismatch: max diff = {max_diff(reference_outputs['grad_ins'], outputs['grad_ins'])}"
    assert allclose(reference_outputs["grad_del"], outputs["grad_del"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_del mismatch: max diff = {max_diff(reference_outputs['grad_del'], outputs['grad_del'])}"
    assert allclose(reference_outputs["grad_trans"], outputs["grad_trans"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_trans mismatch: max diff = {max_diff(reference_outputs['grad_trans'], outputs['grad_trans'])}"
    assert allclose(reference_outputs["hvp"], outputs["hvp"], rtol=2e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(reference_outputs['hvp'], outputs['hvp'])}"
    assert allclose(reference_outputs["dP_dIns"], outputs["dP_dIns"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dins mismatch: max diff = {max_diff(reference_outputs['dP_dIns'], outputs['dP_dIns'])}"
    assert allclose(reference_outputs["dP_dDel"], outputs["dP_dDel"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/ddel mismatch: max diff = {max_diff(reference_outputs['dP_dDel'], outputs['dP_dDel'])}"
    assert allclose(reference_outputs["dP_dTrans"], outputs["dP_dTrans"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dtrans mismatch: max diff = {max_diff(reference_outputs['dP_dTrans'], outputs['dP_dTrans'])}"
    assert allclose(reference_outputs["dP_dT"], outputs["dP_dT"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(reference_outputs['dP_dT'], outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


def create_trans_mask(B, L1, L2, density=0.3, device='cpu'):
    """Create a random transposition mask.

    In practice, trans_mask[i,j] = 1 means s1[i-1]==s2[j] and s1[i]==s2[j-1].
    For testing, we just create random masks.
    """
    mask = (torch.rand(B, L1, L2, device=device) < density).float()
    # Transposition only valid for i >= 2 and j >= 2 (in 1-indexed coords)
    # In 0-indexed coords (mask), this means i >= 1 and j >= 1
    mask[:, 0, :] = 0  # Row 0 cannot transpose (i=1 in DP)
    mask[:, :, 0] = 0  # Col 0 cannot transpose (j=1 in DP)
    return mask


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(8, 10), (16, 16), (5, 20)])
def seq_lengths(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture(params=[(1.0, 1.0, 1.5), (0.5, 1.5, 1.0), (2.0, 0.5, 0.8)])
def cost_params(request):
    """Different (ins_cost, del_cost, trans_cost) combinations."""
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestForward:

    def test_distance(self, batch_size, seq_lengths, temperature, cost_params, device):
        """Test that OSA distances match."""
        L1, L2 = seq_lengths
        ins_cost, del_cost, trans_cost = cost_params

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device) * 2
        trans_mask = create_trans_mask(batch_size, L1, L2, device=device)

        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        distance_orihime = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_orihime), \
            f"Distance mismatch: max diff = {max_diff(distance_ref, distance_orihime)}"

    def test_distance_no_transpositions(self, batch_size, temperature, device):
        """Test OSA reduces to Levenshtein when trans_mask is all zeros."""
        L1, L2 = 10, 12
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device)
        trans_mask = torch.zeros(batch_size, L1, L2, device=device)  # No transpositions

        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        distance_orihime = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_orihime), \
            f"Distance mismatch (no trans): max diff = {max_diff(distance_ref, distance_orihime)}"

    def test_transposition_reduces_distance(self, device):
        """Test that allowing transposition can reduce distance."""
        B = 2
        L = 5
        temperature = 1.0
        ins_cost = del_cost = 1.0
        trans_cost = 0.5  # Transposition cheaper than sub

        torch.manual_seed(42)
        sub_costs = torch.ones(B, L, L, device=device)  # All substitutions cost 1

        # First batch: no transpositions allowed
        trans_mask_none = torch.zeros(B, L, L, device=device)
        # Second batch: some transpositions allowed
        trans_mask_some = torch.zeros(B, L, L, device=device)
        trans_mask_some[:, 1:, 1:] = 1.0  # Allow transpositions

        distance_none = osa_ops.forward(sub_costs, trans_mask_none, ins_cost, del_cost, trans_cost, temperature, None)[0]
        distance_some = osa_ops.forward(sub_costs, trans_mask_some, ins_cost, del_cost, trans_cost, temperature, None)[0]

        # With transpositions allowed and trans_cost < sub_cost, distance should be lower
        # (soft aggregation, so not guaranteed, but on average should be)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, cost_params, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths
        ins_cost, del_cost, trans_cost = cost_params

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device) * 2
        trans_mask = create_trans_mask(batch_size, L1, L2, device=device)

        posteriors_ref = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        posteriors_orihime = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_orihime, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_orihime)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device)
        trans_mask = create_trans_mask(batch_size, L1, L2, device=device)
        sub_costs.requires_grad_(True)

        posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_orihime = sub_costs.grad.clone()

        sub_costs_ref = sub_costs.detach().clone().requires_grad_(True)
        posteriors_ref = osa_naive(sub_costs_ref, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = sub_costs_ref.grad

        assert allclose(grad_ref, grad_orihime, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestWithGrads:

    def test_with_grads_returns_param_grads(self, device):
        """Test that osa_with_grads returns parameter gradients."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 0.8, 1.2

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        distance, posteriors, grad_T, grad_ins, grad_del, grad_trans = osa_forward_with_grads(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None
        )

        # Check shapes
        assert distance.shape == (B,)
        assert posteriors.shape == (B, L1, L2)
        assert grad_T.shape == (B,)
        assert grad_ins.shape == (B,)
        assert grad_del.shape == (B,)
        assert grad_trans.shape == (B,)

    def test_temperature_gradient(self):
        """Test d(distance)/dT against finite differences on CPU."""
        B, L1, L2 = 3, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.8, 1.1, 0.7
        eps = 1e-4

        torch.manual_seed(123)
        sub_costs = torch.rand(B, L1, L2)
        trans_mask = create_trans_mask(B, L1, L2)

        grad_T_orihime = osa_forward_with_grads(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None
        )[2]

        distance_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature + eps)
        distance_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature - eps)
        grad_T_fd = (distance_plus - distance_minus) / (2 * eps)

        assert allclose(grad_T_fd, grad_T_orihime, rtol=1e-2, atol=2e-3), \
            f"Temperature gradient mismatch: max diff = {max_diff(grad_T_fd, grad_T_orihime)}"

    def test_cost_gradients(self):
        """Test d(distance)/d{ins,del,trans} against finite differences on CPU."""
        B, L1, L2 = 3, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.8, 1.1, 0.7
        eps = 1e-4

        torch.manual_seed(321)
        sub_costs = torch.rand(B, L1, L2)
        trans_mask = create_trans_mask(B, L1, L2, density=0.5)

        _, _, _, grad_ins_orihime, grad_del_orihime, grad_trans_orihime = osa_forward_with_grads(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None
        )

        distance_ins_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost + eps, del_cost, trans_cost, temperature)
        distance_ins_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost - eps, del_cost, trans_cost, temperature)
        grad_ins_fd = (distance_ins_plus - distance_ins_minus) / (2 * eps)

        distance_del_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost + eps, trans_cost, temperature)
        distance_del_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost - eps, trans_cost, temperature)
        grad_del_fd = (distance_del_plus - distance_del_minus) / (2 * eps)

        distance_trans_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost + eps, temperature)
        distance_trans_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost - eps, temperature)
        grad_trans_fd = (distance_trans_plus - distance_trans_minus) / (2 * eps)

        assert allclose(grad_ins_fd, grad_ins_orihime, rtol=1e-2, atol=2e-3), \
            f"Insertion gradient mismatch: max diff = {max_diff(grad_ins_fd, grad_ins_orihime)}"
        assert allclose(grad_del_fd, grad_del_orihime, rtol=1e-2, atol=2e-3), \
            f"Deletion gradient mismatch: max diff = {max_diff(grad_del_fd, grad_del_orihime)}"
        assert allclose(grad_trans_fd, grad_trans_orihime, rtol=1e-2, atol=2e-3), \
            f"Transposition gradient mismatch: max diff = {max_diff(grad_trans_fd, grad_trans_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_orihime = osa_ops.marginals_hvp(sub_costs, trans_mask, V, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = osa_naive(sub_costs + eps * V, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        posteriors_minus = osa_naive(sub_costs - eps * V, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_orihime, rtol=2e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"

    def test_hvp_no_transpositions(self, device):
        """Test HVP with no transpositions."""
        B, L1, L2 = 2, 6, 7
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.0
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = torch.zeros(B, L1, L2, device=device)  # No transpositions
        V = torch.randn(B, L1, L2, device=device)

        hvp_orihime = osa_ops.marginals_hvp(sub_costs, trans_mask, V, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = osa_naive(sub_costs + eps * V, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        posteriors_minus = osa_naive(sub_costs - eps * V, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_orihime, rtol=2e-2, atol=5e-3), \
            f"HVP mismatch (no trans): max diff = {max_diff(hvp_fd, hvp_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.75, 1.25, 0.6
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        sub_costs = torch.rand(B, L1, L2)
        trans_mask = create_trans_mask(B, L1, L2, density=0.5)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)
            posteriors_ref = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

            grad_T_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature + eps)
            grad_T_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature - eps)
            grad_T_ref = (grad_T_plus - grad_T_minus) / (2 * eps)

            grad_ins_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost + eps, del_cost, trans_cost, temperature)
            grad_ins_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost - eps, del_cost, trans_cost, temperature)
            grad_ins_ref = (grad_ins_plus - grad_ins_minus) / (2 * eps)

            grad_del_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost + eps, trans_cost, temperature)
            grad_del_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost - eps, trans_cost, temperature)
            grad_del_ref = (grad_del_plus - grad_del_minus) / (2 * eps)

            grad_trans_plus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost + eps, temperature)
            grad_trans_minus, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost - eps, temperature)
            grad_trans_ref = (grad_trans_plus - grad_trans_minus) / (2 * eps)

            posteriors_plus = osa_naive(sub_costs + eps * tangent, trans_mask, ins_cost, del_cost, trans_cost, temperature)
            posteriors_minus = osa_naive(sub_costs - eps * tangent, trans_mask, ins_cost, del_cost, trans_cost, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            dP_dIns_plus = osa_naive(sub_costs, trans_mask, ins_cost + eps, del_cost, trans_cost, temperature)
            dP_dIns_minus = osa_naive(sub_costs, trans_mask, ins_cost - eps, del_cost, trans_cost, temperature)
            dP_dIns_ref = (dP_dIns_plus - dP_dIns_minus) / (2 * eps)

            dP_dDel_plus = osa_naive(sub_costs, trans_mask, ins_cost, del_cost + eps, trans_cost, temperature)
            dP_dDel_minus = osa_naive(sub_costs, trans_mask, ins_cost, del_cost - eps, trans_cost, temperature)
            dP_dDel_ref = (dP_dDel_plus - dP_dDel_minus) / (2 * eps)

            dP_dTrans_plus = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost + eps, temperature)
            dP_dTrans_minus = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost - eps, temperature)
            dP_dTrans_ref = (dP_dTrans_plus - dP_dTrans_minus) / (2 * eps)

            dP_dT_plus = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature + eps)
            dP_dT_minus = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature - eps)
            dP_dT_ref = (dP_dT_plus - dP_dT_minus) / (2 * eps)

        reference_outputs = {
            "distance": distance_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "grad_ins": grad_ins_ref,
            "grad_del": grad_del_ref,
            "grad_trans": grad_trans_ref,
            "hvp": hvp_ref,
            "dP_dIns": dP_dIns_ref,
            "dP_dDel": dP_dDel_ref,
            "dP_dTrans": dP_dTrans_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(
                    sub_costs, trans_mask, tangent, ins_cost, del_cost, trans_cost, temperature
                )
            assert_threaded_osa_correctness(outputs, reference_outputs, thread_count)
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
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs_cpu = torch.rand(B, L1, L2)
        trans_mask_cpu = create_trans_mask(B, L1, L2)
        sub_costs_cuda = sub_costs_cpu.cuda()
        trans_mask_cuda = trans_mask_cpu.cuda()

        posteriors_cpu = osa_ops.forward(sub_costs_cpu, trans_mask_cpu, ins_cost, del_cost, trans_cost, temperature, None)[1]
        posteriors_cuda = osa_ops.forward(sub_costs_cuda, trans_mask_cuda, ins_cost, del_cost, trans_cost, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_with_transpositions(self):
        """Test CPU vs CUDA with transpositions enabled."""
        B, L1, L2 = 2, 10, 12
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.5, 1.5, 0.8

        torch.manual_seed(42)
        sub_costs_cpu = torch.rand(B, L1, L2)
        trans_mask_cpu = create_trans_mask(B, L1, L2, density=0.5)
        sub_costs_cuda = sub_costs_cpu.cuda()
        trans_mask_cuda = trans_mask_cpu.cuda()

        posteriors_cpu = osa_ops.forward(sub_costs_cpu, trans_mask_cpu, ins_cost, del_cost, trans_cost, temperature, None)[1]
        posteriors_cuda = osa_ops.forward(sub_costs_cuda, trans_mask_cuda, ins_cost, del_cost, trans_cost, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch (trans): max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_backward_grad_T_parity(self):
        """Regression: CUDA grad_T must match CPU (r25 sign fix)."""
        B, L1, L2 = 4, 8, 10
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2)
        trans_mask = create_trans_mask(B, L1, L2)

        result_cpu = osa_forward_with_grads(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None
        )
        result_cuda = osa_forward_with_grads(
            sub_costs.cuda(), trans_mask.cuda(), ins_cost, del_cost, trans_cost, temperature, None
        )

        # grad_T is index 2
        assert allclose(result_cpu[2], result_cuda[2], rtol=1e-3, atol=1e-4), \
            f"grad_T CPU/CUDA mismatch: max diff = {max_diff(result_cpu[2], result_cuda[2])}"

    def test_backward_cost_grads_parity(self):
        """Regression: CUDA cost gradients must match CPU (r25)."""
        B, L1, L2 = 4, 8, 10
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.5, 1.5, 0.8

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2)
        trans_mask = create_trans_mask(B, L1, L2, density=0.5)

        result_cpu = osa_forward_with_grads(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None
        )
        result_cuda = osa_forward_with_grads(
            sub_costs.cuda(), trans_mask.cuda(), ins_cost, del_cost, trans_cost, temperature, None
        )

        # grad_ins=3, grad_del=4, grad_trans=5
        assert allclose(result_cpu[3], result_cuda[3], rtol=1e-3, atol=1e-4), \
            f"grad_ins CPU/CUDA mismatch: max diff = {max_diff(result_cpu[3], result_cuda[3])}"
        assert allclose(result_cpu[4], result_cuda[4], rtol=1e-3, atol=1e-4), \
            f"grad_del CPU/CUDA mismatch: max diff = {max_diff(result_cpu[4], result_cuda[4])}"
        assert allclose(result_cpu[5], result_cuda[5], rtol=1e-3, atol=1e-4), \
            f"grad_trans CPU/CUDA mismatch: max diff = {max_diff(result_cpu[5], result_cuda[5])}"

    def test_backward_boundary_cost_grads_parity(self):
        """Regression: CUDA boundary ins/del grads must match CPU (r49)."""
        B, max_L1, max_L2 = 3, 6, 7
        temperature = 0.9
        ins_cost, del_cost, trans_cost = 0.5, 1.4, 0.8

        torch.manual_seed(0)
        sub_costs = torch.rand(B, max_L1, max_L2)
        trans_mask = torch.zeros(B, max_L1, max_L2)
        trans_mask[:, 1:, 1:] = (torch.rand(B, max_L1 - 1, max_L2 - 1) < 0.55).float()
        lengths = torch.tensor([[6, 7], [4, 6], [5, 3]], dtype=torch.int32)

        result_cpu = osa_forward_with_grads(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
        )
        result_cuda = osa_forward_with_grads(
            sub_costs.cuda(),
            trans_mask.cuda(),
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths.cuda(),
        )

        # grad_ins=3, grad_del=4, grad_trans=5
        assert allclose(result_cpu[3], result_cuda[3], rtol=1e-3, atol=1e-4), \
            f"grad_ins CPU/CUDA mismatch: max diff = {max_diff(result_cpu[3], result_cuda[3])}"
        assert allclose(result_cpu[4], result_cuda[4], rtol=1e-3, atol=1e-4), \
            f"grad_del CPU/CUDA mismatch: max diff = {max_diff(result_cpu[4], result_cuda[4])}"
        assert allclose(result_cpu[5], result_cuda[5], rtol=1e-3, atol=1e-4), \
            f"grad_trans CPU/CUDA mismatch: max diff = {max_diff(result_cpu[5], result_cuda[5])}"

    def test_derivative_entrypoints_cpu_cuda_parity(self):
        """All OSA map derivatives and tensor-parameter autograd stay symmetric."""
        B, max_L1, max_L2 = 3, 6, 7
        ins_cost, del_cost, trans_cost, temperature = 0.75, 1.25, 0.6, 0.8
        torch.manual_seed(203)
        sub_costs_cpu = torch.rand(B, max_L1, max_L2)
        trans_mask_cpu = torch.zeros(B, max_L1, max_L2)
        trans_mask_cpu[:, 1:, 1:] = (
            torch.rand(B, max_L1 - 1, max_L2 - 1) < 0.55
        ).float()
        tangent_cpu = torch.randn_like(sub_costs_cpu)
        cotangent_cpu = torch.randn_like(sub_costs_cpu)
        lengths_cpu = torch.tensor(
            [[6, 7], [4, 5], [5, 3]], dtype=torch.int32
        )

        def collect(device):
            sub_costs = sub_costs_cpu.to(device)
            trans_mask = trans_mask_cpu.to(device)
            tangent = tangent_cpu.to(device)
            cotangent = cotangent_cpu.to(device)
            lengths = lengths_cpu.to(device)

            hvp = osa_ops.marginals_hvp(
                sub_costs,
                trans_mask,
                tangent,
                ins_cost,
                del_cost,
                trans_cost,
                temperature,
                lengths,
            )
            param_fields = tuple(
                osa_param_field(
                    sub_costs,
                    trans_mask,
                    index,
                    ins_cost,
                    del_cost,
                    trans_cost,
                    temperature,
                    lengths,
                )
                for index in range(4)
            )
            full_vjp = osa_ops.marginals_backward(
                sub_costs,
                trans_mask,
                cotangent,
                ins_cost,
                del_cost,
                trans_cost,
                temperature,
                lengths,
            )

            scores_req = sub_costs.detach().clone().requires_grad_(True)
            ins_req = sub_costs.new_tensor([ins_cost]).requires_grad_(True)
            del_req = sub_costs.new_tensor([del_cost]).requires_grad_(True)
            trans_req = sub_costs.new_tensor([trans_cost]).requires_grad_(True)
            temp_req = sub_costs.new_tensor([temperature]).requires_grad_(True)
            distance, posteriors = osa_ops.forward_t(
                scores_req,
                trans_mask,
                ins_req,
                del_req,
                trans_req,
                temp_req,
                lengths,
            )
            loss = distance.sum() + (posteriors * cotangent).sum()
            autograd = torch.autograd.grad(
                loss,
                (scores_req, ins_req, del_req, trans_req, temp_req),
            )
            return hvp, param_fields, full_vjp, autograd

        cpu = collect(torch.device("cpu"))
        cuda = collect(torch.device("cuda"))

        assert allclose(cpu[0], cuda[0], rtol=1e-3, atol=1e-4)
        for expected, actual in zip(cpu[1], cuda[1], strict=True):
            assert allclose(expected, actual, rtol=1e-3, atol=1e-4)
        for expected, actual in zip(cpu[2], cuda[2], strict=True):
            assert allclose(expected, actual, rtol=1e-3, atol=1e-4)
        for expected, actual in zip(cpu[3], cuda[3], strict=True):
            assert allclose(expected, actual, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs = torch.rand(B, max_L1, max_L2, device=device)
        trans_mask = create_trans_mask(B, max_L1, max_L2, device=device)

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            sub_costs_b = sub_costs[b:b+1, :l1, :l2]
            trans_mask_b = trans_mask[b:b+1, :l1, :l2]

            distance_ref, _ = osa_forward_naive(sub_costs_b, trans_mask_b, ins_cost, del_cost, trans_cost, temperature)
            posteriors_ref = osa_naive(sub_costs_b, trans_mask_b, ins_cost, del_cost, trans_cost, temperature)

            # Distance should match for this sequence
            assert allclose(distance_ref, distance[b:b+1]), \
                f"Distance mismatch for batch {b}: {distance_ref.item()} vs {distance[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"

    def test_variable_length_derivative_outputs_zero_padded_regions(self, device):
        """Every OSA map derivative leaves rows and columns outside lengths at zero."""
        B, max_L1, max_L2 = 3, 6, 7
        ins_cost, del_cost, trans_cost, temperature = 0.75, 1.25, 0.6, 0.8
        sub_costs = torch.rand(B, max_L1, max_L2, device=device)
        trans_mask = torch.zeros(B, max_L1, max_L2, device=device)
        trans_mask[:, 1:, 1:] = (
            torch.rand(B, max_L1 - 1, max_L2 - 1, device=device) < 0.55
        ).float()
        tangent = torch.randn_like(sub_costs)
        cotangent = torch.randn_like(sub_costs)
        lengths = torch.tensor(
            [[6, 7], [4, 5], [5, 3]], dtype=torch.int32, device=device
        )

        _, posteriors = osa_ops.forward(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
        assert_padded_region_zero("forward_posteriors", posteriors, lengths)

        with_grads = osa_forward_with_grads(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
        assert_padded_region_zero("with_grads_posteriors", with_grads[1], lengths)

        hvp = osa_ops.marginals_hvp(
            sub_costs,
            trans_mask,
            tangent,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
        assert_padded_region_zero("hvp", hvp, lengths)

        for param_type, name in (
            (0, "param_ins"),
            (1, "param_del"),
            (2, "param_trans"),
            (3, "param_temperature"),
        ):
            param = osa_param_field(
                sub_costs,
                trans_mask,
                param_type,
                ins_cost,
                del_cost,
                trans_cost,
                temperature,
                lengths,
            )
            assert_padded_region_zero(name, param, lengths)

        full_vjp = osa_ops.marginals_backward(
            sub_costs,
            trans_mask,
            cotangent,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
        assert_padded_region_zero("backward_full_grad_scores", full_vjp[0], lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 cost matrix."""
        sub_costs = torch.tensor([[[0.5]]], device=device)
        trans_mask = torch.zeros(1, 1, 1, device=device)
        temperature = 1.0
        ins_cost = del_cost = trans_cost = 1.0

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)
        posteriors_ref = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), f"Single element distance wrong: {distance.item()} vs {distance_ref.item()}"
        assert allclose(posteriors, posteriors_ref), "Single element posterior wrong"

    def test_2x2_with_transposition(self, device):
        """Test 2x2 cost matrix with transposition."""
        sub_costs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device)  # Mismatch on diagonal
        trans_mask = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]], device=device)  # Transposition valid at (1,1)
        temperature = 1.0
        ins_cost = del_cost = 1.0
        trans_cost = 0.5  # Transposition cheaper

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), f"2x2 trans distance mismatch: {max_diff(distance, distance_ref)}"

    def test_row_vector(self, device):
        """Test 1xN cost matrix."""
        sub_costs = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        trans_mask = torch.zeros(1, 1, 3, device=device)
        temperature = 1.0
        ins_cost = del_cost = trans_cost = 1.0

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), "Row vector distance mismatch"

    def test_col_vector(self, device):
        """Test Nx1 cost matrix."""
        sub_costs = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        trans_mask = torch.zeros(1, 3, 1, device=device)
        temperature = 1.0
        ins_cost = del_cost = trans_cost = 1.0

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), "Column vector distance mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard OSA)."""
        B, L1, L2 = 2, 6, 8
        temperature = 0.01
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        temperature = 10.0
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)[1]
        posteriors_ref = osa_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_zero_temperature_clamp(self, device):
        """Test that very low temperature doesn't cause NaN."""
        B, L1, L2 = 2, 5, 6
        temperature = 1e-6
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)

        assert not torch.isnan(distance).any(), "Distance contains NaN"
        assert not torch.isnan(posteriors).any(), "Posteriors contain NaN"
        assert not torch.isinf(distance).any(), "Distance contains Inf"

    def test_all_transpositions_valid(self, device):
        """Test with all transpositions valid (except boundaries)."""
        B, L = 3, 6
        temperature = 1.0
        ins_cost = del_cost = 1.0
        trans_cost = 0.5

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L, L, device=device)
        trans_mask = torch.ones(B, L, L, device=device)
        trans_mask[:, 0, :] = 0  # Boundary
        trans_mask[:, :, 0] = 0  # Boundary

        distance, posteriors = osa_ops.forward(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), \
            f"All trans valid distance mismatch: {max_diff(distance, distance_ref)}"


devices = ['cpu']
if CUDA_AVAILABLE:
    devices.append('cuda')


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not installed")
@pytest.mark.parametrize("device", devices)
class TestParamJacobian:

    def test_param_jacobian_ins(self, device):
        """Test parameter Jacobian for insertion cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        dP_dIns = osa_param_field(sub_costs, trans_mask, 0, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost + eps, del_cost, trans_cost, temperature)
        posteriors_minus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost - eps, del_cost, trans_cost, temperature)
        dP_dIns_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dIns_fd, dP_dIns, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (ins) mismatch: max diff = {max_diff(dP_dIns_fd, dP_dIns)}"

    def test_param_jacobian_del(self, device):
        """Test parameter Jacobian for deletion cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        dP_dDel = osa_param_field(sub_costs, trans_mask, 1, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost, del_cost + eps, trans_cost, temperature)
        posteriors_minus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost, del_cost - eps, trans_cost, temperature)
        dP_dDel_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dDel_fd, dP_dDel, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (del) mismatch: max diff = {max_diff(dP_dDel_fd, dP_dDel)}"

    def test_param_jacobian_trans(self, device):
        """Test parameter Jacobian for transposition cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        dP_dTrans = osa_param_field(sub_costs, trans_mask, 2, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost, del_cost, trans_cost + eps, temperature)
        posteriors_minus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost, del_cost, trans_cost - eps, temperature)
        dP_dTrans_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dTrans_fd, dP_dTrans, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (trans) mismatch: max diff = {max_diff(dP_dTrans_fd, dP_dTrans)}"

    def test_param_jacobian_temperature(self, device):
        """Test parameter Jacobian for temperature."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_mask = create_trans_mask(B, L1, L2, device=device)

        dP_dT = osa_param_field(sub_costs, trans_mask, 3, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost, del_cost, trans_cost, temperature + eps)
        posteriors_minus = osa_naive(sub_costs.cpu(), trans_mask.cpu(), ins_cost, del_cost, trans_cost, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"


# --- Memory-safety regression tests (merged from test_osa_{cpp,cuda}_memsafety.py) ---

# CPU memory-safety regression helpers/constants
INS_COST = 0.7
DEL_COST = 1.2
TRANS_COST = 0.5
TEMPERATURE = 0.9


def make_inputs():
    torch.manual_seed(123)
    batch, max_l1, max_l2 = 2, 4, 5
    sub_costs = torch.rand(batch, max_l1, max_l2, dtype=torch.float32)
    trans_mask = torch.zeros(batch, max_l1, max_l2, dtype=torch.float32)
    trans_mask[:, 1:, 1:] = 1.0
    tangent = torch.randn_like(sub_costs)
    grad_output = torch.randn_like(sub_costs)
    lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32)
    return sub_costs, trans_mask, tangent, grad_output, lengths


def call_float(sub_costs, trans_mask, tangent, grad_output, lengths):
    return osa_ops.forward(
        sub_costs, trans_mask, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, lengths
    )


def call_with_grads(sub_costs, trans_mask, tangent, grad_output, lengths):
    return osa_forward_with_grads(
        sub_costs, trans_mask, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, lengths
    )


def call_hvp(sub_costs, trans_mask, tangent, grad_output, lengths):
    return osa_ops.marginals_hvp(
        sub_costs, trans_mask, tangent, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, lengths
    )


def call_param_jacobian(sub_costs, trans_mask, tangent, grad_output, lengths):
    return osa_param_field(
        sub_costs, trans_mask, 0, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, lengths
    )


def call_backward_full(sub_costs, trans_mask, tangent, grad_output, lengths):
    return osa_ops.marginals_backward(
        sub_costs, trans_mask, grad_output, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, lengths
    )


OSA_CPU_CALLS = (
    pytest.param(call_float, id="float"),
    pytest.param(call_with_grads, id="with_grads"),
    pytest.param(call_hvp, id="hvp"),
    pytest.param(call_param_jacobian, id="param_jacobian"),
    pytest.param(call_backward_full, id="backward_full"),
)


@pytest.mark.parametrize("call", OSA_CPU_CALLS)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_valid_inputs_still_run_cpu(call):
    sub_costs, trans_mask, tangent, grad_output, lengths = make_inputs()

    result = call(sub_costs, trans_mask, tangent, grad_output, lengths)

    if isinstance(result, tuple):
        tensors = result
    else:
        tensors = tuple(result)
    assert tensors
    for tensor in tensors:
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("call", OSA_CPU_CALLS)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_lengths_values_are_range_checked_cpu(call):
    sub_costs, trans_mask, tangent, grad_output, _ = make_inputs()
    bad_lengths = torch.tensor([[5, 5], [3, 4]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 0 and 4"):
        call(sub_costs, trans_mask, tangent, grad_output, bad_lengths)


@pytest.mark.parametrize("call", OSA_CPU_CALLS)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_negative_lengths_are_rejected_cpu(call):
    sub_costs, trans_mask, tangent, grad_output, _ = make_inputs()
    bad_lengths = torch.tensor([[4, -1], [3, 4]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match=r"lengths\[0,1\] must be between 0 and 5"):
        call(sub_costs, trans_mask, tangent, grad_output, bad_lengths)


@pytest.mark.parametrize("call", OSA_CPU_CALLS)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_trans_mask_must_match_sub_costs_shape_cpu(call):
    sub_costs, _, tangent, grad_output, lengths = make_inputs()
    bad_trans_mask = torch.zeros(sub_costs.size(0), sub_costs.size(1), sub_costs.size(2) - 1)

    with pytest.raises(RuntimeError, match="trans_mask must have same shape as sub_costs"):
        call(sub_costs, bad_trans_mask, tangent, grad_output, lengths)


@pytest.mark.parametrize("call", OSA_CPU_CALLS)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_explicit_entries_reject_non_float_sub_costs_cpu(call):
    sub_costs, trans_mask, tangent, grad_output, lengths = make_inputs()

    with pytest.raises(RuntimeError, match="sub_costs must be float32"):
        call(sub_costs.to(torch.float64), trans_mask, tangent, grad_output, lengths)


@pytest.mark.parametrize("call", OSA_CPU_CALLS)
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_explicit_entries_reject_non_float_trans_mask_cpu(call):
    sub_costs, trans_mask, tangent, grad_output, lengths = make_inputs()

    with pytest.raises(RuntimeError, match="trans_mask must be float32"):
        call(sub_costs, trans_mask.to(torch.float64), tangent, grad_output, lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_hvp_rejects_non_float_tangent_cpu():
    sub_costs, trans_mask, tangent, _, lengths = make_inputs()

    with pytest.raises(
        RuntimeError, match=r"tangent must have dtype torch\.float32"
    ):
        osa_ops.marginals_hvp(
            sub_costs,
            trans_mask,
            tangent.to(torch.float64),
            INS_COST,
            DEL_COST,
            TRANS_COST,
            TEMPERATURE,
            lengths,
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_backward_full_rejects_mismatched_grad_output_shape_cpu():
    sub_costs, trans_mask, _, grad_output, lengths = make_inputs()
    # Slicing yields a non-contiguous view that would trip the earlier
    # "cotangent must be contiguous" check; make it contiguous so the
    # intended shape-mismatch guard is what rejects the input.
    bad_grad_output = grad_output[:, :-1, :].contiguous()

    with pytest.raises(RuntimeError, match="cotangent must have same shape as sub_costs"):
        osa_ops.marginals_backward(
            sub_costs,
            trans_mask,
            bad_grad_output,
            INS_COST,
            DEL_COST,
            TRANS_COST,
            TEMPERATURE,
            lengths,
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_zero_numel_oversize_dimension_is_rejected_before_int_narrowing_cpu():
    huge = 2**31
    sub_costs = torch.empty((1, 0, huge), dtype=torch.float32)
    trans_mask = torch.empty((1, 0, huge), dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"sub_costs.size\(2\) must fit in int32"):
        osa_ops.forward(
            sub_costs, trans_mask, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_noncontiguous_inputs_follow_the_osa_contract():
    sub_costs = torch.randn(2, 5, 4).transpose(1, 2)
    trans_mask = torch.rand(2, 5, 4).transpose(1, 2)
    tangent = torch.randn(2, 5, 4).transpose(1, 2)
    grad_output = torch.randn(2, 5, 4).transpose(1, 2)
    lengths = torch.tensor([[4, 3], [5, 4]], dtype=torch.int32).transpose(0, 1)
    assert not sub_costs.is_contiguous()
    assert not trans_mask.is_contiguous()
    assert not tangent.is_contiguous()
    assert not grad_output.is_contiguous()
    assert not lengths.is_contiguous()

    with pytest.raises(RuntimeError, match=r"sub_costs must be contiguous"):
        osa_ops.forward(sub_costs, trans_mask.contiguous(), 1.0, 1.0, 0.8, 0.9, None)
    with pytest.raises(RuntimeError, match=r"trans_mask must be contiguous"):
        osa_ops.forward(sub_costs.contiguous(), trans_mask, 1.0, 1.0, 0.8, 0.9, None)
    with pytest.raises(RuntimeError, match=r"tangent must be contiguous"):
        osa_ops.marginals_hvp(
            sub_costs.contiguous(),
            trans_mask.contiguous(),
            tangent,
            1.0,
            1.0,
            0.8,
            0.9,
            None,
        )
    with pytest.raises(RuntimeError, match=r"cotangent must be contiguous"):
        osa_ops.marginals_backward(
            sub_costs.contiguous(),
            trans_mask.contiguous(),
            grad_output,
            1.0,
            1.0,
            0.8,
            0.9,
            None,
        )
    with pytest.raises(RuntimeError, match=r"lengths must be contiguous"):
        osa_ops.forward(
            sub_costs.contiguous(),
            trans_mask.contiguous(),
            1.0,
            1.0,
            0.8,
            0.9,
            lengths,
        )


# CUDA memory-safety regression helpers/constants (constants prefixed CUDA_ to avoid collision with CPU block)
CUDA_INS_COST = 1.0
CUDA_DEL_COST = 0.8
CUDA_TRANS_COST = 1.3
CUDA_TEMPERATURE = 0.9


def make_osa_inputs(batch=2, l1=4, l2=5, device="cuda"):
    device = torch.device(device)
    torch.manual_seed(0)
    sub_costs = torch.rand(batch, l1, l2, device=device, dtype=torch.float32)
    trans_mask = torch.zeros(batch, l1, l2, device=device, dtype=torch.float32)
    trans_mask[:, 1:, 1:] = 1.0
    tangent = torch.rand_like(sub_costs)
    grad_output = torch.rand_like(sub_costs)
    lengths = torch.tensor(
        [[l1, l2], [max(l1 - 1, 0), max(l2 - 2, 0)]],
        device=device,
        dtype=torch.int32,
    )
    return sub_costs, trans_mask, tangent, grad_output, lengths


def run_entrypoint(name, sub_costs, trans_mask, tangent, grad_output, lengths):
    if name == "float":
        return osa_ops.forward(
            sub_costs, trans_mask, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )
    if name == "with_grads":
        return osa_forward_with_grads(
            sub_costs, trans_mask, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )
    if name == "hvp":
        return osa_ops.marginals_hvp(
            sub_costs, trans_mask, tangent, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )
    if name == "param_jacobian":
        return osa_param_field(
            sub_costs, trans_mask, 0, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )
    if name == "backward_full":
        return osa_ops.marginals_backward(
            sub_costs, trans_mask, grad_output, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )
    raise AssertionError(f"unknown entrypoint: {name}")


@pytest.mark.parametrize("entrypoint", ["float", "with_grads", "hvp", "param_jacobian", "backward_full"])
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_lengths_values_rejected_cuda(entrypoint):
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()
    bad_lengths = lengths.clone()
    bad_lengths[0, 0] = sub_costs.size(1) + 1

    with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 0 and 4"):
        run_entrypoint(entrypoint, sub_costs, trans_mask, tangent, grad_output, bad_lengths)


@pytest.mark.parametrize("entrypoint", ["float", "with_grads", "hvp", "param_jacobian", "backward_full"])
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_negative_lengths_rejected_cuda(entrypoint):
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()
    bad_lengths = lengths.clone()
    bad_lengths[1, 1] = -1

    with pytest.raises(RuntimeError, match=r"lengths\[1,1\] must be between 0 and 5"):
        run_entrypoint(entrypoint, sub_costs, trans_mask, tangent, grad_output, bad_lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_lengths_shape_and_dtype_rejected_cuda():
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()

    bad_shape = lengths[:, :1]
    with pytest.raises(RuntimeError, match=r"lengths must be \[B, 2\]"):
        run_entrypoint("float", sub_costs, trans_mask, tangent, grad_output, bad_shape)

    bad_dtype = lengths.to(torch.int64)
    with pytest.raises(RuntimeError, match="lengths must be int32"):
        run_entrypoint("float", sub_costs, trans_mask, tangent, grad_output, bad_dtype)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_lengths_must_be_cuda():
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()

    with pytest.raises(RuntimeError, match="lengths must be a CUDA tensor"):
        run_entrypoint("float", sub_costs, trans_mask, tangent, grad_output, lengths.cpu())


@pytest.mark.parametrize("entrypoint", ["float", "with_grads", "hvp", "param_jacobian", "backward_full"])
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_trans_mask_shape_rejected_cuda(entrypoint):
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()
    bad_trans_mask = torch.zeros(
        sub_costs.size(0), sub_costs.size(1), sub_costs.size(2) + 1, device=sub_costs.device
    )

    with pytest.raises(RuntimeError, match="trans_mask must have same shape as sub_costs"):
        run_entrypoint(entrypoint, sub_costs, bad_trans_mask, tangent, grad_output, lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_trans_mask_dtype_rejected_cuda():
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()

    with pytest.raises(RuntimeError, match="trans_mask must be float32"):
        run_entrypoint("with_grads", sub_costs, trans_mask.to(torch.float64), tangent, grad_output, lengths)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_hvp_tangent_shape_dtype_and_device_rejected_cuda():
    sub_costs, trans_mask, tangent, _, lengths = make_osa_inputs()

    bad_shape = torch.zeros(
        sub_costs.size(0), sub_costs.size(1), sub_costs.size(2) + 1, device=sub_costs.device
    )
    with pytest.raises(RuntimeError, match="tangent must have same shape as sub_costs"):
        osa_ops.marginals_hvp(
            sub_costs, trans_mask, bad_shape, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )

    with pytest.raises(RuntimeError, match="tangent must have dtype torch\\.float32"):
        osa_ops.marginals_hvp(
            sub_costs,
            trans_mask,
            tangent.to(torch.float64),
            CUDA_INS_COST,
            CUDA_DEL_COST,
            CUDA_TRANS_COST,
            CUDA_TEMPERATURE,
            lengths,
        )

    with pytest.raises(RuntimeError, match="tangent must be on same device as sub_costs"):
        osa_ops.marginals_hvp(
            sub_costs, trans_mask, tangent.cpu(), CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_full_grad_output_shape_dtype_and_device_rejected_cuda():
    sub_costs, trans_mask, _, grad_output, lengths = make_osa_inputs()

    bad_shape = torch.zeros(
        sub_costs.size(0), sub_costs.size(1), sub_costs.size(2) + 1, device=sub_costs.device
    )
    with pytest.raises(RuntimeError, match="cotangent must have same shape as sub_costs"):
        osa_ops.marginals_backward(
            sub_costs, trans_mask, bad_shape, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
        )

    with pytest.raises(RuntimeError, match="cotangent must have dtype torch\\.float32"):
        osa_ops.marginals_backward(
            sub_costs,
            trans_mask,
            grad_output.to(torch.float64),
            CUDA_INS_COST,
            CUDA_DEL_COST,
            CUDA_TRANS_COST,
            CUDA_TEMPERATURE,
            lengths,
        )

    with pytest.raises(RuntimeError, match="cotangent must be on same device as sub_costs"):
        osa_ops.marginals_backward(
            sub_costs,
            trans_mask,
            grad_output.cpu(),
            CUDA_INS_COST,
            CUDA_DEL_COST,
            CUDA_TRANS_COST,
            CUDA_TEMPERATURE,
            lengths,
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_large_logical_dimensions_rejected_before_allocation_cuda():
    max_int = torch.iinfo(torch.int32).max
    sub_costs = torch.empty_strided(
        (1, max_int, 1), (0, 0, 0), device="cuda", dtype=torch.float32
    )
    trans_mask = torch.empty_strided(
        (1, max_int, 1), (0, 0, 0), device="cuda", dtype=torch.float32
    )

    with pytest.raises(RuntimeError, match="L1 must be less than INT_MAX"):
        osa_ops.forward(
            sub_costs, trans_mask, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, None
        )


@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cross_device_inputs_rejected_cuda():
    source = torch.device("cuda:0")
    other = torch.device("cuda:1")
    assert source != other
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs(
        device=source
    )

    with pytest.raises(RuntimeError, match="trans_mask must be on same device as sub_costs"):
        run_entrypoint("float", sub_costs, trans_mask.to(other), tangent, grad_output, lengths)

    with pytest.raises(RuntimeError, match="tangent must be on same device as sub_costs"):
        run_entrypoint("hvp", sub_costs, trans_mask, tangent.to(other), grad_output, lengths)

    with pytest.raises(RuntimeError, match="cotangent must be on same device as sub_costs"):
        run_entrypoint("backward_full", sub_costs, trans_mask, tangent, grad_output.to(other), lengths)

    with pytest.raises(RuntimeError, match="lengths must be on same device as sub_costs"):
        run_entrypoint("float", sub_costs, trans_mask, tangent, grad_output, lengths.to(other))

    wrong_insertion_cost = torch.tensor([1.0], device=other)
    allowed = trans_mask.bool()
    allowed[:, 0, :] = False
    allowed[:, :, 0] = False
    with pytest.raises(ValueError, match=r"insertion_cost tensor must be on the same device"):
        orihime.osa_value(
            sub_costs,
            insertion_cost=wrong_insertion_cost,
            allowed_transpositions=allowed,
        )


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.multi_gpu
@TWO_CUDA_DEVICES_REQUIRED
def test_current_device_success_exercises_all_osa_entrypoints():
    target = torch.device("cuda:1")
    original_device = torch.cuda.current_device()
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs(
        batch=2, l1=4, l2=5, device=target
    )
    ins_t = sub_costs.new_tensor([1.0])
    del_t = sub_costs.new_tensor([0.8])
    trans_t = sub_costs.new_tensor([1.3])
    temp_t = sub_costs.new_tensor([0.9])

    try:
        torch.cuda.set_device(0)
        calls = (
            lambda: osa_ops.forward_t(
                sub_costs, trans_mask, ins_t, del_t, trans_t, temp_t, lengths
            ),
            lambda: osa_ops.forward(
                sub_costs, trans_mask, 1.0, 0.8, 1.3, 0.9, lengths
            ),
            lambda: osa_forward_with_grads(
                sub_costs, trans_mask, 1.0, 0.8, 1.3, 0.9, lengths
            ),
            lambda: osa_ops.marginals_hvp(
                sub_costs, trans_mask, tangent, 1.0, 0.8, 1.3, 0.9, lengths
            ),
            lambda: osa_param_field(
                sub_costs, trans_mask, 3, 1.0, 0.8, 1.3, 0.9, lengths
            ),
            lambda: osa_ops.marginals_backward(
                sub_costs, trans_mask, grad_output, 1.0, 0.8, 1.3, 0.9, lengths
            ),
        )
        for call in calls:
            result = call()
            torch.cuda.synchronize(target)
            _assert_tensors_on_device(result, target)

        allowed = trans_mask.bool()
        allowed[:, 0, :] = False
        allowed[:, :, 0] = False
        value = orihime.osa_value(
            sub_costs,
            insertion_cost=ins_t,
            deletion_cost=del_t,
            transposition_cost=trans_t,
            temperature=temp_t,
            allowed_transpositions=allowed,
        )
        torch.cuda.synchronize(target)
        assert value.device == target
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_valid_cuda_inputs_and_score_only_backward_still_work():
    sub_costs, trans_mask, tangent, grad_output, lengths = make_osa_inputs()

    distance, posteriors = osa_ops.forward(
        sub_costs, trans_mask, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
    )
    with_grads = osa_forward_with_grads(
        sub_costs, trans_mask, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
    )
    hvp = osa_ops.marginals_hvp(
        sub_costs, trans_mask, tangent, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
    )
    backward_full = osa_ops.marginals_backward(
        sub_costs, trans_mask, grad_output, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
    )

    assert torch.allclose(distance, with_grads[0], rtol=1e-5, atol=1e-6)
    assert torch.allclose(posteriors, with_grads[1], rtol=1e-5, atol=1e-6)
    assert hvp.shape == sub_costs.shape
    assert backward_full[0].shape == sub_costs.shape

    score_only_costs = sub_costs.detach().clone().requires_grad_(True)
    score, _ = osa_ops.forward(
        score_only_costs, trans_mask, CUDA_INS_COST, CUDA_DEL_COST, CUDA_TRANS_COST, CUDA_TEMPERATURE, lengths
    )
    score.sum().backward()
    assert score_only_costs.grad is not None
    assert score_only_costs.grad.shape == score_only_costs.shape


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_unused_gradient_paths_are_pruned_when_posterior_grad_is_absent():
    sub_costs, trans_mask, _, _, lengths = make_osa_inputs()

    def distance_only_backward():
        scores = sub_costs.detach().clone().requires_grad_(True)
        distance, _ = osa_ops.forward(
            scores,
            trans_mask,
            CUDA_INS_COST,
            CUDA_DEL_COST,
            CUDA_TRANS_COST,
            CUDA_TEMPERATURE,
            lengths,
        )
        distance.sum().backward()

    def explicit_zero_posterior_backward():
        scores = sub_costs.detach().clone().requires_grad_(True)
        distance, posteriors = osa_ops.forward(
            scores,
            trans_mask,
            CUDA_INS_COST,
            CUDA_DEL_COST,
            CUDA_TRANS_COST,
            CUDA_TEMPERATURE,
            lengths,
        )
        (distance.sum() + 0.0 * posteriors.sum()).backward()

    distance_only_kernels = cuda_kernel_names(distance_only_backward)
    assert_cuda_kernel_seen(distance_only_kernels, "osa_forward_diag_kernel")
    assert not any("osa_hvp_" in name for name in distance_only_kernels)
    assert not any("osa_param_grad_" in name for name in distance_only_kernels)

    explicit_zero_kernels = cuda_kernel_names(explicit_zero_posterior_backward)
    assert_cuda_kernel_seen(explicit_zero_kernels, "osa_hvp_forward_diag_kernel")
    assert_cuda_kernel_seen(
        explicit_zero_kernels, "osa_param_grad_forward_diag_kernel"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
