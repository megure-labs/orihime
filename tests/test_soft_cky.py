# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for Soft CKY parsing.
"""

import contextlib
import os
import re
import pytest
import torch
import importlib
import site
import sys
from pathlib import Path

from reference import cky_forward_naive, cky_naive
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
    cky_ops = orihime.ops._kernels["cky"]
    from operator_test_utils import cky_forward_with_grads

CUDA_AVAILABLE = torch.cuda.is_available()


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


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
    relevant = sorted(name for name in kernel_names if "cky_" in name)
    raise AssertionError(
        f"CUDA profiler did not capture {token}; CKY kernels seen: {relevant[:20]}"
    )


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(merge_scores, leaf_scores, v_merge, v_leaf, temperature):
    partition, posteriors = cky_ops.forward(merge_scores, leaf_scores, temperature)
    _, _, grad_leaf, grad_T = cky_forward_with_grads(merge_scores, leaf_scores, temperature)
    hvp = cky_ops.marginals_hvp(merge_scores, leaf_scores, v_merge, v_leaf, temperature)
    dP_dT = cky_ops.marginals_grad_temp(merge_scores, leaf_scores, temperature)
    return {
        "partition": partition,
        "posteriors": posteriors,
        "grad_leaf": grad_leaf,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def run_validation_api(api_name, merge_scores, leaf_scores):
    temperature = 0.9
    if api_name == "cky_float":
        return cky_ops.forward(merge_scores, leaf_scores, temperature)
    if api_name == "cky_with_grads":
        return cky_forward_with_grads(merge_scores, leaf_scores, temperature)
    if api_name == "cky_param_jacobian":
        return cky_ops.marginals_grad_temp(merge_scores, leaf_scores, temperature)
    raise ValueError(f"unsupported api_name={api_name}")


def reference_cky_hvp(merge_scores, leaf_scores, v_merge, v_leaf, temperature):
    def reference_posteriors(merge_scores_inner, leaf_scores_inner):
        return cky_naive(merge_scores_inner, leaf_scores_inner, temperature)

    _, hvp = torch.autograd.functional.jvp(
        reference_posteriors,
        (merge_scores, leaf_scores),
        (v_merge, v_leaf),
        create_graph=False,
    )
    return hvp


def assert_threaded_cky_correctness(outputs, reference_outputs, thread_count):
    assert allclose(reference_outputs["partition"], outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(reference_outputs['partition'], outputs['partition'])}"
    assert allclose(reference_outputs["posteriors"], outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(reference_outputs['posteriors'], outputs['posteriors'])}"
    assert allclose(reference_outputs["grad_leaf"], outputs["grad_leaf"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread grad_leaf mismatch: max diff = {max_diff(reference_outputs['grad_leaf'], outputs['grad_leaf'])}"
    assert allclose(reference_outputs["grad_T"], outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(reference_outputs['grad_T'], outputs['grad_T'])}"
    assert allclose(reference_outputs["hvp"], outputs["hvp"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(reference_outputs['hvp'], outputs['hvp'])}"
    assert allclose(reference_outputs["dP_dT"], outputs["dP_dT"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(reference_outputs['dP_dT'], outputs['dP_dT'])}"


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
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length, device=device)
        leaf_scores = torch.randn(batch_size, seq_length, device=device)

        partition_ref, _ = cky_forward_naive(merge_scores, leaf_scores, temperature)
        partition_orihime = cky_ops.forward(merge_scores, leaf_scores, temperature)[0]

        assert allclose(partition_ref, partition_orihime), \
            f"Partition mismatch: max diff = {max_diff(partition_ref, partition_orihime)}"

    def test_partition_local_temperature(self, device):
        torch.manual_seed(42)
        batch_size, seq_length = 2, 6
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length, device=device)
        leaf_scores = torch.randn(batch_size, seq_length, device=device)
        temperature = 0.5 + torch.rand(batch_size, seq_length, seq_length, device=device)

        partition_ref, _ = cky_forward_naive(merge_scores, leaf_scores, temperature)
        partition_orihime = cky_ops.forward_t(merge_scores, leaf_scores, temperature)[0]

        assert allclose(partition_ref, partition_orihime, rtol=1e-4, atol=1e-5), \
            f"Local-T partition mismatch: max diff = {max_diff(partition_ref, partition_orihime)}"

    def test_uniform_local_matches_scalar(self, device):
        torch.manual_seed(0)
        batch_size, seq_length = 2, 5
        temperature = 0.7
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length, device=device)
        leaf_scores = torch.randn(batch_size, seq_length, device=device)
        local_temperature = torch.full(
            (batch_size, seq_length, seq_length),
            temperature,
            device=device,
        )

        scalar_partition, scalar_marginals = cky_ops.forward(
            merge_scores, leaf_scores, temperature
        )
        tensor_partition, tensor_marginals = cky_ops.forward_t(
            merge_scores, leaf_scores, local_temperature
        )

        assert allclose(scalar_partition, tensor_partition)
        assert allclose(scalar_marginals, tensor_marginals, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("seq_length", (1, 2, 3))
    @pytest.mark.parametrize("score_value", (0.0, 1.0))
    @pytest.mark.parametrize("temperature", (0.05, 5.0))
    def test_chart_edge_shapes_and_temperature_limits(
        self, seq_length, score_value, temperature, device
    ):
        merge_scores = torch.full(
            (1, seq_length, seq_length, seq_length),
            score_value,
            device=device,
        )
        leaf_scores = torch.full(
            (1, seq_length), score_value, device=device
        )

        partition_ref, _ = cky_forward_naive(
            merge_scores, leaf_scores, temperature
        )
        partition, marginals = cky_ops.forward(
            merge_scores, leaf_scores, temperature
        )

        assert partition.shape == (1,)
        assert marginals.shape == merge_scores.shape
        assert allclose(partition_ref, partition)
        assert torch.isfinite(marginals).all()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestBackward:

    def test_marginals(self, batch_size, seq_length, temperature, device):
        """Test that span marginals match."""
        torch.manual_seed(42)
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length, device=device)
        leaf_scores = torch.randn(batch_size, seq_length, device=device)

        marginals_ref = cky_naive(merge_scores, leaf_scores, temperature)
        marginals_orihime = cky_ops.forward(merge_scores, leaf_scores, temperature)[1]

        assert allclose(marginals_ref, marginals_orihime, rtol=1e-3, atol=1e-4), \
            f"Marginals mismatch: max diff = {max_diff(marginals_ref, marginals_orihime)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft CKY."""
        seq_length = 6

        torch.manual_seed(42)
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length, device=device, requires_grad=True)
        leaf_scores = torch.randn(batch_size, seq_length, device=device, requires_grad=True)

        # Create tensor temperature for gradient tracking
        temp_tensor = torch.tensor([temperature], device=device)
        marginals = cky_ops.forward_t(merge_scores, leaf_scores, temp_tensor)[1]
        loss = (marginals ** 2).sum()
        loss.backward()
        grad_orihime_merge = merge_scores.grad.clone()

        merge_scores_ref = merge_scores.detach().clone().requires_grad_(True)
        leaf_scores_ref = leaf_scores.detach().clone().requires_grad_(True)
        marginals_ref = cky_naive(merge_scores_ref, leaf_scores_ref, temperature)
        loss_ref = (marginals_ref ** 2).sum()
        loss_ref.backward()
        grad_ref_merge = merge_scores_ref.grad

        assert allclose(grad_ref_merge, grad_orihime_merge, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref_merge, grad_orihime_merge)}"

    def test_local_temperature_score_gradient(self, device):
        batch_size, seq_length = 2, 6
        torch.manual_seed(123)
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length, device=device)
        leaf_scores = torch.randn(batch_size, seq_length, device=device)
        temperature = (0.5 + torch.rand(batch_size, seq_length, seq_length, device=device)).requires_grad_(True)

        score = cky_ops.forward_t(merge_scores, leaf_scores, temperature)[0].sum()
        grad_T = torch.autograd.grad(score, temperature)[0]

        assert grad_T.shape == temperature.shape
        assert grad_T.isfinite().all()
        assert grad_T.abs().sum().item() > 0.0

    def test_cpu_scalar_temperature_score_backward_without_posterior_grad(self):
        batch_size, seq_length = 2, 5
        torch.manual_seed(681)
        merge_scores = torch.randn(
            batch_size, seq_length, seq_length, seq_length,
            device="cpu",
            requires_grad=True,
        )
        leaf_scores = torch.randn(
            batch_size,
            seq_length,
            device="cpu",
            requires_grad=True,
        )
        temperature = torch.tensor([0.9], device="cpu", requires_grad=True)

        partition = cky_ops.forward_t(merge_scores, leaf_scores, temperature)[0]
        grad_merge, grad_leaf, grad_T = torch.autograd.grad(
            partition.sum(),
            (merge_scores, leaf_scores, temperature),
        )

        merge_ref = merge_scores.detach().clone().requires_grad_(True)
        leaf_ref = leaf_scores.detach().clone().requires_grad_(True)
        temp_ref = temperature.detach().clone().requires_grad_(True)
        partition_ref, _ = cky_forward_naive(merge_ref, leaf_ref, temp_ref)
        grad_merge_ref, grad_leaf_ref, grad_T_ref = torch.autograd.grad(
            partition_ref.sum(),
            (merge_ref, leaf_ref, temp_ref),
        )

        assert allclose(grad_merge_ref, grad_merge, rtol=1e-3, atol=1e-4), \
            f"Score grad_merge mismatch: max diff = {max_diff(grad_merge_ref, grad_merge)}"
        assert allclose(grad_leaf_ref, grad_leaf, rtol=1e-3, atol=1e-4), \
            f"Score grad_leaf mismatch: max diff = {max_diff(grad_leaf_ref, grad_leaf)}"
        assert allclose(grad_T_ref, grad_T, rtol=1e-3, atol=1e-4), \
            f"Score grad_T mismatch: max diff = {max_diff(grad_T_ref, grad_T)}"

    def test_full_vjp_matches_reference(self):
        batch_size, seq_length = 2, 5
        torch.manual_seed(8041)
        merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length)
        leaf_scores = torch.randn(batch_size, seq_length)
        grad_posteriors = torch.randn_like(merge_scores)
        temperature = 0.85

        grad_merge, grad_leaf, grad_T = cky_ops.marginals_backward(
            merge_scores,
            leaf_scores,
            grad_posteriors,
            temperature,
        )

        merge_ref = merge_scores.detach().clone().requires_grad_(True)
        leaf_ref = leaf_scores.detach().clone().requires_grad_(True)
        temp_ref = torch.tensor([temperature], requires_grad=True)
        posteriors_ref = cky_naive(merge_ref, leaf_ref, temp_ref)
        grad_merge_ref, grad_leaf_ref, grad_T_ref = torch.autograd.grad(
            (posteriors_ref * grad_posteriors).sum(),
            (merge_ref, leaf_ref, temp_ref),
        )

        assert allclose(grad_merge_ref, grad_merge, rtol=2e-2, atol=3e-3), \
            f"Full VJP merge mismatch: max diff = {max_diff(grad_merge_ref, grad_merge)}"
        assert allclose(grad_leaf_ref, grad_leaf, rtol=2e-2, atol=3e-3), \
            f"Full VJP leaf mismatch: max diff = {max_diff(grad_leaf_ref, grad_leaf)}"
        assert allclose(grad_T_ref, grad_T, rtol=2e-2, atol=3e-3), \
            f"Full VJP temperature mismatch: max diff = {max_diff(grad_T_ref, grad_T)}"

    def test_public_leaf_map_matches_raw_leaf_derivative(self):
        torch.manual_seed(8044)
        merge_scores = torch.randn(2, 5, 5, 5)
        leaf_scores = torch.randn(2, 5)
        temperature = 0.9
        leaf_map = orihime.cky_leaf_map(
            merge_scores,
            leaf_scores,
            temperature=temperature,
        )
        raw_leaf_map = cky_ops.value_grad_params(
            merge_scores,
            leaf_scores,
            temperature,
        )[0]

        assert allclose(leaf_map, raw_leaf_map)
        assert leaf_map.shape == leaf_scores.shape
        assert not leaf_map.requires_grad


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, N = 2, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        merge_scores = torch.randn(B, N, N, N, device=device)
        leaf_scores = torch.randn(B, N, device=device)
        V_merge = torch.randn(B, N, N, N, device=device)
        V_leaf = torch.randn(B, N, device=device)

        hvp_orihime = cky_ops.marginals_hvp(merge_scores, leaf_scores, V_merge, V_leaf, temperature)

        marginals_plus = cky_naive(merge_scores + eps * V_merge, leaf_scores + eps * V_leaf, temperature)
        marginals_minus = cky_naive(merge_scores - eps * V_merge, leaf_scores - eps * V_leaf, temperature)
        hvp_fd = (marginals_plus - marginals_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_orihime, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_orihime)}"


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestInputValidation:

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
    @pytest.mark.parametrize(
        "api_name",
        ("cky_float", "cky_with_grads", "cky_param_jacobian"),
    )
    def test_non_cubic_merge_scores_raise(self, device_type, api_name):
        device = torch.device(device_type)
        merge_scores = torch.randn(1, 4, 3, 4, device=device)
        leaf_scores = torch.randn(1, 4, device=device)

        with pytest.raises(RuntimeError, match=r"merge_scores must be \[B, n, n, n\]"):
            run_validation_api(api_name, merge_scores, leaf_scores)

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
    @pytest.mark.parametrize(
        "api_name",
        ("cky_float", "cky_with_grads", "cky_param_jacobian"),
    )
    def test_leaf_scores_shape_mismatch_raises(self, device_type, api_name):
        device = torch.device(device_type)
        merge_scores = torch.randn(1, 4, 4, 4, device=device)
        leaf_scores = torch.randn(1, 3, device=device)

        with pytest.raises(RuntimeError, match=r"leaf_scores must be \[B, n\]"):
            run_validation_api(api_name, merge_scores, leaf_scores)

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
        device = torch.device(device_type)
        merge_scores = torch.randn(1, 0, 0, 0, device=device)
        leaf_scores = torch.randn(1, 0, device=device)

        with pytest.raises(RuntimeError, match=r"cky requires n > 0"):
            cky_ops.forward(merge_scores, leaf_scores, 1.0)

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
    @pytest.mark.parametrize(
        ("bad_arg", "match"),
        [
            ("v_merge", r"merge tangent must have same shape as merge_scores"),
            ("v_leaf", r"leaf tangent must have same shape as leaf_scores"),
        ],
    )
    def test_hvp_shape_mismatch_raises(self, device_type, bad_arg, match):
        device = torch.device(device_type)
        merge_scores = torch.randn(1, 4, 4, 4, device=device)
        leaf_scores = torch.randn(1, 4, device=device)
        v_merge = torch.randn(1, 4, 3, 4, device=device) if bad_arg == "v_merge" else torch.randn_like(merge_scores)
        v_leaf = torch.randn(1, 3, device=device) if bad_arg == "v_leaf" else torch.randn_like(leaf_scores)

        with pytest.raises(RuntimeError, match=match):
            cky_ops.marginals_hvp(merge_scores, leaf_scores, v_merge, v_leaf, 1.0)

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
    def test_backward_full_shape_mismatch_raises(self, device_type):
        device = torch.device(device_type)
        merge_scores = torch.randn(1, 4, 4, 4, device=device)
        leaf_scores = torch.randn(1, 4, device=device)
        grad_posteriors = torch.randn(1, 4, 3, 4, device=device)

        with pytest.raises(RuntimeError, match=r"cotangent must have same shape as merge_scores"):
            cky_ops.marginals_backward(merge_scores, leaf_scores, grad_posteriors, 1.0)

    @pytest.mark.parametrize(
        "bad_input",
        ("merge_scores", "leaf_scores", "temperature", "v_merge", "v_leaf", "grad_posteriors"),
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
        merge_scores = torch.randn(1, 4, 4, 4, device=device)
        leaf_scores = torch.randn(1, 4, device=device)
        temperature = 0.9
        local_temperature = torch.rand(1, 4, 4, device=device) + 0.5
        v_merge = torch.randn_like(merge_scores)
        v_leaf = torch.randn_like(leaf_scores)
        grad_posteriors = torch.randn_like(merge_scores)

        if bad_input == "merge_scores":
            bad_merge = merge_scores.transpose(2, 3)
            call = lambda: cky_ops.forward(bad_merge, leaf_scores, temperature)
        elif bad_input == "leaf_scores":
            leaf_storage = torch.randn(1, 8, device=device)
            bad_leaf = leaf_storage[:, ::2]
            call = lambda: cky_ops.forward(merge_scores, bad_leaf, temperature)
        elif bad_input == "temperature":
            bad_temperature = local_temperature.transpose(1, 2)
            call = lambda: cky_ops.forward_t(
                merge_scores, leaf_scores, bad_temperature
            )
        elif bad_input == "v_merge":
            bad_v_merge = v_merge.transpose(2, 3)
            call = lambda: cky_ops.marginals_hvp(
                merge_scores, leaf_scores, bad_v_merge, v_leaf, temperature
            )
        elif bad_input == "v_leaf":
            leaf_storage = torch.randn(1, 8, device=device)
            bad_v_leaf = leaf_storage[:, ::2]
            call = lambda: cky_ops.marginals_hvp(
                merge_scores, leaf_scores, v_merge, bad_v_leaf, temperature
            )
        else:
            bad_grad_posteriors = grad_posteriors.transpose(2, 3)
            call = lambda: cky_ops.marginals_backward(
                merge_scores, leaf_scores, bad_grad_posteriors, temperature
            )

        expected_match = {
            "merge_scores": r"merge_scores must be contiguous",
            "leaf_scores": r"leaf_scores must be contiguous",
            "temperature": r"temperature must be contiguous",
            "v_merge": r"merge tangent must be contiguous",
            "v_leaf": r"leaf tangent must be contiguous",
            "grad_posteriors": r"cotangent must be contiguous",
        }[bad_input]
        with pytest.raises(RuntimeError, match=expected_match):
            call()

    @pytest.mark.multi_gpu
    @TWO_CUDA_DEVICES_REQUIRED
    def test_cuda_leaf_scores_must_match_merge_device(self):
        merge_scores = torch.randn(1, 4, 4, 4, device="cuda:0")
        leaf_scores = torch.randn(1, 4, device="cuda:1")
        assert merge_scores.device.index != leaf_scores.device.index

        with pytest.raises(RuntimeError, match=r"leaf_scores must be on same device as merge_scores"):
            cky_ops.forward(merge_scores, leaf_scores, 1.0)

    @pytest.mark.multi_gpu
    @TWO_CUDA_DEVICES_REQUIRED
    def test_cuda_temperature_must_match_merge_device(self):
        merge_scores = torch.randn(1, 4, 4, 4, device="cuda:0")
        leaf_scores = torch.randn(1, 4, device="cuda:0")
        temperature = torch.full((1, 4, 4), 0.9, device="cuda:1")
        assert merge_scores.device.index != temperature.device.index

        with pytest.raises(RuntimeError, match=r"temperature must be on same device as merge_scores"):
            cky_ops.forward_t(merge_scores, leaf_scores, temperature)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, N = 8, 6
        temperature = 1.0
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        merge_scores = torch.randn(B, N, N, N)
        leaf_scores = torch.randn(B, N)
        v_merge = torch.randn(B, N, N, N)
        v_leaf = torch.randn(B, N)

        with torch_num_threads(1):
            partition_ref, _ = cky_forward_naive(merge_scores, leaf_scores, temperature)

            merge_scores_ref = merge_scores.detach().clone().requires_grad_(True)
            leaf_scores_ref = leaf_scores.detach().clone().requires_grad_(True)
            partition_for_grads, _ = cky_forward_naive(
                merge_scores_ref, leaf_scores_ref, temperature
            )
            posteriors_ref, grad_leaf_ref = torch.autograd.grad(
                partition_for_grads.sum(),
                (merge_scores_ref, leaf_scores_ref),
            )

            partition_temp_plus, _ = cky_forward_naive(
                merge_scores, leaf_scores, temperature + eps
            )
            partition_temp_minus, _ = cky_forward_naive(
                merge_scores, leaf_scores, temperature - eps
            )
            grad_T_ref = (partition_temp_plus - partition_temp_minus) / (2 * eps)

            posteriors_temp_plus = cky_naive(
                merge_scores, leaf_scores, temperature + eps
            )
            posteriors_temp_minus = cky_naive(
                merge_scores, leaf_scores, temperature - eps
            )
            dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

            hvp_ref = reference_cky_hvp(
                merge_scores, leaf_scores, v_merge, v_leaf, temperature
            )

        reference_outputs = {
            "partition": partition_ref,
            "posteriors": posteriors_ref,
            "grad_leaf": grad_leaf_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(
                    merge_scores, leaf_scores, v_merge, v_leaf, temperature
                )
            assert_threaded_cky_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)

    def test_local_temperature_thread_count_stability(self):
        B, N = 6, 6
        thread_counts = (1, 2, 4)

        torch.manual_seed(321)
        merge_scores = torch.randn(B, N, N, N)
        leaf_scores = torch.randn(B, N)
        temperature = 0.5 + torch.rand(B, N, N)

        with torch_num_threads(1):
            partition_ref, _ = cky_forward_naive(merge_scores, leaf_scores, temperature)
            posteriors_ref = cky_naive(merge_scores, leaf_scores, temperature)

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = cky_ops.forward_t(merge_scores, leaf_scores, temperature)
            partition, posteriors = outputs
            assert allclose(partition_ref, partition, rtol=1e-4, atol=1e-5), \
                f"{thread_count}-thread local-T partition mismatch: max diff = {max_diff(partition_ref, partition)}"
            assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
                f"{thread_count}-thread local-T posterior mismatch: max diff = {max_diff(posteriors_ref, posteriors)}"
            outputs_by_thread[thread_count] = {
                "partition": partition,
                "posteriors": posteriors,
            }

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, N = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        merge_scores_cpu = torch.randn(B, N, N, N)
        leaf_scores_cpu = torch.randn(B, N)
        merge_scores_cuda = merge_scores_cpu.cuda()
        leaf_scores_cuda = leaf_scores_cpu.cuda()

        marginals_cpu = cky_ops.forward(merge_scores_cpu, leaf_scores_cpu, temperature)[1]
        marginals_cuda = cky_ops.forward(merge_scores_cuda, leaf_scores_cuda, temperature)[1]

        assert allclose(marginals_cpu, marginals_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(marginals_cpu, marginals_cuda)}"

    def test_local_temperature_consistency(self):
        B, N = 2, 6
        torch.manual_seed(7)
        merge_scores_cpu = torch.randn(B, N, N, N)
        leaf_scores_cpu = torch.randn(B, N)
        temperature_cpu = 0.5 + torch.rand(B, N, N)

        merge_scores_cuda = merge_scores_cpu.cuda()
        leaf_scores_cuda = leaf_scores_cpu.cuda()
        temperature_cuda = temperature_cpu.cuda()

        partition_cpu, marginals_cpu = cky_ops.forward_t(
            merge_scores_cpu, leaf_scores_cpu, temperature_cpu
        )
        partition_cuda, marginals_cuda = cky_ops.forward_t(
            merge_scores_cuda, leaf_scores_cuda, temperature_cuda
        )

        assert allclose(partition_cpu, partition_cuda, rtol=1e-4, atol=1e-5)
        assert allclose(marginals_cpu, marginals_cuda, rtol=1e-3, atol=1e-4)

    def test_derivative_consistency(self):
        B, N = 2, 6
        temperature = 0.9

        torch.manual_seed(5108)
        merge_scores_cpu = torch.randn(B, N, N, N)
        leaf_scores_cpu = torch.randn(B, N)
        v_merge_cpu = torch.randn_like(merge_scores_cpu)
        v_leaf_cpu = torch.randn_like(leaf_scores_cpu)
        grad_posteriors_cpu = torch.randn_like(merge_scores_cpu)

        def raw_outputs(device):
            merge_scores = merge_scores_cpu.to(device)
            leaf_scores = leaf_scores_cpu.to(device)
            v_merge = v_merge_cpu.to(device)
            v_leaf = v_leaf_cpu.to(device)
            grad_posteriors = grad_posteriors_cpu.to(device)
            value, marginals, grad_leaf, grad_T = cky_forward_with_grads(
                merge_scores, leaf_scores, temperature
            )
            hvp = cky_ops.marginals_hvp(
                merge_scores, leaf_scores, v_merge, v_leaf, temperature
            )
            leaf_hvp = cky_ops.marginals_grad_leaf(
                merge_scores, leaf_scores, v_leaf, temperature
            )
            dP_dT = cky_ops.marginals_grad_temp(
                merge_scores, leaf_scores, temperature
            )
            full_vjp = cky_ops.marginals_backward(
                merge_scores, leaf_scores, grad_posteriors, temperature
            )
            return (value, marginals, grad_leaf, grad_T, hvp, leaf_hvp, dP_dT, *full_vjp)

        cpu_outputs = raw_outputs("cpu")
        cuda_outputs = raw_outputs("cuda")
        for cpu_output, cuda_output in zip(cpu_outputs, cuda_outputs):
            assert allclose(cpu_output, cuda_output, rtol=2e-3, atol=2e-4), \
                f"CPU/CUDA derivative mismatch: max diff = {max_diff(cpu_output, cuda_output)}"

        def autograd_outputs(device):
            merge_scores = merge_scores_cpu.to(device).requires_grad_(True)
            leaf_scores = leaf_scores_cpu.to(device).requires_grad_(True)
            temp = torch.tensor([temperature], device=device, requires_grad=True)
            value, marginals = cky_ops.forward_t(merge_scores, leaf_scores, temp)
            loss = value.sum() + 0.25 * marginals.square().sum()
            return torch.autograd.grad(loss, (merge_scores, leaf_scores, temp))

        cpu_autograd = autograd_outputs("cpu")
        cuda_autograd = autograd_outputs("cuda")
        for cpu_output, cuda_output in zip(cpu_autograd, cuda_autograd):
            assert allclose(cpu_output, cuda_output, rtol=2e-3, atol=2e-4), \
                f"CPU/CUDA autograd mismatch: max diff = {max_diff(cpu_output, cuda_output)}"

    def test_score_backward_skips_unused_posterior_hvp(self):
        B, N = 1, 8

        def score_only_backward():
            torch.manual_seed(8675309)
            merge_scores = torch.randn(B, N, N, N, device="cuda", requires_grad=True)
            leaf_scores = torch.randn(B, N, device="cuda", requires_grad=True)
            temperature = torch.tensor([0.9], device="cuda", requires_grad=True)
            score, _ = cky_ops.forward_t(merge_scores, leaf_scores, temperature)
            score.sum().backward()

        def explicit_zero_posterior_backward():
            torch.manual_seed(8675309)
            merge_scores = torch.randn(B, N, N, N, device="cuda", requires_grad=True)
            leaf_scores = torch.randn(B, N, device="cuda", requires_grad=True)
            temperature = torch.tensor([0.9], device="cuda", requires_grad=True)
            score, posteriors = cky_ops.forward_t(merge_scores, leaf_scores, temperature)
            (score.sum() + 0.0 * posteriors.sum()).backward()

        score_only_kernels = cuda_kernel_names(score_only_backward)
        assert_cuda_kernel_seen(score_only_kernels, "cky_forward_persistent_kernel")
        assert not any("cky_hvp_" in name for name in score_only_kernels)
        assert not any("cky_param_grad_" in name for name in score_only_kernels)

        explicit_zero_kernels = cuda_kernel_names(explicit_zero_posterior_backward)
        assert_cuda_kernel_seen(explicit_zero_kernels, "cky_hvp_forward_persistent_kernel")
        assert_cuda_kernel_seen(explicit_zero_kernels, "cky_param_grad_forward_persistent_kernel")


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestWarpReductionChunking:

    def test_large_span_partition_cuda(self):
        """Exercise the >32-split warp-reduction chunk path."""
        B, N = 1, 40
        temperature = 1.0

        torch.manual_seed(42)
        merge_scores = torch.randn(B, N, N, N, device="cuda")
        leaf_scores = torch.randn(B, N, device="cuda")

        partition_ref, _ = cky_forward_naive(
            merge_scores.cpu(), leaf_scores.cpu(), temperature
        )
        partition_orihime = cky_ops.forward(merge_scores, leaf_scores, temperature)[0]

        assert allclose(partition_ref, partition_orihime, rtol=1e-4, atol=1e-4), \
            f"Large-span partition mismatch: max diff = {max_diff(partition_ref, partition_orihime)}"


# --- Memory-safety regression tests (merged from test_cky_{cpp,cuda}_memsafety.py) ---

ROOT = Path(__file__).resolve().parents[1]
KERNELS_CPU = ROOT / "src" / "cky" / "kernels_cpu.cpp"
KERNELS_CPU_H = ROOT / "src" / "cky" / "kernels_cpu.h"
TORCH_CPU = ROOT / "src" / "cky" / "torch_cpu.cpp"

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_CU = REPO_ROOT / "src" / "cky" / "kernels_gpu.cu"
KERNELS_CUH = REPO_ROOT / "src" / "cky" / "kernels_gpu.cuh"
TORCH_CUDA_CPP = REPO_ROOT / "src" / "cky" / "torch_cuda.cpp"


def _schema_missing(op_name):
    with pytest.raises(RuntimeError, match=r"(Could not find schema|Found no matching schema)"):
        torch._C._dispatch_find_schema_or_throw(op_name, "")


def _read(path):
    return path.read_text()


def _small_cky_inputs(device):
    torch.manual_seed(2707)
    B, N = 1, 4
    merge_scores = torch.randn(B, N, N, N, device=device)
    leaf_scores = torch.randn(B, N, device=device)
    v_merge = torch.randn(B, N, N, N, device=device)
    v_leaf = torch.randn(B, N, device=device)
    grad_posteriors = torch.randn(B, N, N, N, device=device)
    return merge_scores, leaf_scores, v_merge, v_leaf, grad_posteriors


# --- CPU memory-safety regressions (from test_cky_cpp_memsafety.py) ---


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_local_temperature_cpu_indices_are_widened():
    kernels = KERNELS_CPU.read_text()
    torch_cpu = TORCH_CPU.read_text()

    assert "return T[(size_t)b * n * n + i * n + j];" not in kernels
    assert "grad_T[(size_t)b * n * n + i * n + j]" not in kernels
    assert "return T[(size_t)b * n * n + (size_t)i * n + j];" in kernels
    assert "grad_T[(size_t)b * n * n + (size_t)i * n + j]" in kernels
    assert "kMaxCkyCpuNForIntChartIndex" in torch_cpu
    assert "cky CPU sequence length is too large" in torch_cpu


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_chart_overflow_guard_precedes_chart_allocation():
    source = TORCH_CPU.read_text()
    validation_start = source.index("void validate_cky_chart_shapes_cpu")
    guard = source.index(
        "n <= kMaxCkyCpuNForIntChartIndex",
        validation_start,
    )
    allocation = source.index("torch::Tensor Z = torch::zeros", validation_start)

    assert validation_start < guard < allocation
    assert "constexpr int64_t kMaxCkyCpuNForIntChartIndex = 46340" in source


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_dead_cpu_helpers_are_not_shipped_or_registered():
    kernels = KERNELS_CPU.read_text()
    header = KERNELS_CPU_H.read_text()

    for token in ("cky_thermodynamics_cpu", "cky_forward_pos_feats_cpu", "merge_weights"):
        assert token not in kernels
        assert token not in header

    _schema_missing("orihime::cky_thermodynamics")
    _schema_missing("orihime::cky_forward_pos_feats")


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_local_temperature_path_still_matches_scalar_temperature():
    torch.manual_seed(2104)
    batch_size, seq_length = 2, 5
    temperature = 0.8

    merge_scores = torch.randn(batch_size, seq_length, seq_length, seq_length)
    leaf_scores = torch.randn(batch_size, seq_length)
    local_temperature = torch.full(
        (batch_size, seq_length, seq_length),
        temperature,
        requires_grad=True,
    )

    scalar_partition, scalar_posteriors = cky_ops.forward(
        merge_scores, leaf_scores, temperature
    )
    local_partition, local_posteriors = cky_ops.forward_t(
        merge_scores, leaf_scores, local_temperature
    )

    assert torch.allclose(local_partition, scalar_partition, rtol=1e-4, atol=1e-5)
    assert torch.allclose(local_posteriors, scalar_posteriors, rtol=1e-3, atol=1e-4)

    (grad_temperature,) = torch.autograd.grad(local_partition.sum(), (local_temperature,))
    assert grad_temperature.shape == local_temperature.shape
    assert torch.isfinite(grad_temperature).all()


@pytest.mark.skipif(not ORIHIME_AVAILABLE, reason="orihime not built")
def test_cpu_local_temperature_bad_shape_rejects_cleanly():
    merge_scores = torch.randn(1, 4, 4, 4)
    leaf_scores = torch.randn(1, 4)
    bad_temperature = torch.ones(1, 4, 3)

    with pytest.raises(RuntimeError, match=r"temperature must be scalar or \[B, n, n\]"):
        cky_ops.forward_t(merge_scores, leaf_scores, bad_temperature)


# --- CUDA memory-safety regressions (from test_cky_cuda_memsafety.py) ---


def test_cuda_dead_diag_and_thermo_code_removed():
    source = _read(KERNELS_CU)
    header = _read(KERNELS_CUH)
    combined = source + "\n" + header

    for token in ("_diag_kernel", "cky_thermo", "cky_thermodynamics"):
        assert token not in combined


def test_cuda_residual_index_sites_are_widened():
    source = _read(KERNELS_CU)

    assert "return d_T[(size_t)b * n * n + (size_t)i * n + j];" in source
    assert "grad_T[(size_t)b * n * n + (size_t)i * n + j] = grad_T_ij;" in source
    assert "grad_leaf[idx] = beta[chart_idx];" in source
    assert "HVP_leaf[idx] = d_beta[chart_idx];" in source
    assert "dP_dT_leaf[idx] = W[chart_idx];" in source

    stale_patterns = (
        "return d_T[(size_t)b * n * n + i * n + j]",
        "grad_T[(size_t)b * n * n + i * n + j]",
        "grad_leaf[idx] = beta[b * n * n + i * n + i]",
        "HVP_leaf[idx] = d_beta[b * n * n + i * n + i]",
        "dP_dT_leaf[idx] = W[b * n * n + i * n + i]",
    )
    for pattern in stale_patterns:
        assert pattern not in source


def test_cky_cuda_wrappers_have_device_guards():
    source = _read(TORCH_CUDA_CPP)
    assert source.count("ORIHIME_CUDA_GUARD(merge_scores);") >= 6


@pytest.mark.multi_gpu
@pytest.mark.skipif(not (ORIHIME_AVAILABLE and CUDA_AVAILABLE), reason="CUDA orihime backend required")
@TWO_CUDA_DEVICES_REQUIRED
def test_cuda_public_ops_run_on_input_device_when_current_device_differs():
    input_device = torch.device("cuda:1")
    current_device = 0
    merge_scores, leaf_scores, v_merge, v_leaf, grad_posteriors = _small_cky_inputs(input_device)
    assert input_device.index != current_device

    with torch.cuda.device(current_device):
        partition, posteriors = cky_ops.forward(merge_scores, leaf_scores, 0.9)
        _, _, grad_leaf, grad_T = cky_forward_with_grads(merge_scores, leaf_scores, 0.9)
        hvp = cky_ops.marginals_hvp(merge_scores, leaf_scores, v_merge, v_leaf, 0.9)
        leaf_hvp = cky_ops.marginals_grad_leaf(
            merge_scores, leaf_scores, v_leaf, 0.9
        )
        dP_dT = cky_ops.marginals_grad_temp(merge_scores, leaf_scores, 0.9)
        grad_merge, full_grad_leaf, full_grad_T = cky_ops.marginals_backward(
            merge_scores,
            leaf_scores,
            grad_posteriors,
            0.9,
        )
        assert torch.cuda.current_device() == current_device

    for tensor in (
        partition,
        posteriors,
        grad_leaf,
        grad_T,
        hvp,
        leaf_hvp,
        dP_dT,
        grad_merge,
        full_grad_leaf,
        full_grad_T,
    ):
        assert tensor.device == input_device
        assert torch.isfinite(tensor).all()

    torch.cuda.synchronize(input_device)


@pytest.mark.multi_gpu
@pytest.mark.skipif(not (ORIHIME_AVAILABLE and CUDA_AVAILABLE), reason="CUDA orihime backend required")
@TWO_CUDA_DEVICES_REQUIRED
def test_cuda_autograd_backward_uses_input_device_when_current_device_differs():
    input_device = torch.device("cuda:1")
    current_device = 0
    merge_scores, leaf_scores, _, _, _ = _small_cky_inputs(input_device)
    assert input_device.index != current_device
    merge_scores.requires_grad_(True)
    leaf_scores.requires_grad_(True)
    temperature = torch.tensor([0.9], device=input_device, requires_grad=True)

    with torch.cuda.device(current_device):
        partition, posteriors = cky_ops.forward_t(merge_scores, leaf_scores, temperature)
        loss = partition.sum() + 0.1 * posteriors.square().sum()
        loss.backward()
        assert torch.cuda.current_device() == current_device

    for tensor in (merge_scores.grad, leaf_scores.grad, temperature.grad):
        assert tensor is not None
        assert tensor.device == input_device
        assert torch.isfinite(tensor).all()

    torch.cuda.synchronize(input_device)


@pytest.mark.multi_gpu
@pytest.mark.skipif(not (ORIHIME_AVAILABLE and CUDA_AVAILABLE), reason="CUDA orihime backend required")
@TWO_CUDA_DEVICES_REQUIRED
@pytest.mark.parametrize("bad_tangent", ("v_merge", "v_leaf"))
def test_cuda_hvp_rejects_cross_device_tangent(bad_tangent):
    merge_scores, leaf_scores, _, v_leaf, _ = _small_cky_inputs(torch.device("cuda:0"))
    v_merge = torch.randn_like(merge_scores, device="cuda:0")
    if bad_tangent == "v_merge":
        v_merge = torch.randn_like(merge_scores, device="cuda:1")
        bad_device = v_merge.device
        good_device = v_leaf.device
    else:
        v_leaf = torch.randn_like(leaf_scores, device="cuda:1")
        bad_device = v_leaf.device
        good_device = v_merge.device
    assert bad_device.index != merge_scores.device.index
    assert good_device == merge_scores.device

    expected = f"{bad_tangent} must be on same device as merge_scores"
    with pytest.raises(RuntimeError, match=expected):
        cky_ops.marginals_hvp(merge_scores, leaf_scores, v_merge, v_leaf, 0.9)


@pytest.mark.multi_gpu
@pytest.mark.skipif(not (ORIHIME_AVAILABLE and CUDA_AVAILABLE), reason="CUDA orihime backend required")
@TWO_CUDA_DEVICES_REQUIRED
def test_cuda_backward_full_rejects_cross_device_grad_posteriors():
    merge_scores, leaf_scores, _, _, _ = _small_cky_inputs(torch.device("cuda:0"))
    grad_posteriors = torch.randn_like(merge_scores, device="cuda:1")
    assert merge_scores.device.index != grad_posteriors.device.index

    with pytest.raises(RuntimeError, match="cotangent must be on same device as merge_scores"):
        cky_ops.marginals_backward(merge_scores, leaf_scores, grad_posteriors, 0.9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
