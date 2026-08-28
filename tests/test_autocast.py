# SPDX-License-Identifier: Apache-2.0
"""Autocast coverage for the uniform and named low-level surfaces."""

import pytest
import torch

import orihime

sw_ops = orihime.ops._kernels["sw"]
sw_affine_ops = orihime.ops._kernels["sw_affine"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutocastSW:
    def test_sw_autocast_promotes_to_fp32(self):
        scores = torch.randn(
            2, 8, 10, device="cuda", dtype=torch.float16
        )

        with torch.autocast("cuda", dtype=torch.float16):
            value = orihime.sw_value(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            )
            marginals = orihime.sw(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32

    def test_sw_autocast_bf16(self):
        scores = torch.randn(
            2, 8, 10, device="cuda", dtype=torch.bfloat16
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = orihime.sw_value(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            )
            marginals = orihime.sw(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32

    def test_sw_affine_autocast(self):
        scores = torch.randn(
            2, 8, 10, device="cuda", dtype=torch.float16
        )

        with torch.autocast("cuda", dtype=torch.float16):
            value = orihime.sw_affine_value(
                scores,
                gap_open_score=-2.0,
                gap_extend_score=-0.5,
                temperature=1.0,
            )
            marginals = orihime.sw_affine(
                scores,
                gap_open_score=-2.0,
                gap_extend_score=-0.5,
                temperature=1.0,
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32

    def test_autocast_gradient_flow(self):
        scores = torch.randn(
            2,
            8,
            10,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )

        with torch.autocast("cuda", dtype=torch.float16):
            loss = orihime.sw_value(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            ).sum()

        loss.backward()

        assert scores.grad is not None
        assert scores.grad.dtype == torch.float16

    def test_autocast_numerical_stability(self):
        torch.manual_seed(42)
        scores_fp16 = torch.randn(
            4, 16, 20, device="cuda", dtype=torch.float16
        )
        scores_fp32 = scores_fp16.float()

        with torch.autocast("cuda", dtype=torch.float16):
            value_autocast = orihime.sw_value(
                scores_fp16,
                gap_score=-1.0,
                temperature=1.0,
            )
            map_autocast = orihime.sw(
                scores_fp16,
                gap_score=-1.0,
                temperature=1.0,
            )

        value_fp32 = orihime.sw_value(
            scores_fp32,
            gap_score=-1.0,
            temperature=1.0,
        )
        map_fp32 = orihime.sw(
            scores_fp32,
            gap_score=-1.0,
            temperature=1.0,
        )

        torch.testing.assert_close(
            value_autocast, value_fp32, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            map_autocast, map_fp32, rtol=1e-5, atol=1e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutocastNoEffect:
    def test_fp32_unchanged_under_autocast(self):
        scores = torch.randn(2, 8, 10, device="cuda")

        with torch.autocast("cuda", dtype=torch.float16):
            value = orihime.sw_value(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            )
            marginals = orihime.sw(
                scores,
                gap_score=-1.0,
                temperature=1.0,
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32

    def test_disabled_autocast(self):
        scores = torch.randn(2, 8, 10, device="cuda")
        value = orihime.sw_value(
            scores,
            gap_score=-1.0,
            temperature=1.0,
        )
        marginals = orihime.sw(
            scores,
            gap_score=-1.0,
            temperature=1.0,
        )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutocastLowLevel:
    def test_sw_forward_autocast(self):
        scores = torch.randn(
            2, 8, 10, device="cuda", dtype=torch.float16
        )

        with torch.autocast("cuda", dtype=torch.float16):
            value, marginals = sw_ops.forward(
                scores, -1.0, 1.0, None
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32

    def test_sw_affine_forward_autocast(self):
        scores = torch.randn(
            2, 8, 10, device="cuda", dtype=torch.float16
        )

        with torch.autocast("cuda", dtype=torch.float16):
            value, marginals = sw_affine_ops.forward(
                scores, -2.0, -0.5, 1.0, None
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32


# ---------------------------------------------------------------------------
# Autocast map-backward-after-exit across all fourteen top-level operators, plus
# MAS contracted backward. Self-contained cases mirror tests/test_compile.py
# and exercise map autograd backward across the autocast boundary: FP16 inputs
# are promoted to FP32 inside autocast, gradients return to the FP16 input
# dtype after exit, and MAS runs its explicit contracted backward in autocast.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from operator_cases import AUTOCAST_DTYPE_IDS, OPERATOR_CASES


@dataclass(frozen=True)
class _AutocastCase:
    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    params: tuple[tuple[str, float], ...]


_AUTOCAST_CASES = tuple(
    _AutocastCase(spec.name, spec.shapes("autocast"), spec.matrix_params)
    for spec in OPERATOR_CASES
)

def _autocast_tensor_args(case, device):
    torch.manual_seed(1234)
    if case.name == "cky":
        return tuple(
            torch.randn(shape, device=device)
            for shape in case.input_shapes
        )
    if case.name == "osa":
        scores = torch.rand(case.input_shapes[0], device=device)
        trans_mask = torch.zeros_like(scores, dtype=torch.bool)
        trans_mask[:, 1:, 1:] = True
        return scores, trans_mask
    if case.name == "damerau":
        scores = torch.rand(case.input_shapes[0], device=device)
        source_tokens = torch.zeros(
            (case.input_shapes[0][0], case.input_shapes[0][1]),
            dtype=torch.int64,
            device=device,
        )
        target_tokens = torch.zeros_like(source_tokens)
        trans_src = orihime.build_damerau_transposition_sources(
            source_tokens,
            target_tokens,
        )
        assert torch.any(trans_src != -1)
        return scores, trans_src
    if case.name in {"dtw", "lev"}:
        return (torch.rand(case.input_shapes[0], device=device),)
    return (torch.randn(case.input_shapes[0], device=device),)


def _autocast_differentiable_indices(case):
    # index 0 is always the primary differentiable score tensor; CKY's second
    # tensor input (leaf scores) is also differentiable. OSA/Damerau's second
    # tensor argument is structural config and is never differentiated.
    return (0, 1) if case.name == "cky" else (0,)


def _autocast_call_map(case, args):
    op = getattr(orihime, case.name)
    params = dict(case.params)
    if case.name == "osa":
        return op(
            args[0],
            **params,
            allowed_transpositions=args[1],
        )
    if case.name == "damerau":
        return op(
            args[0],
            **params,
            transposition_sources=args[1],
        )
    return op(*args, **params)


def _autocast_call_value(case, args, param_values):
    op = getattr(orihime, f"{case.name}_value")
    params = {
        name: value
        for (name, _), value in zip(
            case.params,
            param_values,
            strict=True,
        )
    }
    if case.name == "osa":
        return op(
            args[0],
            **params,
            allowed_transpositions=args[1],
        )
    if case.name == "damerau":
        return op(
            args[0],
            **params,
            transposition_sources=args[1],
        )
    return op(*args, **params)


def _autocast_grad_args(source, differentiable, dtype):
    args = []
    for index, tensor in enumerate(source):
        if tensor.is_floating_point():
            tensor = tensor.detach().to(dtype=dtype).clone()
            tensor.requires_grad_(index in differentiable)
        else:
            tensor = tensor.detach().clone()
        args.append(tensor)
    return tuple(args)


def _autocast_params(case, device, dtype, *, requires_grad):
    return tuple(
        torch.tensor(
            value,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for _, value in case.params
    )


def _assert_autocast_grad_matches(actual, expected):
    assert actual.grad is not None
    assert expected.grad is not None
    assert torch.isfinite(actual.grad).all()
    assert actual.grad.dtype == actual.dtype
    torch.testing.assert_close(
        actual.grad,
        expected.grad.to(dtype=actual.dtype),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _AUTOCAST_CASES, ids=lambda case: case.name)
def test_autocast_map_backward_after_exit(case):
    """Map produced under CUDA autocast must back-propagate after the context
    exits: FP16 inputs promote to a finite FP32 map inside autocast, and the
    map's autograd backward (run outside autocast) yields finite gradients in
    the original FP16 input dtype for every differentiated tensor input."""
    device = torch.device("cuda")
    raw = _autocast_tensor_args(case, device)
    diff = _autocast_differentiable_indices(case)
    args = []
    for index, tensor in enumerate(raw):
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float16)
            if index in diff:
                tensor = tensor.detach().clone().requires_grad_(True)
        args.append(tensor)
    args = tuple(args)

    with torch.autocast("cuda", dtype=torch.float16):
        marginals = _autocast_call_map(case, args)
    assert isinstance(marginals, torch.Tensor)
    assert marginals.dtype == torch.float32
    assert torch.isfinite(marginals).all()

    # backward AFTER leaving the autocast context
    marginals.square().sum().backward()
    for index in diff:
        grad = args[index].grad
        assert grad is not None, f"missing grad for input {index} of {case.name}"
        assert torch.isfinite(grad).all(), f"non-finite grad for {case.name}"
        assert grad.dtype == torch.float16, (
            f"expected FP16 grad for {case.name}, got {grad.dtype}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16),
    ids=AUTOCAST_DTYPE_IDS,
)
@pytest.mark.parametrize("case", _AUTOCAST_CASES, ids=lambda case: case.name)
def test_autocast_value_backward_after_exit(case, dtype):
    """Value reverse mode recomputes in FP32 after the autocast context exits.

    Every tensor parameter and differentiable tensor input, including CKY leaf
    scores, receives the FP32 oracle gradient rounded only once to its original
    autocast dtype.
    """

    device = torch.device("cuda")
    differentiable = _autocast_differentiable_indices(case)
    low_precision_source = tuple(
        tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
        for tensor in _autocast_tensor_args(case, device)
    )
    low_precision_params = _autocast_params(
        case,
        device,
        dtype,
        requires_grad=False,
    )

    expected_args = _autocast_grad_args(
        low_precision_source,
        differentiable,
        torch.float32,
    )
    expected_params = tuple(
        param.float().detach().clone().requires_grad_(True)
        for param in low_precision_params
    )
    expected = _autocast_call_value(
        case,
        expected_args,
        expected_params,
    )
    expected.sum().backward()

    actual_args = _autocast_grad_args(
        low_precision_source,
        differentiable,
        dtype,
    )
    actual_params = tuple(
        param.detach().clone().requires_grad_(True)
        for param in low_precision_params
    )
    with torch.autocast("cuda", dtype=dtype):
        actual = _autocast_call_value(
            case,
            actual_args,
            actual_params,
        )

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    # Backward intentionally runs after leaving autocast.
    actual.sum().backward()
    for index in differentiable:
        _assert_autocast_grad_matches(
            actual_args[index],
            expected_args[index],
        )
    for actual_param, expected_param in zip(
        actual_params,
        expected_params,
        strict=True,
    ):
        _assert_autocast_grad_matches(
            actual_param,
            expected_param,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _AUTOCAST_CASES, ids=lambda case: case.name)
def test_autocast_bf16_value_vjp_pullback_after_exit(case):
    """A BF16 ``torch.func.vjp`` pullback remains valid after autocast exits."""

    device = torch.device("cuda")
    dtype = torch.bfloat16
    differentiable = _autocast_differentiable_indices(case)
    low_precision_args = tuple(
        tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
        for tensor in _autocast_tensor_args(case, device)
    )
    low_precision_params = _autocast_params(
        case,
        device,
        dtype,
        requires_grad=False,
    )

    def bind_value(base_args):
        def value(*primals):
            tensor_args = list(base_args)
            input_count = len(differentiable)
            for index, primal in zip(
                differentiable,
                primals[:input_count],
                strict=True,
            ):
                tensor_args[index] = primal
            return _autocast_call_value(
                case,
                tuple(tensor_args),
                primals[input_count:],
            )

        return value

    actual_primals = (
        *(low_precision_args[index] for index in differentiable),
        *low_precision_params,
    )
    expected_base_args = tuple(
        tensor.float() if tensor.is_floating_point() else tensor
        for tensor in low_precision_args
    )
    expected_primals = tuple(
        primal.float() for primal in actual_primals
    )
    expected, expected_pullback = torch.func.vjp(
        bind_value(expected_base_args),
        *expected_primals,
    )
    expected_grads = expected_pullback(torch.ones_like(expected))

    with torch.autocast("cuda", dtype=dtype):
        actual, actual_pullback = torch.func.vjp(
            bind_value(low_precision_args),
            *actual_primals,
        )

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    # The pullback intentionally runs after leaving autocast.
    actual_grads = actual_pullback(torch.ones_like(actual))
    for actual_grad, expected_grad, primal in zip(
        actual_grads,
        expected_grads,
        actual_primals,
        strict=True,
    ):
        assert actual_grad.dtype == primal.dtype
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad,
            expected_grad.to(dtype=primal.dtype),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mas_contracted_backward_in_autocast():
    """MAS explicit contracted backward must run inside autocast: the promoted
    FP32 cotangent must not fault against the original FP16 scores, and every
    returned gradient field must be finite."""
    device = torch.device("cuda")
    scores = torch.randn(1, 5, 3, device=device, dtype=torch.float16)
    temp = 0.9
    with torch.autocast("cuda", dtype=torch.float16):
        marginals = orihime.mas(scores, temperature=temp)
        grads = orihime.raw.mas.vjp(
            scores,
            temperature=temp,
            cotangent=torch.ones_like(marginals),
            wrt=orihime.raw.mas.vjp_fields,
        )
    assert isinstance(grads, dict) and grads
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"non-finite MAS backward field {name!r}"
