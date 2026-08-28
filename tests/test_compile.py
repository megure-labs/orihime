# SPDX-License-Identifier: Apache-2.0
"""torch.compile coverage for the fourteen-operator top-level surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch
from torch import Tensor

import orihime
from operator_cases import OPERATOR_CASES


@dataclass(frozen=True)
class CompileCase:
    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    params: tuple[tuple[str, float], ...]


CASES = tuple(
    CompileCase(spec.name, spec.shapes("compile"), spec.matrix_params)
    for spec in OPERATOR_CASES
)

OBSERVABLES = ("map", "value", "entropy")
MAS_LENGTH_BACKENDS = ("aot_eager", "inductor")


def _tensor_args(case: CompileCase, device: torch.device) -> tuple[Tensor, ...]:
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


def _bind_call(
    case: CompileCase,
    observable: str,
) -> Callable[..., Tensor]:
    target_name = (
        case.name
        if observable == "map"
        else f"{case.name}_{observable}"
    )
    target = getattr(orihime, target_name)
    params = dict(case.params)

    if case.name == "osa":

        def call(*args: Tensor, **kwargs):
            return target(
                args[0],
                **params,
                allowed_transpositions=args[1],
                **kwargs,
            )

        return call

    if case.name == "damerau":

        def call(*args: Tensor, **kwargs):
            return target(
                args[0],
                **params,
                transposition_sources=args[1],
                **kwargs,
            )

        return call

    def call(*args: Tensor, **kwargs):
        return target(*args, **params, **kwargs)

    return call


def _tensor_parameters(
    case: CompileCase,
    device: torch.device,
) -> tuple[Tensor, ...]:
    return tuple(
        torch.tensor(value, device=device)
        for _, value in case.params
    )


def _bind_tensor_parameter_call(
    case: CompileCase,
    observable: str,
) -> Callable[..., Tensor]:
    target_name = (
        case.name
        if observable == "map"
        else f"{case.name}_{observable}"
    )
    target = getattr(orihime, target_name)
    param_names = tuple(name for name, _ in case.params)
    param_count = len(param_names)

    def call(*args: Tensor) -> Tensor:
        tensor_args = args[:-param_count]
        param_values = args[-param_count:]
        params = dict(
            zip(param_names, param_values, strict=True)
        )
        if case.name == "osa":
            return target(
                tensor_args[0],
                allowed_transpositions=tensor_args[1],
                **params,
            )
        if case.name == "damerau":
            return target(
                tensor_args[0],
                transposition_sources=tensor_args[1],
                **params,
            )
        return target(*tensor_args, **params)

    return call


def _assert_finite(*values: Tensor) -> None:
    for value in values:
        assert torch.isfinite(value).all(), value


def _assert_tensor_close(
    actual: Tensor,
    expected: Tensor,
) -> None:
    _assert_finite(actual, expected)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
        equal_nan=False,
    )


def _assert_close(
    actual: tuple[Tensor, ...],
    expected: tuple[Tensor, ...],
) -> None:
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        _assert_tensor_close(
            actual_tensor,
            expected_tensor,
        )


def _compile_fullgraph(function, backend: str):
    if backend == "inductor":
        # Omit ``backend=`` to exercise ordinary default Inductor.
        return torch.compile(function, fullgraph=True)
    return torch.compile(
        function,
        backend=backend,
        fullgraph=True,
    )


def _mas_length_inputs(
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(2468)
    scores = torch.randn(2, 5, 3, device=device)
    valid_ragged = torch.tensor(
        [[5, 3], [4, 2]],
        dtype=torch.int32,
        device=device,
    )
    changed_valid_ragged = torch.tensor(
        [[4, 3], [3, 2]],
        dtype=torch.int32,
        device=device,
    )
    invalid_ragged = torch.tensor(
        [[2, 3], [3, 2]],
        dtype=torch.int32,
        device=device,
    )
    return (
        scores,
        valid_ragged,
        changed_valid_ragged,
        invalid_ragged,
    )


def _mas_observable(
    observable: str,
    scores: Tensor,
    lengths: Tensor,
) -> Tensor:
    target_name = (
        "mas" if observable == "map" else f"mas_{observable}"
    )
    target = getattr(orihime, target_name)
    return target(
        scores,
        temperature=0.9,
        lengths=lengths,
    )


def _grad_args(
    case: CompileCase,
    observable: str,
    source_args: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    result = []
    for index, tensor in enumerate(source_args):
        differentiable = index == 0 or (
            case.name == "cky"
            and observable in {"map", "value"}
        )
        result.append(
            tensor.detach()
            .clone()
            .requires_grad_(
                differentiable and tensor.is_floating_point()
            )
        )
    return tuple(result)


@pytest.mark.parametrize("fullgraph", (True, False))
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_compile_observables(
    case: CompileCase,
    fullgraph: bool,
) -> None:
    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _tensor_args(case, device)
    call_map = _bind_call(case, "map")
    call_value = _bind_call(case, "value")
    call_entropy = _bind_call(case, "entropy")

    def observables(*tensor_args: Tensor) -> tuple[Tensor, ...]:
        marginals = call_map(*tensor_args)
        assert isinstance(marginals, Tensor)
        return (
            marginals,
            call_value(*tensor_args),
            call_entropy(*tensor_args),
        )

    expected = observables(*args)
    compiled = torch.compile(
        observables,
        backend="aot_eager",
        fullgraph=fullgraph,
    )
    actual = compiled(*args)
    _assert_close(actual, expected)


@pytest.mark.parametrize("fullgraph", (True, False))
@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_compile_reverse_mode(
    case: CompileCase,
    observable: str,
    fullgraph: bool,
) -> None:
    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_args = _tensor_args(case, device)
    call = _bind_call(case, observable)

    def loss(*tensor_args: Tensor) -> Tensor:
        output = call(*tensor_args)
        assert isinstance(output, Tensor)
        return output.square().sum()

    expected_args = _grad_args(case, observable, source_args)
    expected_loss = loss(*expected_args)
    expected_loss.backward()

    actual_args = _grad_args(case, observable, source_args)
    compiled = torch.compile(
        loss,
        backend="aot_eager",
        fullgraph=fullgraph,
    )
    actual_loss = compiled(*actual_args)
    actual_loss.backward()

    _assert_tensor_close(
        actual_loss,
        expected_loss,
    )
    for actual_arg, expected_arg in zip(
        actual_args, expected_args, strict=True
    ):
        if not expected_arg.requires_grad:
            continue
        assert actual_arg.grad is not None
        assert expected_arg.grad is not None
        _assert_tensor_close(
            actual_arg.grad,
            expected_arg.grad,
        )


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_default_inductor_observable(
    case: CompileCase,
    observable: str,
) -> None:
    """Default Inductor keeps every observable fullgraph and repeatable."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _tensor_args(case, device)
    call = _bind_call(case, observable)
    expected = call(*args)
    assert isinstance(expected, Tensor)

    # Intentionally omit ``backend=``: this is the ordinary default-Inductor
    # API promised by the frozen v3 support matrix.
    compiled = torch.compile(call, fullgraph=True)
    actual = compiled(*args)
    repeated = compiled(*args)
    for result in (actual, repeated):
        assert isinstance(result, Tensor)
        _assert_tensor_close(result, expected)


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_default_inductor_reverse_mode(
    case: CompileCase,
    observable: str,
) -> None:
    """Primary reverse mode is numerical for all 36 compiled observables."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_args = _tensor_args(case, device)
    call = _bind_call(case, observable)

    def loss(*tensor_args: Tensor) -> Tensor:
        output = call(*tensor_args)
        assert isinstance(output, Tensor)
        return output.square().sum()

    expected_args = _grad_args(case, observable, source_args)
    expected_loss = loss(*expected_args)
    expected_loss.backward()

    actual_args = _grad_args(case, observable, source_args)
    # Intentionally no backend override: exercise default Inductor.
    compiled = torch.compile(loss, fullgraph=True)
    actual_loss = compiled(*actual_args)
    actual_loss.backward()

    _assert_tensor_close(
        actual_loss,
        expected_loss,
    )
    for actual_arg, expected_arg in zip(
        actual_args, expected_args, strict=True
    ):
        if not expected_arg.requires_grad:
            continue
        assert actual_arg.grad is not None
        assert expected_arg.grad is not None
        _assert_tensor_close(
            actual_arg.grad,
            expected_arg.grad,
        )


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_default_inductor_tensor_parameters(
    case: CompileCase,
    observable: str,
) -> None:
    """All tensor parameters stay dynamic across a fullgraph compile."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_args = _tensor_args(case, device)
    params = _tensor_parameters(case, device)
    call = _bind_tensor_parameter_call(case, observable)

    expected = call(*tensor_args, *params)
    assert isinstance(expected, Tensor)
    compiled = torch.compile(call, fullgraph=True)
    actual = compiled(*tensor_args, *params)
    _assert_tensor_close(actual, expected)

    for index, (param_name, _) in enumerate(case.params):
        changed_params = list(params)
        changed_params[index] = changed_params[index] + 0.2
        changed_expected = call(*tensor_args, *changed_params)
        assert not torch.allclose(
            changed_expected,
            expected,
            rtol=1e-4,
            atol=1e-5,
        ), (
            f"{case.name} {observable} parameter {param_name} is inert in "
            "the compile fixture"
        )
        changed_actual = compiled(*tensor_args, *changed_params)
        _assert_tensor_close(changed_actual, changed_expected)


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_default_inductor_tensor_parameter_reverse_mode(
    case: CompileCase,
    observable: str,
) -> None:
    """Dynamic tensor parameters retain every supported reverse direction."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_args = _tensor_args(case, device)
    source_params = _tensor_parameters(case, device)
    call = _bind_tensor_parameter_call(case, observable)

    def loss(*args: Tensor) -> Tensor:
        output = call(*args)
        return output.square().sum()

    def grad_args() -> tuple[Tensor, ...]:
        tensor_args = _grad_args(
            case,
            observable,
            source_args,
        )
        params = tuple(
            param.detach()
            .clone()
            .requires_grad_(observable != "entropy")
            for param in source_params
        )
        return (*tensor_args, *params)

    expected_args = grad_args()
    expected_loss = loss(*expected_args)
    expected_loss.backward()
    if observable == "entropy":
        assert expected_args[0].grad is not None
        assert expected_args[0].grad.abs().max() > 1e-6

    actual_args = grad_args()
    # Intentionally no backend override: exercise default Inductor.
    compiled = torch.compile(loss, fullgraph=True)
    actual_loss = compiled(*actual_args)
    actual_loss.backward()

    _assert_tensor_close(
        actual_loss,
        expected_loss,
    )
    for actual_arg, expected_arg in zip(
        actual_args, expected_args, strict=True
    ):
        if not expected_arg.requires_grad:
            continue
        assert actual_arg.grad is not None
        assert expected_arg.grad is not None
        _assert_tensor_close(
            actual_arg.grad,
            expected_arg.grad,
        )


def test_default_inductor_nn_module() -> None:
    """A stateful module keeps tensor parameters inside the full graph."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = orihime.nn.SmithWaterman(
        gap_score=-0.7,
        temperature=0.9,
        learnable=("gap_score", "temperature"),
        device=device,
    )
    scores = torch.randn(1, 4, 4, device=device)
    expected = module(scores)

    compiled = torch.compile(module, fullgraph=True)
    actual = compiled(scores)
    _assert_tensor_close(actual, expected)

    with torch.no_grad():
        module.gap_score.add_(0.2)
        module.temperature.add_(0.2)
    changed_expected = module(scores)
    changed_actual = compiled(scores)
    _assert_tensor_close(changed_actual, changed_expected)


@pytest.mark.parametrize("backend", MAS_LENGTH_BACKENDS)
def test_mas_lengths_fullgraph_observables(backend: str) -> None:
    """Dynamic MAS lengths keep all three observables in one full graph."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores, valid, changed_valid, invalid = _mas_length_inputs(
        device
    )

    def observables(
        input_scores: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, ...]:
        return tuple(
            _mas_observable(
                observable,
                input_scores,
                lengths,
            )
            for observable in OBSERVABLES
        )

    expected = observables(scores, valid)
    changed_expected = observables(scores, changed_valid)
    assert any(
        not torch.allclose(
            first,
            second,
            rtol=1e-4,
            atol=1e-5,
        )
        for first, second in zip(
            expected,
            changed_expected,
            strict=True,
        )
    )

    compiled = _compile_fullgraph(observables, backend)
    _assert_close(compiled(scores, valid), expected)
    _assert_close(
        compiled(scores, changed_valid),
        changed_expected,
    )

    with pytest.raises(
        RuntimeError,
        match=r"mas requires lengths\[:, 0\] >= lengths\[:, 1\]",
    ):
        compiled(scores, invalid)


@pytest.mark.parametrize("backend", MAS_LENGTH_BACKENDS)
@pytest.mark.parametrize("observable", OBSERVABLES)
def test_mas_lengths_fullgraph_reverse_mode(
    backend: str,
    observable: str,
) -> None:
    """Dynamic valid and invalid MAS lengths remain graph-safe in reverse mode."""

    torch._dynamo.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores, valid, changed_valid, invalid = _mas_length_inputs(
        device
    )

    def loss(input_scores: Tensor, lengths: Tensor) -> Tensor:
        output = _mas_observable(
            observable,
            input_scores,
            lengths,
        )
        return output.square().sum()

    compiled = _compile_fullgraph(loss, backend)
    expected_losses = []
    for current_lengths in (valid, changed_valid):
        expected_scores = (
            scores.detach().clone().requires_grad_(True)
        )
        expected_loss = loss(
            expected_scores,
            current_lengths,
        )
        (expected_grad,) = torch.autograd.grad(
            expected_loss,
            (expected_scores,),
        )
        expected_losses.append(expected_loss)

        actual_scores = (
            scores.detach().clone().requires_grad_(True)
        )
        actual_loss = compiled(
            actual_scores,
            current_lengths,
        )
        (actual_grad,) = torch.autograd.grad(
            actual_loss,
            (actual_scores,),
        )
        _assert_tensor_close(actual_loss, expected_loss)
        _assert_tensor_close(actual_grad, expected_grad)

    assert not torch.allclose(
        expected_losses[0],
        expected_losses[1],
        rtol=1e-4,
        atol=1e-5,
    )

    with pytest.raises(
        RuntimeError,
        match=r"mas requires lengths\[:, 0\] >= lengths\[:, 1\]",
    ):
        compiled(scores, invalid)
