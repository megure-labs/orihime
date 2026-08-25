# SPDX-License-Identifier: Apache-2.0
"""Numerical torch.func coverage for every frozen v3 observable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch
from torch import Tensor

import d2p
from operator_cases import OPERATOR_CASES


@dataclass(frozen=True)
class FuncCase:
    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    params: tuple[tuple[str, float], ...]


CASES = tuple(
    FuncCase(spec.name, spec.shapes("func"), spec.matrix_params)
    for spec in OPERATOR_CASES
)

OBSERVABLES = ("map", "value", "entropy")
TRANSFORMS = ("jvp", "vjp", "jacrev", "vmap")

PARAMETER_VMAP_ARGUMENT_NAMES = {
    spec.name: spec.raw_vmap_names for spec in OPERATOR_CASES
}

PARAMETER_VMAP_DIRECTIONS = tuple(
    (case, public_name, raw_name, value)
    for case in CASES
    for (public_name, value), raw_name in zip(
        case.params,
        PARAMETER_VMAP_ARGUMENT_NAMES[case.name],
        strict=True,
    )
)

ENTROPY_UNSUPPORTED_DIRECTIONS = tuple(
    (case, parameter_name)
    for case in CASES
    for parameter_name, _ in case.params
) + (
    (
        next(case for case in CASES if case.name == "cky"),
        "leaf_scores",
    ),
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inputs(
    case: FuncCase,
    device: torch.device,
    *,
    batch_size: int = 2,
) -> tuple[Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(
        5100 + CASES.index(case)
    )
    shapes = tuple(
        (batch_size, *shape[1:]) for shape in case.input_shapes
    )
    tensors = []
    for shape in shapes:
        if case.name in {"dtw", "lev", "osa", "damerau"}:
            tensor = torch.rand(
                shape,
                generator=generator,
                device=device,
            )
        else:
            tensor = 0.2 * torch.randn(
                shape,
                generator=generator,
                device=device,
            )
        tensors.append(tensor)
    return tuple(tensors)


def _structural_kwargs(
    case: FuncCase,
    primary: Tensor,
) -> dict[str, Tensor]:
    if case.name == "osa":
        mask = torch.zeros_like(primary, dtype=torch.bool)
        mask[:, 1:, 1:] = True
        return {"allowed_transpositions": mask}
    if case.name == "damerau":
        sources = torch.full(
            (*primary.shape, 2),
            -1,
            dtype=torch.int32,
            device=primary.device,
        )
        return {"transposition_sources": sources}
    return {}


def _target(case: FuncCase, observable: str) -> Callable[..., Tensor]:
    suffix = "" if observable == "map" else f"_{observable}"
    return getattr(d2p, f"{case.name}{suffix}")


def _function_and_primals(
    case: FuncCase,
    observable: str,
    device: torch.device,
) -> tuple[Callable[..., Tensor], tuple[Tensor, ...]]:
    inputs = _inputs(case, device)
    structural = _structural_kwargs(case, inputs[0])
    target = _target(case, observable)

    # Entropy exposes only its primary-score derivative in 0.1.0. Scalar
    # parameter directions, and CKY's leaf direction, are explicitly outside
    # the shipped derivative coverage in docs/DERIVATIVE_COVERAGE.md.
    if observable == "entropy":
        frozen_params = dict(case.params)
        other_inputs = inputs[1:]

        def entropy(primary: Tensor) -> Tensor:
            return target(
                primary,
                *other_inputs,
                **frozen_params,
                **structural,
            )

        return entropy, (inputs[0],)

    input_count = len(inputs)
    tensor_params = tuple(
        torch.tensor(value, dtype=torch.float32, device=device)
        for _, value in case.params
    )
    primals = (*inputs, *tensor_params)

    def function(*values: Tensor) -> Tensor:
        dynamic_inputs = values[:input_count]
        dynamic_params = values[input_count:]
        params = {
            name: value
            for (name, _), value in zip(
                case.params, dynamic_params, strict=True
            )
        }
        return target(
            *dynamic_inputs,
            **params,
            **structural,
        )

    return function, primals


def _tangents(primals: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    tangents = []
    for index, primal in enumerate(primals):
        tangent = torch.linspace(
            -0.2 + 0.03 * index,
            0.3 + 0.03 * index,
            primal.numel(),
            dtype=primal.dtype,
            device=primal.device,
        ).reshape_as(primal)
        tangents.append(tangent)
    return tuple(tangents)


def _cotangent(output: Tensor) -> Tensor:
    return torch.linspace(
        -0.3,
        0.4,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)


def _finite_difference_vjp(
    function: Callable[..., Tensor],
    primals: tuple[Tensor, ...],
    cotangent: Tensor,
) -> tuple[Tensor, ...]:
    """Differentiate a projected output without using autograd."""

    epsilon = 1e-2
    expected = []
    for primal_index, primal in enumerate(primals):
        gradient = torch.empty_like(primal)
        for element_index in range(primal.numel()):
            positive = [value.detach().clone() for value in primals]
            negative = [value.detach().clone() for value in primals]
            positive[primal_index].reshape(-1)[element_index] += epsilon
            negative[primal_index].reshape(-1)[element_index] -= epsilon
            positive_output = function(*positive)
            negative_output = function(*negative)
            _assert_finite(positive_output, negative_output)
            positive_projection = (positive_output * cotangent).sum()
            negative_projection = (negative_output * cotangent).sum()
            _assert_finite(positive_projection, negative_projection)
            gradient.reshape(-1)[element_index] = (
                positive_projection - negative_projection
            ) / (2.0 * epsilon)
        expected.append(gradient)
    return tuple(expected)


def _assert_finite(*values: Tensor) -> None:
    for value in values:
        assert torch.isfinite(value).all(), value


def _assert_close(
    actual: Tensor | tuple[Tensor, ...],
    expected: Tensor | tuple[Tensor, ...],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    assert len(actual_values) == len(expected_values)
    for actual_value, expected_value in zip(
        actual_values, expected_values, strict=True
    ):
        _assert_finite(actual_value, expected_value)
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )


def _run_jvp(
    function: Callable[..., Tensor],
    primals: tuple[Tensor, ...],
) -> None:
    tangents = _tangents(primals)
    _assert_finite(*primals, *tangents)
    actual_primal, actual_tangent = torch.func.jvp(
        function,
        primals,
        tangents,
    )
    eager_primal = function(*primals)
    _assert_close(actual_primal, eager_primal)
    epsilon = 1e-2
    positive = tuple(
        primal + epsilon * tangent
        for primal, tangent in zip(primals, tangents, strict=True)
    )
    negative = tuple(
        primal - epsilon * tangent
        for primal, tangent in zip(primals, tangents, strict=True)
    )
    expected = (function(*positive) - function(*negative)) / (
        2.0 * epsilon
    )
    _assert_finite(actual_tangent, expected)
    expected_scale = expected.abs().max()
    actual_scale = actual_tangent.abs().max()
    assert expected_scale > 1e-6, (
        "the deterministic JVP direction must have a nonzero numerical "
        "derivative"
    )
    assert actual_scale > 0.1 * expected_scale, (
        "a nonzero numerical JVP direction produced a zero or negligible "
        "transformed tangent"
    )
    _assert_close(
        actual_tangent,
        expected,
        rtol=5e-2,
        atol=1e-4,
    )


def _run_vjp(
    function: Callable[..., Tensor],
    primals: tuple[Tensor, ...],
) -> None:
    output, pullback = torch.func.vjp(function, *primals)
    eager_primals = tuple(
        primal.detach().clone().requires_grad_() for primal in primals
    )
    eager_output = function(*eager_primals)
    _assert_close(output, eager_output)
    cotangent = _cotangent(output)
    _assert_finite(cotangent)
    actual = pullback(cotangent)
    expected = torch.autograd.grad(
        eager_output,
        eager_primals,
        grad_outputs=cotangent,
    )
    finite_difference = _finite_difference_vjp(
        function,
        primals,
        cotangent,
    )
    _assert_close(actual, expected)
    _assert_close(
        expected,
        finite_difference,
        rtol=5e-2,
        atol=5e-3,
    )


def _run_jacrev(
    function: Callable[..., Tensor],
    primals: tuple[Tensor, ...],
) -> None:
    argnums = tuple(range(len(primals)))
    actual = torch.func.jacrev(function, argnums=argnums)(*primals)
    eager_primals = tuple(
        primal.detach().clone().requires_grad_() for primal in primals
    )
    expected = torch.autograd.functional.jacobian(
        function,
        eager_primals,
    )
    _assert_close(actual, expected)


def _run_eager_backward(
    function: Callable[..., Tensor],
    primals: tuple[Tensor, ...],
) -> None:
    eager_primals = tuple(
        primal.detach().clone().requires_grad_() for primal in primals
    )
    output = function(*eager_primals)
    cotangent = _cotangent(output)
    _assert_finite(output, cotangent)
    loss = (output * cotangent).sum()
    loss.backward()
    actual = tuple(primal.grad for primal in eager_primals)
    assert all(gradient is not None for gradient in actual)

    expected = _finite_difference_vjp(
        function,
        primals,
        cotangent,
    )
    _assert_close(actual, expected, rtol=5e-2, atol=5e-3)


def _run_vmap(
    case: FuncCase,
    observable: str,
    device: torch.device,
    *,
    native_batch_size: int = 2,
) -> None:
    inputs = _inputs(case, device, batch_size=native_batch_size)
    primary = inputs[0]
    structural = _structural_kwargs(case, primary)
    other_inputs = inputs[1:]
    params = dict(case.params)
    target = _target(case, observable)

    def function(mapped_primary: Tensor) -> Tensor:
        return target(
            mapped_primary,
            *other_inputs,
            **params,
            **structural,
        )

    generator = torch.Generator(device=device).manual_seed(
        7100
        + 10 * CASES.index(case)
        + OBSERVABLES.index(observable)
        + native_batch_size
    )
    perturbations = 0.05 * torch.randn(
        3,
        *primary.shape,
        generator=generator,
        dtype=primary.dtype,
        device=device,
    )
    assert not torch.equal(perturbations[0], perturbations[1])
    assert not torch.equal(perturbations[1], perturbations[2])
    batched_primary = primary.unsqueeze(0) + perturbations
    actual = torch.vmap(function)(batched_primary)
    expected = torch.stack(
        tuple(function(value) for value in batched_primary)
    )
    _assert_close(actual, expected)


def _primary_function(
    case: FuncCase,
    observable: str,
    device: torch.device,
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    inputs = _inputs(case, device)
    primary = inputs[0]
    other_inputs = inputs[1:]
    structural = _structural_kwargs(case, primary)
    target = _target(case, observable)
    params = dict(case.params)

    def function(dynamic_primary: Tensor) -> Tensor:
        return target(
            dynamic_primary,
            *other_inputs,
            **params,
            **structural,
        )

    return function, primary


def _run_transform(
    case: FuncCase,
    observable: str,
    transform: str,
    device: torch.device,
) -> None:
    if transform == "vmap":
        _run_vmap(case, observable, device)
        return
    function, primals = _function_and_primals(case, observable, device)
    if transform == "jvp":
        _run_jvp(function, primals)
    elif transform == "vjp":
        _run_vjp(function, primals)
    elif transform == "jacrev":
        _run_jacrev(function, primals)
    else:
        raise AssertionError(transform)


def _entropy_direction_function(
    case: FuncCase,
    direction_name: str,
    device: torch.device,
) -> tuple[Callable[[Tensor, Tensor], Tensor], Tensor, Tensor]:
    inputs = _inputs(case, device)
    primary = inputs[0]
    target = _target(case, "entropy")
    structural = _structural_kwargs(case, primary)
    frozen_params = dict(case.params)

    if direction_name == "leaf_scores":
        assert case.name == "cky"
        leaf_scores = inputs[1]

        def entropy_with_leaf(
            dynamic_primary: Tensor,
            dynamic_leaf: Tensor,
        ) -> Tensor:
            return target(
                dynamic_primary,
                dynamic_leaf,
                **frozen_params,
                **structural,
            )

        return entropy_with_leaf, primary, leaf_scores

    other_inputs = inputs[1:]
    direction = torch.tensor(
        frozen_params.pop(direction_name),
        dtype=primary.dtype,
        device=device,
    )

    def entropy_with_parameter(
        dynamic_primary: Tensor,
        dynamic_parameter: Tensor,
    ) -> Tensor:
        return target(
            dynamic_primary,
            *other_inputs,
            **frozen_params,
            **{direction_name: dynamic_parameter},
            **structural,
        )

    return entropy_with_parameter, primary, direction


def _assert_entropy_direction_error(
    error: NotImplementedError,
    case: FuncCase,
    direction_name: str,
) -> None:
    message = str(error)
    assert f"{case.name}_entropy" in message
    assert f"parameter {direction_name}" in message
    assert "DERIVATIVE_COVERAGE.md" in message
    assert "finite differences" in message


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_func_transform_matches_numerical_oracle(
    case: FuncCase,
    observable: str,
    transform: str,
) -> None:
    _run_transform(case, observable, transform, _device())


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_eager_backward_matches_finite_difference(
    case: FuncCase,
    observable: str,
) -> None:
    function, primals = _function_and_primals(
        case,
        observable,
        _device(),
    )
    _run_eager_backward(function, primals)


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_vmap_native_batch_one_matches_loop(
    case: FuncCase,
    observable: str,
) -> None:
    _run_vmap(
        case,
        observable,
        _device(),
        native_batch_size=1,
    )


@pytest.mark.parametrize("composition", ("jacfwd", "vmap_jvp"))
@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_unsupported_forward_mode_composition_raises(
    case: FuncCase,
    observable: str,
    composition: str,
) -> None:
    function, primary = _primary_function(
        case,
        observable,
        _device(),
    )
    with pytest.raises(NotImplementedError) as caught:
        if composition == "jacfwd":
            torch.func.jacfwd(function)(primary)
        else:
            batched_primary = torch.stack(
                (primary, primary + 0.05),
            )
            batched_tangent = torch.stack(
                (
                    torch.ones_like(primary),
                    torch.linspace(
                        -0.2,
                        0.3,
                        primary.numel(),
                        device=primary.device,
                        dtype=primary.dtype,
                    ).reshape_as(primary),
                )
            )

            def mapped_jvp(
                mapped_primary: Tensor,
                mapped_tangent: Tensor,
            ) -> Tensor:
                return torch.func.jvp(
                    function,
                    (mapped_primary,),
                    (mapped_tangent,),
                )[1]

            torch.vmap(mapped_jvp)(
                batched_primary,
                batched_tangent,
            )
    assert "jvp function" in str(caught.value)


@pytest.mark.parametrize("observable", OBSERVABLES)
@pytest.mark.parametrize(
    ("case", "parameter_name", "raw_name", "base_value"),
    PARAMETER_VMAP_DIRECTIONS,
    ids=lambda value: (
        value.name
        if isinstance(value, FuncCase)
        else value
        if isinstance(value, str)
        else None
    ),
)
def test_parameter_vmap_raises_clean_scalar_guard(
    case: FuncCase,
    parameter_name: str,
    raw_name: str,
    base_value: float,
    observable: str,
) -> None:
    inputs = _inputs(case, _device())
    primary = inputs[0]
    other_inputs = inputs[1:]
    structural = _structural_kwargs(case, primary)
    target = _target(case, observable)
    frozen_params = dict(case.params)
    frozen_params.pop(parameter_name)
    batched_parameter = torch.tensor(
        (base_value, base_value + 0.1),
        device=primary.device,
        dtype=primary.dtype,
    )

    def function(mapped_parameter: Tensor) -> Tensor:
        return target(
            primary,
            *other_inputs,
            **frozen_params,
            **{parameter_name: mapped_parameter},
            **structural,
        )

    with pytest.raises(RuntimeError) as caught:
        torch.vmap(function)(batched_parameter)
    message = str(caught.value)
    assert f"d2p::{case.name}_forward_t" in message
    assert f"mapping scalar argument {raw_name!r}" in message
    assert "keep scalar parameters unbatched" in message
    assert ".item()" not in message


@pytest.mark.parametrize(
    ("case", "direction_name"),
    ENTROPY_UNSUPPORTED_DIRECTIONS,
    ids=lambda value: value.name if isinstance(value, FuncCase) else value,
)
def test_entropy_unsupported_direction_jvp_raises(
    case: FuncCase,
    direction_name: str,
) -> None:
    function, primary, direction = _entropy_direction_function(
        case,
        direction_name,
        _device(),
    )
    with pytest.raises(NotImplementedError) as caught:
        torch.func.jvp(
            function,
            (primary, direction),
            (torch.zeros_like(primary), torch.ones_like(direction)),
        )
    _assert_entropy_direction_error(
        caught.value,
        case,
        direction_name,
    )


@pytest.mark.parametrize(
    ("case", "direction_name"),
    ENTROPY_UNSUPPORTED_DIRECTIONS,
    ids=lambda value: value.name if isinstance(value, FuncCase) else value,
)
def test_entropy_unsupported_direction_backward_raises(
    case: FuncCase,
    direction_name: str,
) -> None:
    function, primary, direction = _entropy_direction_function(
        case,
        direction_name,
        _device(),
    )
    differentiable_direction = (
        direction.detach().clone().requires_grad_()
    )
    with pytest.raises(NotImplementedError) as caught:
        function(primary, differentiable_direction).sum().backward()
    _assert_entropy_direction_error(
        caught.value,
        case,
        direction_name,
    )


def test_cky_leaf_map_is_non_differentiable() -> None:
    case = next(case for case in CASES if case.name == "cky")
    merge_scores, leaf_scores = _inputs(case, _device())
    leaf_map = d2p.cky_leaf_map(
        merge_scores.requires_grad_(),
        leaf_scores.requires_grad_(),
        temperature=torch.tensor(
            0.9,
            device=merge_scores.device,
            requires_grad=True,
        ),
    )
    _assert_finite(leaf_map)
    assert not leaf_map.requires_grad
    assert leaf_map.grad_fn is None
