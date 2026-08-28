# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the finite 0.1 named-operation API."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import Tensor

import orihime as ohm
from operator_cases import OPERATOR_CASES


def _public_name(name: str) -> str:
    return "sv" if name == "sv_linear" else name


def _inputs(case) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(9107 + OPERATOR_CASES.index(case))
    values = []
    for shape in case.shapes("contract"):
        value = (
            torch.rand(shape, generator=generator)
            if case.nonnegative
            else 0.2 * torch.randn(shape, generator=generator)
        )
        values.append(value.contiguous())
    return tuple(values)


def test_public_export_surface_uses_semantic_operation_names() -> None:
    expected = tuple(_public_name(case.name) for case in OPERATOR_CASES)

    assert tuple(ohm.ops.__all__) == expected
    assert "raw" not in ohm.__all__
    assert "sv" in ohm.__all__
    assert "sv_linear" not in ohm.__all__
    assert not hasattr(ohm.ops, "sv_linear")


@pytest.mark.parametrize("case", OPERATOR_CASES, ids=lambda case: case.name)
def test_forward_selection_matches_top_level_functions(case) -> None:
    name = _public_name(case.name)
    operation = getattr(ohm.ops, name)
    inputs = _inputs(case)
    outputs = operation.forward(*inputs)

    assert tuple(outputs) == ("map", "value", "entropy")
    for field, suffix in (("map", ""), ("value", "_value"), ("entropy", "_entropy")):
        expected = getattr(ohm, f"{name}{suffix}")(*inputs)
        torch.testing.assert_close(outputs[field], expected)
        selected = operation.forward(*inputs, output=field)
        assert isinstance(selected, Tensor)
        torch.testing.assert_close(selected, expected)

    subset = operation.forward(*inputs, output=("entropy", "map"))
    assert tuple(subset) == ("entropy", "map")


@pytest.mark.parametrize("case", OPERATOR_CASES, ids=lambda case: case.name)
def test_backward_and_sensitivity_have_finite_selected_fields(case) -> None:
    operation = getattr(ohm.ops, _public_name(case.name))
    inputs = _inputs(case)
    map_result = operation.forward(*inputs, output="map")
    grad_map = torch.linspace(
        -0.75,
        0.5,
        map_result.numel(),
        dtype=map_result.dtype,
        device=map_result.device,
    ).reshape_as(map_result).contiguous()

    backward = operation.backward(*inputs, grad_map=grad_map)
    sensitivity = operation.sensitivity(*inputs)

    assert tuple(backward) == operation.backward_fields
    assert tuple(sensitivity) == operation.sensitivity_fields
    assert all(torch.isfinite(value).all() for value in backward.values())
    assert all(torch.isfinite(value).all() for value in sensitivity.values())
    for input_name, input_value in zip(
        operation.input_fields,
        inputs,
        strict=True,
    ):
        assert backward[input_name].shape == input_value.shape
    for field in operation.parameter_fields:
        assert backward[field].shape == torch.Size([])
        torch.testing.assert_close(
            backward[field],
            (sensitivity[field] * grad_map).sum(),
            rtol=2e-4,
            atol=2e-5,
        )
        selected = operation.backward(
            *inputs,
            grad_map=grad_map,
            output=field,
        )
        assert isinstance(selected, Tensor)
        torch.testing.assert_close(selected, backward[field])


def test_output_selection_is_explicit_and_validated() -> None:
    scores = torch.randn(1, 3, 3)
    grad_map = torch.ones_like(scores)

    with pytest.raises(ValueError, match="unknown output field"):
        ohm.ops.sw.forward(scores, output="scores")
    with pytest.raises(ValueError, match="at least one"):
        ohm.ops.sw.sensitivity(scores, output=())
    with pytest.raises(ValueError, match="duplicate"):
        ohm.ops.sw.backward(
            scores,
            grad_map=grad_map,
            output=("temperature", "temperature"),
        )
    with pytest.raises(TypeError, match="output must be"):
        ohm.ops.sw.forward(scores, output=object())


def test_operation_signatures_preserve_algorithm_arguments() -> None:
    forward = inspect.signature(ohm.ops.osa.forward)
    backward = inspect.signature(ohm.ops.osa.backward)
    sensitivity = inspect.signature(ohm.ops.osa.sensitivity)

    expected_prefix = (
        "substitution_costs",
        "insertion_cost",
        "deletion_cost",
        "transposition_cost",
        "temperature",
        "lengths",
        "allowed_transpositions",
        "mask",
        "dtype",
    )
    assert tuple(forward.parameters) == (*expected_prefix, "output")
    assert tuple(backward.parameters) == (*expected_prefix, "grad_map", "output")
    assert tuple(sensitivity.parameters) == (*expected_prefix, "output")
    assert backward.parameters["grad_map"].default is inspect.Parameter.empty
