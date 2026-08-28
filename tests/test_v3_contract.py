# SPDX-License-Identifier: Apache-2.0
"""Cross-operator tests for the frozen v3 public contract."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch
from torch import Tensor

import orihime
from operator_cases import OPERATOR_CASES


@dataclass(frozen=True)
class ContractCase:
    name: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    param_defaults: tuple[tuple[str, float], ...]
    structural_defaults: tuple[tuple[str, Any], ...]
    vjp_fields: tuple[str, ...]
    orientation: str
    nn_class: str
    operator_module: str
    operator_name: str
    nonnegative: bool = False

    @property
    def batch_size(self) -> int:
        return self.input_shapes[0][0]


CASES = tuple(
    ContractCase(
        name=spec.name,
        input_names=spec.input_names,
        input_shapes=spec.shapes("contract"),
        param_defaults=spec.contract_params,
        structural_defaults=spec.structural_defaults,
        vjp_fields=spec.vjp_fields,
        orientation=spec.orientation,
        nn_class=spec.nn_class,
        operator_module=spec.operator_module,
        operator_name=spec.operator_name,
        nonnegative=spec.nonnegative,
    )
    for spec in OPERATOR_CASES
)

CASE_BY_NAME = {case.name: case for case in CASES}
NN_CASES = tuple(
    (case, param_name)
    for case in CASES
    for param_name, _ in case.param_defaults
)


def _inputs(case: ContractCase) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(1701 + CASES.index(case))
    tensors = []
    for shape in case.input_shapes:
        if case.nonnegative:
            tensor = torch.rand(shape, generator=generator)
        else:
            tensor = 0.25 * torch.randn(shape, generator=generator)
        tensors.append(tensor.contiguous())
    return tuple(tensors)


def _map_function(case: ContractCase) -> Any:
    return getattr(orihime, case.name)


def _operator(case: ContractCase) -> Any:
    module = importlib.import_module(case.operator_module)
    return getattr(module, case.operator_name)


def _topology_kwargs(
    case: ContractCase,
    inputs: tuple[Tensor, ...],
) -> dict[str, Tensor]:
    if case.name == "osa":
        mask = torch.zeros_like(inputs[0], dtype=torch.bool)
        mask[:, 1:, 1:] = True
        return {"allowed_transpositions": mask}
    if case.name == "damerau":
        batch, length_1, length_2 = inputs[0].shape
        source_tokens = torch.zeros((batch, length_1), dtype=torch.int64)
        target_tokens = torch.zeros((batch, length_2), dtype=torch.int64)
        return {
            "transposition_sources": (
                orihime.build_damerau_transposition_sources(
                    source_tokens,
                    target_tokens,
                )
            )
        }
    return {}


DERIVATIVE_INVALID_CASES = ("type", "shape", "dtype", "device", "layout")


def _invalid_derivative_vector(vector: Tensor, kind: str) -> Any:
    if kind == "type":
        return object()
    if kind == "shape":
        return torch.empty(
            (*vector.shape[:-1], vector.shape[-1] + 1),
            dtype=vector.dtype,
            device=vector.device,
        )
    if kind == "dtype":
        return vector.to(dtype=torch.float64)
    if kind == "device":
        return vector.to(device="meta")
    if kind == "layout":
        storage = torch.empty(
            (*vector.shape, 2), dtype=vector.dtype, device=vector.device
        )
        result = storage[..., 0]
        assert result.shape == vector.shape
        assert not result.is_contiguous()
        return result
    raise AssertionError(f"unknown invalid derivative case: {kind}")


def _raw_named_derivative_call(
    case: ContractCase,
    inputs: tuple[Tensor, ...],
    primitive: str,
    vector: Any,
    *,
    slot: str = "primary",
) -> Any:
    """Call one named raw derivative primitive with a substituted vector."""

    raw = getattr(orihime.raw, case.name)
    params = tuple(value for _, value in case.param_defaults)
    primary = inputs[0]

    if case.name == "cky":
        merge_scores, leaf_scores = inputs
        merge_vector = vector if slot == "merge" else torch.randn_like(merge_scores)
        leaf_vector = vector if slot == "leaf" else torch.randn_like(leaf_scores)
        if primitive == "hvp":
            return raw.marginals_hvp(
                merge_scores,
                leaf_scores,
                merge_vector,
                leaf_vector,
                *params,
            )
        return raw.marginals_backward(
            merge_scores,
            leaf_scores,
            vector,
            *params,
        )

    if case.name == "osa":
        trans_mask = torch.zeros_like(primary)
        args = (primary, trans_mask, vector, *params, None)
    elif case.name == "damerau":
        trans_src = torch.full(
            (*primary.shape, 2), -1, dtype=torch.int32, device=primary.device
        )
        args = (primary, trans_src, vector, *params, None)
    elif case.name in {
        "sw",
        "sw_affine",
        "sv_linear",
        "sv_affine",
        "nw",
        "nw_affine",
    }:
        args = (primary, vector, *params, None)
    elif case.name == "dtw":
        args = (primary, vector, *params, None, None)
    elif case.name in {"lcs", "lev", "mas"}:
        args = (primary, vector, *params, None)
    elif case.name == "eisner":
        args = (primary, vector, *params, None)
    else:
        raise AssertionError(f"unhandled raw derivative case: {case.name}")

    return getattr(raw, f"marginals_{'hvp' if primitive == 'hvp' else 'backward'}")(
        *args
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("cardinality", ("one", "many"))
@pytest.mark.parametrize("invalid", DERIVATIVE_INVALID_CASES)
def test_raw_vjp_rejects_invalid_cotangent_contract(
    case: ContractCase,
    cardinality: str,
    invalid: str,
) -> None:
    inputs = _inputs(case)
    map_result = _map_function(case)(*inputs)
    cotangent = _invalid_derivative_vector(map_result, invalid)
    raw = getattr(orihime.raw, case.name)
    expected_exception = TypeError if invalid in {"type", "dtype"} else ValueError

    with pytest.raises(expected_exception, match=r"cotangent"):
        if cardinality == "one":
            raw.vjp_one(
                *inputs,
                wrt=raw.vjp_fields[0],
                cotangent=cotangent,
            )
        else:
            raw.vjp(
                *inputs,
                wrt=raw.vjp_fields,
                cotangent=cotangent,
            )


NAMED_DERIVATIVE_SLOTS = tuple(
    (case, "merge") if case.name == "cky" else (case, "primary")
    for case in CASES
)


@pytest.mark.parametrize(
    ("case", "slot"),
    NAMED_DERIVATIVE_SLOTS,
    ids=lambda value: value.name if isinstance(value, ContractCase) else value,
)
@pytest.mark.parametrize("primitive", ("hvp", "backward"))
@pytest.mark.parametrize("invalid", DERIVATIVE_INVALID_CASES)
def test_raw_named_derivatives_reject_invalid_vector_contract(
    case: ContractCase,
    slot: str,
    primitive: str,
    invalid: str,
) -> None:
    inputs = _inputs(case)
    if case.name == "cky" and slot == "leaf":
        valid = inputs[1]
    else:
        valid = inputs[0]
    vector = _invalid_derivative_vector(valid, invalid)

    with pytest.raises(RuntimeError, match=r"(?:tangent|cotangent)"):
        _raw_named_derivative_call(
            case,
            inputs,
            primitive,
            vector,
            slot=slot,
        )


@pytest.mark.parametrize("invalid", DERIVATIVE_INVALID_CASES)
def test_raw_cky_leaf_hvp_rejects_invalid_vector_contract(invalid: str) -> None:
    case = CASE_BY_NAME["cky"]
    inputs = _inputs(case)
    vector = _invalid_derivative_vector(inputs[1], invalid)

    with pytest.raises(RuntimeError, match=r"leaf tangent"):
        _raw_named_derivative_call(
            case,
            inputs,
            "hvp",
            vector,
            slot="leaf",
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_bare_functions_have_frozen_signatures_and_shapes(
    case: ContractCase,
) -> None:
    expected_names = (
        *case.input_names,
        *(name for name, _ in case.param_defaults),
        *(name for name, _ in case.structural_defaults),
        "mask",
        "dtype",
    )
    expected_defaults = {
        **dict(case.param_defaults),
        **dict(case.structural_defaults),
        "mask": None,
        "dtype": None,
    }

    for suffix in ("", "_value", "_entropy"):
        function = getattr(orihime, f"{case.name}{suffix}")
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == expected_names
        for input_name in case.input_names:
            parameter = signature.parameters[input_name]
            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert parameter.default is inspect.Parameter.empty
        for name, expected_default in expected_defaults.items():
            parameter = signature.parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default == expected_default

    inputs = _inputs(case)
    map_result = _map_function(case)(*inputs)
    value_result = getattr(orihime, f"{case.name}_value")(*inputs)
    entropy_result = getattr(orihime, f"{case.name}_entropy")(*inputs)

    assert isinstance(map_result, Tensor)
    assert map_result.shape == inputs[0].shape
    assert isinstance(value_result, Tensor)
    assert value_result.shape == (case.batch_size,)
    assert isinstance(entropy_result, Tensor)
    assert entropy_result.shape == (case.batch_size,)
    assert torch.isfinite(map_result).all()
    assert torch.isfinite(value_result).all()
    assert torch.isfinite(entropy_result).all()

    if case.name == "cky":
        leaf_map = orihime.cky_leaf_map(*inputs)
        assert isinstance(leaf_map, Tensor)
        assert leaf_map.shape == inputs[1].shape
        assert torch.isfinite(leaf_map).all()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_entropy_matches_signed_temperature_derivative_of_value(
    case: ContractCase,
) -> None:
    """Preserve the useful entropy identity from the superseded API suite."""

    inputs = _inputs(case)
    topology = _topology_kwargs(case, inputs)
    value_function = getattr(orihime, f"{case.name}_value")
    entropy_function = getattr(orihime, f"{case.name}_entropy")
    epsilon = 1.0e-2

    below = value_function(
        *inputs,
        temperature=1.0 - epsilon,
        **topology,
    )
    above = value_function(
        *inputs,
        temperature=1.0 + epsilon,
        **topology,
    )
    temperature_derivative = (above - below) / (2.0 * epsilon)
    orientation_sign = -1 if case.orientation == "cost-native" else 1

    torch.testing.assert_close(
        entropy_function(*inputs, temperature=1.0, **topology),
        orientation_sign * temperature_derivative,
        rtol=2.0e-2,
        atol=2.0e-3,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_map_autograd_and_func_grad_are_finite(case: ContractCase) -> None:
    inputs = tuple(value.detach().requires_grad_() for value in _inputs(case))
    map_result = _map_function(case)(*inputs)
    weights = torch.linspace(
        0.25,
        1.25,
        map_result.numel(),
        dtype=map_result.dtype,
        device=map_result.device,
    ).reshape_as(map_result)
    (map_result * weights).sum().backward()

    for value in inputs:
        assert value.grad is not None
        assert value.grad.shape == value.shape
        assert torch.isfinite(value.grad).all()

    primary, *other_inputs = (value.detach() for value in inputs)

    def scalarized_map(mapped_primary: Tensor) -> Tensor:
        result = _map_function(case)(mapped_primary, *other_inputs)
        return result.square().sum()

    transformed_grad = torch.func.grad(scalarized_map)(primary)
    assert isinstance(transformed_grad, Tensor)
    assert transformed_grad.shape == primary.shape
    assert torch.isfinite(transformed_grad).all()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_float_parameters_prune_map_parameter_kernels(
    case: ContractCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator(case)
    original_kernels = operator._kernels
    parameter_calls: list[str] = []

    def traced_param_backward(
        call: Any,
        state: Any,
        param_name: str,
        cotangent: Tensor | None,
    ) -> Tensor:
        parameter_calls.append(param_name)
        return original_kernels.param_backward(
            call,
            state,
            param_name,
            cotangent,
        )

    monkeypatch.setattr(
        operator,
        "_kernels",
        replace(
            original_kernels,
            param_backward=traced_param_backward,
        ),
    )
    inputs = tuple(value.requires_grad_() for value in _inputs(case))
    _map_function(case)(*inputs).square().sum().backward()

    assert parameter_calls == []
    assert all(value.grad is not None for value in inputs)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_raw_vjp_tier_is_mandatory_selected_and_type_stable(
    case: ContractCase,
) -> None:
    inputs = _inputs(case)
    map_result = _map_function(case)(*inputs)
    cotangent = torch.linspace(
        -0.5,
        0.75,
        map_result.numel(),
        dtype=map_result.dtype,
        device=map_result.device,
    ).reshape_as(map_result)
    raw = getattr(orihime.raw, case.name)

    # Collapsed low-level tier: orihime.raw.<op> re-exports the kernel bindings that
    # formerly lived under the internal orihime.ops.<op>, so orihime.raw is the single
    # low-level surface. Alias the internal module locally so it does not shadow
    # the module-level ``orihime`` referenced above.
    public_name = "sv" if case.name == "sv_linear" else case.name
    kernel_module = orihime.ops._kernels[public_name]
    for binding in kernel_module.__all__:
        assert hasattr(raw, binding), f"orihime.raw.{case.name} missing {binding}"
        assert getattr(raw, binding) is getattr(kernel_module, binding)
    assert set(kernel_module.__all__) <= set(raw.__all__)
    assert not hasattr(raw, "kernels")

    assert isinstance(raw.vjp_fields, tuple)
    assert raw.vjp_fields == case.vjp_fields
    structural_names = {
        name for name, _ in case.structural_defaults
    }
    assert structural_names.isdisjoint(raw.vjp_fields)

    with pytest.raises(TypeError):
        raw.vjp_one(*inputs, cotangent=cotangent)
    with pytest.raises(TypeError):
        raw.vjp(*inputs, cotangent=cotangent)

    for field in raw.vjp_fields:
        one = raw.vjp_one(
            *inputs,
            wrt=field,
            cotangent=cotangent,
        )
        selected = raw.vjp(
            *inputs,
            wrt=(field,),
            cotangent=cotangent,
        )
        assert isinstance(one, Tensor)
        assert not one.requires_grad
        assert torch.isfinite(one).all()
        assert isinstance(selected, dict)
        assert tuple(selected) == (field,)
        assert isinstance(selected[field], Tensor)
        assert not selected[field].requires_grad
        assert torch.isfinite(selected[field]).all()
        torch.testing.assert_close(selected[field], one)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_nn_modules_register_frozen_parameters_as_buffers(
    case: ContractCase,
) -> None:
    module_class = getattr(orihime.nn, case.nn_class)
    layer = module_class()
    expected_names = {name for name, _ in case.param_defaults}

    assert dict(layer.named_parameters()) == {}
    assert set(dict(layer.named_buffers())) == expected_names
    assert set(layer.state_dict()) == expected_names


@pytest.mark.parametrize(
    ("case", "param_name"),
    NN_CASES,
    ids=lambda value: value.name if isinstance(value, ContractCase) else value,
)
def test_nn_modules_train_each_selected_parameter(
    case: ContractCase,
    param_name: str,
) -> None:
    module_class = getattr(orihime.nn, case.nn_class)
    layer = module_class(learnable=(param_name,))
    parameter_names = {name for name, _ in case.param_defaults}

    assert set(dict(layer.named_parameters())) == {param_name}
    assert set(dict(layer.named_buffers())) == parameter_names - {param_name}
    assert set(layer.state_dict()) == parameter_names

    inputs = _inputs(case)
    output = layer(*inputs, **_topology_kwargs(case, inputs))
    weights = torch.linspace(
        0.5,
        1.5,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (output * weights).sum().backward()

    parameter = dict(layer.named_parameters())[param_name]
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad).item() > 0

    before = parameter.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    optimizer.step()
    assert not torch.equal(parameter.detach(), before)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_operator_docstrings_state_the_frozen_orientation(
    case: ContractCase,
) -> None:
    docstring = inspect.getdoc(_map_function(case))
    assert docstring is not None
    assert case.orientation in docstring.lower()


def test_osa_structural_mask_contract() -> None:
    case = CASE_BY_NAME["osa"]
    (costs,) = _inputs(case)
    disabled = torch.zeros_like(costs, dtype=torch.bool)
    torch.testing.assert_close(
        orihime.osa(costs),
        orihime.osa(costs, allowed_transpositions=disabled),
    )

    enabled = disabled.clone()
    enabled[:, 1:, 1:] = True
    result = orihime.osa(costs, allowed_transpositions=enabled)
    assert result.shape == costs.shape
    assert torch.isfinite(result).all()

    with pytest.raises(TypeError, match="torch.bool"):
        orihime.osa(
            costs,
            allowed_transpositions=enabled.to(dtype=torch.int32),
        )
    invalid_boundary = enabled.clone()
    invalid_boundary[:, 0, 1] = True
    with pytest.raises(ValueError, match="first row and first column"):
        orihime.osa(costs, allowed_transpositions=invalid_boundary)


def test_damerau_structural_sources_and_builder_contract() -> None:
    case = CASE_BY_NAME["damerau"]
    (costs,) = _inputs(case)
    sentinel = torch.full(
        (*costs.shape, 2),
        -1,
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        orihime.damerau(costs),
        orihime.damerau(costs, transposition_sources=sentinel),
    )

    batch, length_1, length_2 = costs.shape
    source_tokens = torch.zeros((batch, length_1), dtype=torch.int64)
    target_tokens = torch.zeros((batch, length_2), dtype=torch.int64)
    lengths = torch.tensor(
        ((length_1, length_2), (length_1 - 1, length_2 - 2)),
        dtype=torch.int32,
    )
    built = orihime.build_damerau_transposition_sources(
        source_tokens,
        target_tokens,
        lengths=lengths,
    )
    assert built.dtype is torch.int32
    assert built.shape == (*costs.shape, 2)
    assert built.is_contiguous()
    assert torch.any(built[0, 1:, 1:] != -1)
    assert torch.all(built[1, length_1 - 1] == -1)
    assert torch.all(built[1, :, length_2 - 2 :] == -1)
    result = orihime.damerau(
        costs,
        lengths=lengths,
        transposition_sources=built,
    )
    assert result.shape == costs.shape
    assert torch.isfinite(result).all()

    with pytest.raises(TypeError, match="torch.int32"):
        orihime.damerau(
            costs,
            transposition_sources=sentinel.to(dtype=torch.int64),
        )
    mixed_sentinel = sentinel.clone()
    mixed_sentinel[0, 1, 1] = torch.tensor(
        (0, -1),
        dtype=torch.int32,
    )
    with pytest.raises(ValueError, match="exact.*sentinel"):
        orihime.damerau(
            costs,
            transposition_sources=mixed_sentinel,
        )
    forward_source = sentinel.clone()
    forward_source[0, 1, 1] = torch.tensor(
        (1, 0),
        dtype=torch.int32,
    )
    with pytest.raises(ValueError, match="earlier predecessor"):
        orihime.damerau(
            costs,
            transposition_sources=forward_source,
        )


def test_public_version_is_available():
    assert isinstance(orihime.__version__, str)
    assert orihime.__version__.startswith("0.1.0")
