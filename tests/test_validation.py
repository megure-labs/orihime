# SPDX-License-Identifier: Apache-2.0
"""Release-blocking validation coverage for the frozen v3 Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor

import d2p
from operator_cases import OPERATOR_CASES


@dataclass(frozen=True)
class ValidationCase:
    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    params: tuple[tuple[str, float], ...]


CASES = tuple(
    ValidationCase(
        spec.name,
        spec.shapes("validation"),
        spec.matrix_params,
    )
    for spec in OPERATOR_CASES
)

PARAMETER_DIRECTIONS = tuple(
    (case, name, value)
    for case in CASES
    for name, value in case.params
)

NN_CLASSES = tuple(
    getattr(d2p.nn, spec.nn_class) for spec in OPERATOR_CASES
)

TEMPERATURE_ERROR = "temperature must be finite and strictly positive"
SCALAR_SHAPE_ERROR = "must be a scalar tensor with shape \\[\\] or \\[1\\]"
DTYPE_ERROR = "d2p computes in float32.*only torch.float32 is supported"
DOMAIN_ERROR = "supported FP32 numerical domain"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inputs(case: ValidationCase, device: torch.device) -> tuple[Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(
        8700 + CASES.index(case)
    )
    tensors = []
    for shape in case.input_shapes:
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
    case: ValidationCase,
    primary: Tensor,
) -> dict[str, Tensor]:
    if case.name == "osa":
        return {
            "allowed_transpositions": torch.zeros_like(
                primary, dtype=torch.bool
            )
        }
    if case.name == "damerau":
        return {
            "transposition_sources": torch.full(
                (*primary.shape, 2),
                -1,
                dtype=torch.int32,
                device=primary.device,
            )
        }
    return {}


def _call(
    case: ValidationCase,
    *,
    observable: str = "map",
    inputs: tuple[Tensor, ...] | None = None,
    parameter_overrides: dict[str, Any] | None = None,
    structural_overrides: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if inputs is None:
        inputs = _inputs(case, _device())
    suffix = "" if observable == "map" else f"_{observable}"
    target = getattr(d2p, f"{case.name}{suffix}")
    params = dict(case.params)
    if parameter_overrides:
        params.update(parameter_overrides)
    structural = _structural_kwargs(case, inputs[0])
    if structural_overrides:
        structural.update(structural_overrides)
    kwargs: dict[str, Any] = {**params, **structural}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return target(*inputs, **kwargs)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "bad_temperature",
    (-1.0, 0.0, float("nan"), float("inf"), -float("inf")),
)
def test_invalid_python_temperatures_raise(
    case: ValidationCase,
    bad_temperature: float,
):
    with pytest.raises((ValueError, RuntimeError), match=TEMPERATURE_ERROR):
        _call(
            case,
            parameter_overrides={"temperature": bad_temperature},
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "bad_temperature",
    (-1.0, 0.0, float("nan"), float("inf")),
)
def test_invalid_grad_tensor_temperatures_raise(
    case: ValidationCase,
    bad_temperature: float,
):
    temperature = torch.tensor(
        bad_temperature,
        device=_device(),
        requires_grad=True,
    )
    with pytest.raises(RuntimeError, match=TEMPERATURE_ERROR):
        _call(
            case,
            parameter_overrides={"temperature": temperature},
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_invalid_dynamic_temperature_raises_under_fullgraph_compile(
    case: ValidationCase,
):
    inputs = _inputs(case, _device())
    structural = _structural_kwargs(case, inputs[0])
    target = getattr(d2p, case.name)
    fixed_params = {
        name: value
        for name, value in case.params
        if name != "temperature"
    }

    def function(*values: Tensor) -> Tensor:
        *dynamic_inputs, temperature = values
        return target(
            *dynamic_inputs,
            **fixed_params,
            temperature=temperature,
            **structural,
        )

    torch._dynamo.reset()
    compiled = torch.compile(
        function,
        fullgraph=True,
        backend="aot_eager",
    )
    valid = torch.tensor(0.9, device=_device())
    assert torch.isfinite(compiled(*inputs, valid)).all()
    for bad_value in (-1.0, float("nan"), float("inf")):
        bad = torch.tensor(bad_value, device=_device())
        with pytest.raises(RuntimeError, match=TEMPERATURE_ERROR):
            compiled(*inputs, bad)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float64),
)
def test_non_fp32_dtype_escape_hatch_raises(
    case: ValidationCase,
    dtype: torch.dtype,
):
    with pytest.raises(ValueError, match=DTYPE_ERROR):
        _call(case, dtype=dtype)


@pytest.mark.parametrize(
    ("case", "parameter_name", "_"),
    PARAMETER_DIRECTIONS,
    ids=lambda value: value.name if isinstance(value, ValidationCase) else None,
)
def test_non_scalar_parameter_shapes_raise(
    case: ValidationCase,
    parameter_name: str,
    _: float,
):
    bad_parameter = torch.ones((1, 1), device=_device())
    with pytest.raises(ValueError, match=SCALAR_SHAPE_ERROR):
        _call(
            case,
            parameter_overrides={parameter_name: bad_parameter},
        )


@pytest.mark.parametrize(
    ("case", "parameter_name", "parameter_value"),
    PARAMETER_DIRECTIONS,
    ids=lambda value: value.name if isinstance(value, ValidationCase) else None,
)
def test_scalar_vector_value_jvp_matches_finite_difference(
    case: ValidationCase,
    parameter_name: str,
    parameter_value: float,
):
    inputs = _inputs(case, _device())
    parameter = torch.tensor(
        [parameter_value],
        dtype=torch.float32,
        device=_device(),
        requires_grad=True,
    )

    def function(value: Tensor) -> Tensor:
        return _call(
            case,
            observable="value",
            inputs=inputs,
            parameter_overrides={parameter_name: value},
        )

    value, tangent = torch.func.jvp(
        function,
        (parameter,),
        (torch.ones_like(parameter),),
    )
    epsilon = 1e-2
    finite_difference = (
        function(parameter.detach() + epsilon)
        - function(parameter.detach() - epsilon)
    ) / (2.0 * epsilon)

    assert torch.isfinite(value).all()
    torch.testing.assert_close(
        tangent,
        finite_difference,
        rtol=3e-3,
        atol=3e-3,
    )

    value.sum().backward()
    assert parameter.grad is not None
    assert parameter.grad.shape == parameter.shape
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_out_of_domain_scores_raise(case: ValidationCase):
    inputs = list(_inputs(case, _device()))
    inputs[0] = torch.full_like(inputs[0], 81.0)
    with pytest.raises(RuntimeError, match=DOMAIN_ERROR):
        _call(case, inputs=tuple(inputs), parameter_overrides={"temperature": 1.0})


def test_numerical_domain_boundary_is_accepted():
    scores = torch.full((1, 3, 3), 80.0, device=_device())
    result = d2p.sw(scores, temperature=1.0)
    assert torch.isfinite(result).all()


def test_out_of_domain_dynamic_scores_raise_under_fullgraph_compile():
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda scores: d2p.nw(scores, temperature=1.0),
        fullgraph=True,
        backend="aot_eager",
    )
    valid = torch.full((1, 3, 3), 80.0, device=_device())
    invalid = torch.full((1, 3, 3), 81.0, device=_device())
    assert torch.isfinite(compiled(valid)).all()
    with pytest.raises(RuntimeError, match=DOMAIN_ERROR):
        compiled(invalid)


@pytest.mark.parametrize("observable", ("map", "value", "entropy"))
@pytest.mark.parametrize(
    "lengths",
    (
        torch.tensor([[0, 4]], dtype=torch.int32),
        torch.tensor([[3, 0]], dtype=torch.int32),
    ),
)
def test_one_sided_empty_dtw_raises(
    observable: str,
    lengths: Tensor,
):
    target_name = "dtw" if observable == "map" else f"dtw_{observable}"
    target = getattr(d2p, target_name)
    costs = torch.rand(1, 4, 4, device=_device())
    with pytest.raises(
        RuntimeError,
        match="infeasible/empty instance.*one-sided empty",
    ):
        target(costs, lengths=lengths.to(_device()))


@pytest.mark.parametrize("module_type", NN_CLASSES)
@pytest.mark.parametrize(
    "bad_temperature",
    (-1.0, 0.0, float("nan"), float("inf")),
)
def test_nn_modules_reject_invalid_temperature(
    module_type: type[torch.nn.Module],
    bad_temperature: float,
):
    with pytest.raises((ValueError, RuntimeError), match=TEMPERATURE_ERROR):
        module_type(temperature=bad_temperature, device=_device())


def test_nn_module_rejects_bad_scalar_shape():
    with pytest.raises(ValueError, match=SCALAR_SHAPE_ERROR):
        d2p.nn.NeedlemanWunsch(
            temperature=torch.ones((1, 1), device=_device())
        )


def test_nn_compiled_mutated_temperature_raises():
    device = _device()
    layer = d2p.nn.NeedlemanWunsch(temperature=0.9, device=device)
    scores = torch.randn(1, 3, 3, device=device)
    torch._dynamo.reset()
    compiled = torch.compile(layer, fullgraph=True, backend="aot_eager")
    assert torch.isfinite(compiled(scores)).all()

    for bad_value in (-1.0, float("nan")):
        layer.temperature.fill_(bad_value)
        with pytest.raises(RuntimeError, match=TEMPERATURE_ERROR):
            compiled(scores)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("observable", ("map", "value", "entropy"))
def test_empty_leading_batch_is_rejected_consistently(
    case: ValidationCase,
    observable: str,
):
    inputs = tuple(value[:0] for value in _inputs(case, _device()))
    with pytest.raises(ValueError, match=r"empty leading batch \(B=0\)"):
        _call(case, observable=observable, inputs=inputs)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_primary_input_rank_has_a_public_shape_error(case: ValidationCase):
    inputs = list(_inputs(case, _device()))
    inputs[0] = inputs[0].squeeze(0)
    with pytest.raises(ValueError, match="must have shape"):
        _call(case, inputs=tuple(inputs))


def test_cky_and_eisner_validate_repeated_dimensions():
    with pytest.raises(ValueError, match="leaf_scores dimension N"):
        d2p.cky(
            torch.randn(1, 3, 3, 3),
            torch.randn(1, 4),
        )
    with pytest.raises(ValueError, match="arc_scores dimension N"):
        d2p.eisner(torch.randn(1, 3, 4))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_primary_inputs_are_fp32_outside_autocast(case: ValidationCase):
    inputs = tuple(
        value.to(dtype=torch.float64) if value.is_floating_point() else value
        for value in _inputs(case, _device())
    )
    with pytest.raises(TypeError, match="must have dtype torch.float32"):
        _call(case, inputs=inputs)

    result = _call(case, inputs=inputs, dtype=torch.float32)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("case", "parameter_name", "parameter_value"),
    PARAMETER_DIRECTIONS,
    ids=lambda value: value.name if isinstance(value, ValidationCase) else None,
)
def test_tensor_scalar_parameters_are_fp32_outside_autocast(
    case: ValidationCase,
    parameter_name: str,
    parameter_value: float,
):
    parameter = torch.tensor(
        parameter_value,
        dtype=torch.float64,
        device=_device(),
    )
    with pytest.raises(TypeError, match="tensor must have dtype torch.float32"):
        _call(
            case,
            parameter_overrides={parameter_name: parameter},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("case", "parameter_name", "parameter_value"),
    PARAMETER_DIRECTIONS,
    ids=lambda value: value.name if isinstance(value, ValidationCase) else None,
)
def test_tensor_scalar_parameters_share_the_primary_device(
    case: ValidationCase,
    parameter_name: str,
    parameter_value: float,
):
    inputs = _inputs(case, torch.device("cuda"))
    parameter = torch.tensor(parameter_value, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="same device"):
        _call(
            case,
            inputs=inputs,
            parameter_overrides={parameter_name: parameter},
        )


LENGTH_CASES = tuple(case for case in CASES if case.name != "cky")


def _valid_lengths(case: ValidationCase, primary: Tensor) -> Tensor:
    if case.name == "eisner":
        return torch.full(
            (primary.shape[0],),
            primary.shape[-1],
            dtype=torch.int32,
            device=primary.device,
        )
    bounds = torch.tensor(
        primary.shape[-2:],
        dtype=torch.int32,
        device=primary.device,
    )
    return bounds.expand(primary.shape[0], 2).contiguous()


@pytest.mark.parametrize("case", LENGTH_CASES, ids=lambda case: case.name)
def test_lengths_shape_dtype_and_bounds_are_shared(case: ValidationCase):
    inputs = _inputs(case, _device())
    valid = _valid_lengths(case, inputs[0])
    wrong_shape = valid.unsqueeze(-1) if valid.ndim == 1 else valid[:, :1]
    with pytest.raises(ValueError, match="lengths must have shape"):
        _call(
            case,
            inputs=inputs,
            structural_overrides={"lengths": wrong_shape},
        )
    with pytest.raises(TypeError, match="lengths must have dtype torch.int32"):
        _call(
            case,
            inputs=inputs,
            structural_overrides={"lengths": valid.to(torch.int64)},
        )

    out_of_range = valid.clone()
    first_bound = (
        inputs[0].shape[-1]
        if case.name == "eisner"
        else inputs[0].shape[-2]
    )
    out_of_range.reshape(-1)[0] = first_bound + 1
    with pytest.raises(RuntimeError, match="within the padded input shape"):
        _call(
            case,
            inputs=inputs,
            structural_overrides={"lengths": out_of_range},
        )


def test_pairwise_lengths_must_be_contiguous():
    scores = torch.randn(2, 3, 3, device=_device())
    lengths = torch.tensor(
        [[3, 3], [3, 3]],
        dtype=torch.int32,
        device=_device(),
    ).t()
    assert not lengths.is_contiguous()
    with pytest.raises(ValueError, match="lengths must be contiguous"):
        d2p.sw(scores, lengths=lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lengths_and_mask_share_the_primary_device():
    scores = torch.randn(1, 3, 3, device="cuda")
    lengths = torch.tensor([[3, 3]], dtype=torch.int32, device="cpu")
    with pytest.raises(ValueError, match="same device"):
        d2p.sw(scores, lengths=lengths)
    mask = torch.zeros(1, 3, 3, dtype=torch.bool, device="cpu")
    with pytest.raises(ValueError, match="same device"):
        d2p.sw(scores, mask=mask)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_raw_cotangent_dtype_and_contiguity_are_exact(case: ValidationCase):
    inputs = _inputs(case, _device())
    map_result = _call(case, inputs=inputs)
    raw = getattr(d2p.raw, case.name)
    field = raw.vjp_fields[0]

    with pytest.raises(TypeError, match="cotangent.*dtype torch.float32"):
        raw.vjp_one(
            *inputs,
            wrt=field,
            cotangent=torch.ones_like(map_result, dtype=torch.float64),
        )

    storage = torch.ones(
        (*map_result.shape, 2),
        dtype=torch.float32,
        device=map_result.device,
    )
    noncontiguous = storage[..., 0]
    assert noncontiguous.shape == map_result.shape
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="cotangent.*contiguous"):
        raw.vjp_one(
            *inputs,
            wrt=field,
            cotangent=noncontiguous,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_raw_cotangent_shares_the_map_device():
    scores = torch.randn(1, 3, 3, device="cuda")
    with pytest.raises(ValueError, match="cotangent.*same device"):
        d2p.raw.sw.vjp_one(
            scores,
            wrt="temperature",
            cotangent=torch.ones(1, 3, 3, device="cpu"),
        )
