# SPDX-License-Identifier: Apache-2.0
"""Answer-preserving finite-sentinel masking across every v3 family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch
from torch import Tensor

import orihime
from operator_cases import OPERATOR_CASES

# Independent exclusion oracle for masking: the TEST sets a masked cell to this ratio
# times temperature (in-domain, <= 80), NOT orihime's internal sentinel, so the test cannot
# pass by comparing the implementation against itself.
_INDEPENDENT_EXCLUDED_RATIO = 75.0


@dataclass(frozen=True)
class MaskingCase:
    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    cost_native: bool
    params: tuple[tuple[str, float], ...]
    off_optimal_index: tuple[int, ...]
    padded_index: tuple[int, ...]


CASES = tuple(
    MaskingCase(
        name=spec.name,
        input_shapes=spec.shapes("masking"),
        cost_native=spec.cost_native,
        params=spec.mask_params,
        off_optimal_index=spec.off_optimal_index,
        padded_index=spec.padded_index,
    )
    for spec in OPERATOR_CASES
)

OBSERVABLES = ("map", "value", "entropy")
SCENARIOS = ("off_optimal", "padded")
DEVICES = ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
FINITE_SENTINEL = 1.0e4


def _inputs(case: MaskingCase, device: torch.device) -> tuple[Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(
        9600 + CASES.index(case)
    )
    tensors = []
    for shape in case.input_shapes:
        if case.cost_native:
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
    case: MaskingCase,
    scenario: str,
    device: torch.device,
) -> dict[str, Any]:
    if scenario != "padded" or case.name == "cky":
        return {}
    if case.name == "eisner":
        return {"lengths": torch.tensor([3], dtype=torch.int32, device=device)}
    if case.name == "mas":
        lengths = [[4, 3]]
    else:
        lengths = [[3, 3]]
    return {
        "lengths": torch.tensor(lengths, dtype=torch.int32, device=device)
    }


def _target(case: MaskingCase, observable: str) -> Callable[..., Tensor]:
    suffix = "" if observable == "map" else f"_{observable}"
    return getattr(orihime, f"{case.name}{suffix}")


def _masked_and_reference(
    case: MaskingCase,
    scenario: str,
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[int, ...]]:
    inputs = _inputs(case, device)
    masked = tuple(value.clone() for value in inputs)
    reference = tuple(value.clone() for value in inputs)
    index = (
        case.off_optimal_index
        if scenario == "off_optimal"
        else case.padded_index
    )
    temperature = dict(case.params)["temperature"]
    infinity = float("inf") if case.cost_native else -float("inf")
    # Independent, temperature-scaled exclusion the test constructs itself (not orihime's sentinel).
    excluded = (
        _INDEPENDENT_EXCLUDED_RATIO
        if case.cost_native
        else -_INDEPENDENT_EXCLUDED_RATIO
    ) * temperature
    masked[0][index] = infinity
    reference[0][index] = excluded
    return masked, reference, index


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("observable", OBSERVABLES)
def test_documented_mask_is_finite_and_matches_excluded_reference(
    device_name: str,
    case: MaskingCase,
    scenario: str,
    observable: str,
):
    device = torch.device(device_name)
    masked, reference, index = _masked_and_reference(
        case, scenario, device
    )
    kwargs = {
        **dict(case.params),
        **_structural_kwargs(case, scenario, device),
    }
    target = _target(case, observable)

    masked_result = target(*masked, **kwargs)
    reference_result = target(*reference, **kwargs)

    assert torch.isfinite(masked_result).all()
    torch.testing.assert_close(
        masked_result,
        reference_result,
        rtol=1e-4,
        atol=1e-4,
    )
    if observable == "map":
        # a masked cell's marginal is exp(-ratio): vanishingly small, not exactly zero
        assert masked_result[index].abs().item() < 1e-6


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_wrong_infinity_orientation_is_rejected(
    device_name: str,
    case: MaskingCase,
):
    device = torch.device(device_name)
    inputs = list(_inputs(case, device))
    wrong_infinity = -float("inf") if case.cost_native else float("inf")
    inputs[0][case.off_optimal_index] = wrong_infinity

    with pytest.raises(RuntimeError, match="unsupported infinity"):
        _target(case, "map")(*inputs, **dict(case.params))


@pytest.mark.parametrize(
    ("name", "documented_infinity", "wrong_infinity"),
    (("sw", -float("inf"), float("inf")), ("dtw", float("inf"), -float("inf"))),
)
def test_masking_validation_is_fullgraph_traceable(
    name: str,
    documented_infinity: float,
    wrong_infinity: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = getattr(orihime, name)

    def function(values: Tensor) -> Tensor:
        return target(values, temperature=1.0)

    torch._dynamo.reset()
    compiled = torch.compile(
        function,
        fullgraph=True,
        backend="aot_eager",
    )
    values = 0.2 * torch.randn(1, 4, 4, device=device)
    documented = values.clone()
    documented[0, 0, 2] = documented_infinity
    assert torch.isfinite(compiled(documented)).all()

    wrong = values.clone()
    wrong[0, 0, 2] = wrong_infinity
    with pytest.raises(RuntimeError, match="unsupported infinity"):
        compiled(wrong)
