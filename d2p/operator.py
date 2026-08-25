# SPDX-License-Identifier: Apache-2.0
"""Shared Python surface for differentiable dynamic-programming operators."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._pt2_ops import (
    functorch_batching_active,
    functorch_jvp_active,
    functorch_transform_active,
    pt2_compiling,
    unwrap_grad_tracking_tensor,
)


Param = float | Tensor
_MAX_ABS_SCORE_TEMPERATURE_RATIO = 80.0
_MASK_SENTINEL_MAGNITUDE = 1.0e4
# Masked cells are placed this many temperature-units past zero so their soft-max/
# soft-min weight is exp(-ratio) at EVERY temperature (answer-preserving masking).
# A fixed magnitude stops excluding once temperature grows. Kept < the domain ratio.
_MASK_EXCLUSION_RATIO = 200.0  # FP32 underflow + detour safety margin (temperature-units)

# Operators whose value is a soft-MIN (cost-native); entropy = -dV/dtemp there,
# +dV/dtemp for the score-native (soft-MAX) operators. See docs/API_V3_FROZEN.md.
_COST_NATIVE = frozenset({"dtw", "lev", "osa", "damerau"})

_PUBLIC_PARAMETER_DOCS = {
    "pair_scores": "FP32 pair-score tensor shaped ``[B, L1, L2]``.",
    "costs": "FP32 cost tensor shaped ``[B, L1, L2]``.",
    "match_scores": "FP32 match-score tensor shaped ``[B, L1, L2]``.",
    "substitution_costs": (
        "FP32 substitution-cost tensor shaped ``[B, L1, L2]``."
    ),
    "scores": "FP32 MAS score tensor shaped ``[B, T, S]``.",
    "merge_scores": "FP32 CKY merge chart shaped ``[B, N, N, N]``.",
    "leaf_scores": "FP32 CKY leaf chart shaped ``[B, N]``.",
    "arc_scores": "FP32 dependency-arc chart shaped ``[B, N, N]``.",
    "source_tokens": "Integer source-token IDs shaped ``[B, L1]``.",
    "target_tokens": "Integer target-token IDs shaped ``[B, L2]``.",
    "gap_score": "Linear-gap score as a Python number or FP32 scalar tensor.",
    "gap_open_score": (
        "Gap-open score as a Python number or FP32 scalar tensor."
    ),
    "gap_extend_score": (
        "Gap-extension score as a Python number or FP32 scalar tensor."
    ),
    "insertion_cost": (
        "Insertion cost as a Python number or FP32 scalar tensor."
    ),
    "deletion_cost": (
        "Deletion cost as a Python number or FP32 scalar tensor."
    ),
    "transposition_cost": (
        "Transposition cost as a Python number or FP32 scalar tensor."
    ),
    "temperature": (
        "Finite positive temperature as a Python number or FP32 scalar tensor."
    ),
    "lengths": (
        "Optional contiguous ``torch.int32`` active lengths on the input "
        "device: ``[B, 2]`` for pairwise operators or ``[B]`` for Eisner."
    ),
    "bandwidth": (
        "Optional non-negative Sakoe-Chiba radius; ``None`` is unrestricted."
    ),
    "allowed_transpositions": (
        "Optional boolean OSA topology tensor shaped like the costs; ``True`` "
        "allows the corresponding adjacent-transposition edge."
    ),
    "transposition_sources": (
        "Optional contiguous ``torch.int32`` Damerau predecessor coordinates "
        "shaped ``[B, L1, L2, 2]``."
    ),
    "mask": (
        "Optional boolean tensor shaped like the primary input; ``True`` "
        "excludes that DP cell."
    ),
    "dtype": (
        "Optional compute override. Only ``torch.float32`` is accepted; use "
        "it to cast floating inputs to the native accumulation dtype."
    ),
}


def _document_public_function(
    function: Callable[..., Tensor],
) -> Callable[..., Tensor]:
    """Append the shared, signature-complete public argument contract."""

    parameters = tuple(inspect.signature(function).parameters)
    missing = [name for name in parameters if name not in _PUBLIC_PARAMETER_DOCS]
    if missing:
        raise RuntimeError(
            f"missing public documentation for {function.__name__}: {missing}"
        )
    if function.__name__ == "build_damerau_transposition_sources":
        result = (
            "Contiguous ``torch.int32`` predecessor coordinates shaped "
            "``[B, L1, L2, 2]``."
        )
    elif function.__name__ == "cky_leaf_map":
        result = "A detached FP32 leaf derivative tensor shaped like ``leaf_scores``."
    elif function.__name__.endswith(("_value", "_entropy")):
        result = "An FP32 tensor with one value per batch item, shaped ``[B]``."
    else:
        result = "An FP32 map tensor shaped like the primary tensor input."
    args = "\n".join(
        f"    {name}: {_PUBLIC_PARAMETER_DOCS[name]}" for name in parameters
    )
    function.__doc__ = (
        (function.__doc__ or "").rstrip()
        + "\n\nArgs:\n"
        + args
        + "\n\nReturns:\n    "
        + result
        + "\n\nRaises:\n"
        + "    TypeError: An input has the wrong tensor kind, dtype, or scalar form.\n"
        + "    ValueError: A shape, device, length, mask, or static option is "
        + "invalid.\n"
        + "    RuntimeError: Dynamic values violate the numerical or length domain."
    )
    return function


def _graph_safe_assert(condition: Tensor, message: str) -> None:
    """Raise at runtime while remaining traceable by fullgraph PT2.

    ``aten::_assert_async`` is Dynamo-traceable.  Move a CUDA condition to the
    host first so an invalid user input raises a normal, recoverable
    ``RuntimeError`` instead of poisoning the CUDA context with a device-side
    assertion.  Functorch wrappers are unwrapped because the assertion itself
    has no batching rule.
    """

    if not isinstance(condition, Tensor) or condition.dtype != torch.bool:
        raise TypeError("internal validation condition must be a bool tensor")

    physical_condition = condition
    if not pt2_compiling():
        functorch = torch._C._functorch
        while functorch.is_gradtrackingtensor(
            physical_condition
        ) or functorch.is_batchedtensor(physical_condition):
            physical_condition = functorch.get_unwrapped(physical_condition)

    physical_condition = physical_condition.all()
    if physical_condition.device.type != "cpu":
        physical_condition = physical_condition.to(device="cpu")
    torch._assert_async(physical_condition, message)


def _validate_scalar_parameter_shape(value: Param, name: str) -> None:
    """Accept only Python scalars, 0-D tensors, and one-element vectors."""

    if not isinstance(value, Tensor):
        return
    if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
        return
    raise ValueError(
        f"{name} must be a scalar tensor with shape [] or [1], "
        f"got shape {tuple(value.shape)}"
    )


def _validate_derivative_vector(
    primary: Tensor,
    vector: Tensor,
    *,
    name: str,
    primary_name: str,
    error_type: type[Exception] | None = RuntimeError,
) -> Tensor:
    """Validate a strict FP32 derivative vector before native dispatch.

    Named low-level derivative primitives use ``RuntimeError`` for their
    established kernel-boundary validation failures.  The raw VJP tier passes
    ``error_type=None`` so it can retain its public ``TypeError``/``ValueError``
    split while sharing the same shape, dtype, device, and layout contract.
    No normalization belongs here: a user-supplied vector must already be
    contiguous when it reaches a native kernel.
    """

    if not isinstance(primary, Tensor):
        raise TypeError(f"{primary_name} must be a tensor")

    def fail(message: str, default: type[Exception]) -> None:
        raise (error_type or default)(message)

    if not isinstance(vector, Tensor):
        fail(f"{name} must be a tensor", TypeError)
    if vector.shape != primary.shape:
        fail(
            f"{name} must have same shape as {primary_name}; "
            f"got {tuple(vector.shape)} and {tuple(primary.shape)}",
            ValueError,
        )
    if vector.dtype != torch.float32:
        fail(
            f"{name} must have dtype torch.float32, got {vector.dtype}",
            TypeError,
        )
    if vector.device != primary.device:
        fail(
            f"{name} must be on same device as {primary_name}; "
            f"got {vector.device} and {primary.device}",
            ValueError,
        )
    if not vector.is_contiguous():
        fail(f"{name} must be contiguous", ValueError)
    return vector


def _validate_temperature(value: Param) -> None:
    """Require one finite, strictly positive temperature value."""

    _validate_scalar_parameter_shape(value, "temperature")
    message = "temperature must be finite and strictly positive"
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise TypeError("temperature tensor must have a floating-point dtype")
        _graph_safe_assert(torch.isfinite(value) & (value > 0), message)
        return

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("temperature must be a real number or tensor") from exc
    if not math.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{message}, got {scalar}")


def _temperature_on(value: Param, tensor: Tensor) -> float | Tensor:
    if isinstance(value, Tensor):
        return value.to(device=tensor.device, dtype=tensor.dtype)
    return float(value)


def _validate_numerical_domain(
    tensor_input_names: tuple[str, ...],
    tensor_inputs: tuple[Tensor, ...],
    param_names: tuple[str, ...],
    params: tuple[Param, ...],
    *,
    cost_native: bool = False,
) -> None:
    """Enforce the documented conservative FP32 score/temperature domain."""

    try:
        temperature_index = param_names.index("temp")
    except ValueError as exc:
        raise RuntimeError("operator is missing its temperature parameter") from exc
    temperature = params[temperature_index]
    _validate_temperature(temperature)

    limit = _MAX_ABS_SCORE_TEMPERATURE_RATIO
    message = (
        "d2p's supported FP32 numerical domain requires every finite "
        f"|score or cost| / temperature to be <= {limit:g}"
    )

    for name, value in zip(
        tensor_input_names, tensor_inputs, strict=True
    ):
        if not value.is_floating_point():
            continue
        mask_value = (
            _MASK_SENTINEL_MAGNITUDE
            if cost_native
            else -_MASK_SENTINEL_MAGNITUDE
        )
        documented_mask = (
            torch.isposinf(value)
            if cost_native
            else torch.isneginf(value)
        )
        wrong_infinity = (
            torch.isneginf(value)
            if cost_native
            else torch.isposinf(value)
        )
        orientation = "+inf cost masks" if cost_native else "-inf score masks"
        _graph_safe_assert(
            ~wrong_infinity,
            f"{name} contains an unsupported infinity; use {orientation}",
        )
        scaled_limit = limit * _temperature_on(temperature, value)
        valid = ~torch.isnan(value) & (
            documented_mask
            | (value == mask_value)
            | (value.abs() <= scaled_limit)
        )
        _graph_safe_assert(valid, f"{message}; {name} is out of domain")

    for name, value in zip(param_names, params, strict=True):
        if name == "temp":
            continue
        _validate_scalar_parameter_shape(value, name)
        if isinstance(value, Tensor):
            if not value.is_floating_point():
                raise TypeError(
                    f"{name} tensor must have a floating-point dtype"
                )
            scaled_limit = limit * _temperature_on(temperature, value)
            _graph_safe_assert(
                torch.isfinite(value) & (value.abs() <= scaled_limit),
                f"{message}; {name} is out of domain",
            )
            continue

        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{message}; {name} must be finite")
        if isinstance(temperature, Tensor):
            scaled_limit = limit * temperature
            _graph_safe_assert(
                scaled_limit >= abs(scalar),
                f"{message}; {name} is out of domain",
            )
        elif abs(scalar) > limit * float(temperature):
            raise ValueError(f"{message}; {name} is out of domain")


def _normalize_masked_tensor_inputs(
    tensor_inputs: tuple[Tensor, ...],
    *,
    cost_native: bool,
    temperature: Param,
) -> tuple[Tensor, ...]:
    """Replace documented infinite masks with a finite exclusion sentinel.

    A masked cell is placed past the RETAINED extremum by more than the maximum
    possible cost of any path through the grid -- ``(sum(shape[1:]) * 80 +
    _MASK_EXCLUSION_RATIO) * temperature`` units: ``retained_amax - offset`` for a
    score-native soft-max, ``retained_amin + offset`` for a cost-native soft-min.
    Going through the masked cell is then strictly more expensive than any detour,
    so it is excluded at ANY in-domain scale -- a fixed or zero-relative sentinel
    can instead be *cheaper* than a detour at the ``|score|/temperature = 80``
    domain edge, silently changing the answer for min-DPs. Temperature and the
    retained extremum are detached so the mask leaks no grad/forward tangent.
    Graph-safe (no ``.item()``).
    """

    normalized = []
    for value in tensor_inputs:
        if not value.is_floating_point():
            normalized.append(value)
            continue
        mask = torch.isposinf(value) if cost_native else torch.isneginf(value)
        # A mask is a structural boundary: detach temperature (and the retained extremum)
        # so the sentinel carries no grad/forward-tangent -- otherwise ``torch.where`` would
        # leak a tangent onto every (even unmasked) tensor, corrupting downstream autograd.
        temp = _temperature_on(temperature, value)
        if isinstance(temp, Tensor):
            temp = temp.detach()
        # Exclude the cell by making it cost more than ANY whole path could: a monotonic DP
        # path visits at most sum(shape[1:]) cells, each bounded by the domain ratio 80, so an
        # offset of (path_bound*80 + margin) temperature-units past the retained extremum makes
        # going through the masked cell strictly more expensive than any detour, at any input
        # scale -- and its Boltzmann weight underflows to exactly 0 in FP32. A fixed or
        # zero-relative offset fails for min-DPs at the domain edge (a finite sentinel can be
        # cheaper than the detour). Detached so the mask leaks no grad/forward tangent.
        dims = tuple(range(1, value.ndim))
        path_len = float(sum(value.shape[1:])) if value.ndim > 1 else 1.0
        offset = (path_len * _MAX_ABS_SCORE_TEMPERATURE_RATIO + _MASK_EXCLUSION_RATIO) * temp
        if cost_native:
            retained = value.amin(dim=dims, keepdim=True) if dims else value
            retained = torch.where(torch.isfinite(retained), retained, retained.new_zeros(()))
            sentinel = retained.detach() + offset
        else:
            retained = value.amax(dim=dims, keepdim=True) if dims else value
            retained = torch.where(torch.isfinite(retained), retained, retained.new_zeros(()))
            sentinel = retained.detach() - offset
        sentinel = sentinel.to(dtype=value.dtype, device=value.device)
        normalized.append(torch.where(mask, sentinel, value))
    return tuple(normalized)


@dataclass(frozen=True)
class KernelCall:
    """Inputs shared by the four per-operator kernel callables."""

    primary: Tensor
    params: tuple[Param, ...]
    lengths: Tensor | None
    config: Mapping[str, Any]
    other_inputs: tuple[Tensor, ...] = ()

    @property
    def tensor_inputs(self) -> tuple[Tensor, ...]:
        """All tensor-valued operator inputs, including the primary one."""

        return (self.primary, *self.other_inputs)


@dataclass(frozen=True)
class ForwardPass:
    """Value and tensor state produced by an operator's forward DP."""

    value: Tensor
    saved_tensors: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class BackwardPass:
    """Marginals and byproducts produced by an operator's backward DP."""

    marginals: Tensor
    entropy: Tensor
    param_grads: tuple[Tensor, ...]
    saved_tensors: tuple[Tensor, ...] = ()
    input_grads: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class OperatorKernels:
    """The callable roles needed to bind one operator.

    ``forward`` runs the forward DP. ``backward`` consumes its result and
    runs the backward DP. ``map_backward`` contracts the score-space
    derivative with a map cotangent. ``param_backward`` runs the parameter
    U+W pass named by its third argument; a ``None`` cotangent requests the
    raw sensitivity field, while a tensor requests its contraction.
    ``map_jvp`` is needed only when a family has multiple tensor-valued
    inputs whose cross-Jacobian action cannot be recovered from
    ``map_backward`` alone.
    """

    forward: Callable[[KernelCall], ForwardPass]
    backward: Callable[[KernelCall, ForwardPass], BackwardPass]
    map_backward: Callable[
        [KernelCall, ForwardPass | BackwardPass, Tensor],
        Tensor | Sequence[Tensor] | Mapping[str, Tensor],
    ]
    param_backward: Callable[
        [KernelCall, ForwardPass | BackwardPass, str, Tensor | None], Tensor
    ]
    map_jvp: (
        Callable[
            [
                KernelCall,
                ForwardPass | BackwardPass,
                tuple[Tensor | None, ...],
            ],
            Tensor,
        ]
        | None
    ) = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FunctionSpec:
    """Static metadata for a flattened custom-Function invocation."""

    name: str
    tensor_input_names: tuple[str, ...]
    param_names: tuple[str, ...]
    tensor_input_indices: tuple[int, ...]
    param_specs: tuple[tuple[str, Any], ...]
    lengths_spec: tuple[str, Any]
    config_specs: tuple[tuple[str, str, Any], ...]


_FUNCTION_KERNELS: dict[str, OperatorKernels] = {}


def _as_tensor_tuple(values: tuple[Tensor, ...], label: str) -> tuple[Tensor, ...]:
    values = tuple(values)
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"{label} must contain only tensors")
    return values


def _normalize_forward(result: ForwardPass) -> ForwardPass:
    if not isinstance(result, ForwardPass):
        raise TypeError("forward kernel must return ForwardPass")
    if not isinstance(result.value, Tensor):
        raise TypeError("ForwardPass.value must be a tensor")
    return ForwardPass(
        value=result.value,
        saved_tensors=_as_tensor_tuple(
            result.saved_tensors, "ForwardPass.saved_tensors"
        ),
    )


def _normalize_backward(
    result: BackwardPass,
    param_count: int,
    input_count: int,
) -> BackwardPass:
    if not isinstance(result, BackwardPass):
        raise TypeError("backward kernel must return BackwardPass")
    if not isinstance(result.marginals, Tensor):
        raise TypeError("BackwardPass.marginals must be a tensor")
    if not isinstance(result.entropy, Tensor):
        raise TypeError("BackwardPass.entropy must be a tensor")
    param_grads = _as_tensor_tuple(
        result.param_grads, "BackwardPass.param_grads"
    )
    if len(param_grads) != param_count:
        raise ValueError(
            "backward kernel returned "
            f"{len(param_grads)} parameter gradients; expected {param_count}"
        )
    input_grads = _as_tensor_tuple(
        result.input_grads, "BackwardPass.input_grads"
    )
    if not input_grads and input_count == 1:
        input_grads = (result.marginals,)
    if len(input_grads) != input_count:
        raise ValueError(
            "backward kernel returned "
            f"{len(input_grads)} tensor-input gradients; expected {input_count}"
        )
    return BackwardPass(
        marginals=result.marginals,
        entropy=result.entropy,
        param_grads=param_grads,
        saved_tensors=_as_tensor_tuple(
            result.saved_tensors, "BackwardPass.saved_tensors"
        ),
        input_grads=input_grads,
    )


def _normalize_map_backward(
    result: Tensor | Sequence[Tensor] | Mapping[str, Tensor],
    input_names: tuple[str, ...],
) -> tuple[Tensor, ...]:
    if isinstance(result, Tensor):
        if len(input_names) != 1:
            raise ValueError(
                "map_backward must return one gradient per tensor input"
            )
        return (result,)

    if isinstance(result, Mapping):
        missing = [name for name in input_names if name not in result]
        unknown = set(result) - set(input_names)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError(
                "map_backward returned invalid tensor-input fields: "
                + "; ".join(details)
            )
        values = tuple(result[name] for name in input_names)
    elif isinstance(result, Sequence):
        values = tuple(result)
    else:
        raise TypeError(
            "map_backward must return a tensor, tensor sequence, or mapping"
        )

    values = _as_tensor_tuple(values, "map_backward result")
    if len(values) != len(input_names):
        raise ValueError(
            "map_backward returned "
            f"{len(values)} tensor-input gradients; expected {len(input_names)}"
        )
    return values


def _normalize_kernels(
    kernels: OperatorKernels | Mapping[str, Any],
) -> OperatorKernels:
    if isinstance(kernels, OperatorKernels):
        result = kernels
    elif isinstance(kernels, Mapping):
        required = ("forward", "backward", "map_backward", "param_backward")
        missing = [name for name in required if name not in kernels]
        unknown = set(kernels) - {*required, "map_jvp", "config"}
        if missing:
            raise ValueError(
                "kernels is missing required callables: " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "kernels contains unknown fields: " + ", ".join(sorted(unknown))
            )
        result = OperatorKernels(
            forward=kernels["forward"],
            backward=kernels["backward"],
            map_backward=kernels["map_backward"],
            param_backward=kernels["param_backward"],
            map_jvp=kernels.get("map_jvp"),
            config=kernels.get("config", {}),
        )
    else:
        raise TypeError("kernels must be an OperatorKernels or mapping")

    for name in ("forward", "backward", "map_backward", "param_backward"):
        if not callable(getattr(result, name)):
            raise TypeError(f"kernels.{name} must be callable")
    if result.map_jvp is not None and not callable(result.map_jvp):
        raise TypeError("kernels.map_jvp must be callable or None")
    if not isinstance(result.config, Mapping):
        raise TypeError("kernels.config must be a mapping")
    return OperatorKernels(
        forward=result.forward,
        backward=result.backward,
        map_backward=result.map_backward,
        param_backward=result.param_backward,
        map_jvp=result.map_jvp,
        config=MappingProxyType(dict(result.config)),
    )


def _make_call(
    tensor_inputs: tuple[Tensor, ...],
    params: tuple[Param, ...],
    lengths: Tensor | None,
    config_items: tuple[tuple[str, Any], ...],
) -> KernelCall:
    return KernelCall(
        primary=tensor_inputs[0],
        params=params,
        lengths=lengths,
        config=MappingProxyType(dict(config_items)),
        other_inputs=tuple(tensor_inputs[1:]),
    )


def _flatten_value(
    value: Any,
    flat_tensors: list[Tensor],
) -> tuple[str, Any]:
    if isinstance(value, Tensor):
        index = len(flat_tensors)
        flat_tensors.append(value)
        return ("tensor", index)
    if not isinstance(
        value,
        (type(None), bool, int, float, str, tuple),
    ):
        raise TypeError(
            "non-tensor Function metadata must be hashable"
        )
    return ("value", value)


def _flatten_function_inputs(
    *,
    name: str,
    tensor_input_names: tuple[str, ...],
    param_names: tuple[str, ...],
    kernels: OperatorKernels,
    tensor_inputs: tuple[Tensor, ...],
    params: tuple[Param, ...],
    lengths: Tensor | None,
    config_items: tuple[tuple[str, Any], ...],
) -> tuple[_FunctionSpec, tuple[Tensor, ...]]:
    flat_tensors = list(tensor_inputs)
    tensor_input_indices = tuple(range(len(tensor_inputs)))
    param_specs = tuple(
        _flatten_value(param, flat_tensors) for param in params
    )
    lengths_spec = _flatten_value(lengths, flat_tensors)
    config_specs = tuple(
        (key, *_flatten_value(value, flat_tensors))
        for key, value in config_items
    )
    spec = _FunctionSpec(
        name=name,
        tensor_input_names=tensor_input_names,
        param_names=param_names,
        tensor_input_indices=tensor_input_indices,
        param_specs=param_specs,
        lengths_spec=lengths_spec,
        config_specs=config_specs,
    )
    if _FUNCTION_KERNELS[name] is not kernels:
        _FUNCTION_KERNELS[name] = kernels
    return spec, tuple(flat_tensors)


def _restore_value(
    spec: tuple[str, Any],
    flat_tensors: tuple[Tensor, ...],
) -> Any:
    kind, value = spec
    return flat_tensors[value] if kind == "tensor" else value


def _restore_function_call(
    spec: _FunctionSpec,
    flat_tensors: tuple[Tensor, ...],
) -> KernelCall:
    tensor_inputs = tuple(
        flat_tensors[index] for index in spec.tensor_input_indices
    )
    params = tuple(
        _restore_value(param_spec, flat_tensors)
        for param_spec in spec.param_specs
    )
    lengths = _restore_value(spec.lengths_spec, flat_tensors)
    config_items = tuple(
        (key, _restore_value((kind, value), flat_tensors))
        for key, kind, value in spec.config_specs
    )
    return _make_call(tensor_inputs, params, lengths, config_items)


def _run_forward(
    spec: _FunctionSpec,
    call: KernelCall,
) -> ForwardPass:
    return _normalize_forward(_FUNCTION_KERNELS[spec.name].forward(call))


def _run_backward(
    spec: _FunctionSpec,
    call: KernelCall,
    forward: ForwardPass,
) -> BackwardPass:
    kernels = _FUNCTION_KERNELS[spec.name]
    return _normalize_backward(
        kernels.backward(call, forward),
        len(spec.param_names),
        len(spec.tensor_input_names),
    )


def _run_map_backward(
    spec: _FunctionSpec,
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    grad_map: Tensor,
) -> tuple[Tensor, ...]:
    kernels = _FUNCTION_KERNELS[spec.name]
    return _normalize_map_backward(
        kernels.map_backward(call, state, grad_map),
        spec.tensor_input_names,
    )


def _run_map_jvp(
    spec: _FunctionSpec,
    call: KernelCall,
    state: ForwardPass | BackwardPass,
    tangents: tuple[Tensor | None, ...],
) -> Tensor:
    kernels = _FUNCTION_KERNELS[spec.name]
    if kernels.map_jvp is not None:
        result = kernels.map_jvp(call, state, tangents)
        if not isinstance(result, Tensor):
            raise TypeError("map_jvp must return a tensor")
        return result

    if any(tangent is not None for tangent in tangents[1:]):
        raise RuntimeError(
            f"{spec.name} map JVP is missing a cross-input rule"
        )
    primary_tangent = tangents[0]
    if primary_tangent is None:
        return state.marginals.new_zeros(state.marginals.shape)
    return _run_map_backward(
        spec,
        call,
        state,
        primary_tangent,
    )[0]


def _save_function_state(
    ctx: Any,
    spec: _FunctionSpec,
    flat_tensors: tuple[Tensor, ...],
    state_tensors: tuple[Tensor, ...],
    *,
    save_for_forward: bool = False,
) -> None:
    ctx.function_name = spec.name
    ctx.tensor_input_names = spec.tensor_input_names
    ctx.param_names = spec.param_names
    ctx.tensor_input_indices = spec.tensor_input_indices
    ctx.param_specs = spec.param_specs
    ctx.lengths_spec = spec.lengths_spec
    ctx.config_specs = spec.config_specs
    ctx.flat_tensor_count = len(flat_tensors)
    ctx.save_for_backward(*flat_tensors, *state_tensors)
    if save_for_forward:
        ctx.save_for_forward(*flat_tensors, *state_tensors)


def _restore_function_spec(ctx: Any) -> _FunctionSpec:
    return _FunctionSpec(
        name=ctx.function_name,
        tensor_input_names=ctx.tensor_input_names,
        param_names=ctx.param_names,
        tensor_input_indices=ctx.tensor_input_indices,
        param_specs=ctx.param_specs,
        lengths_spec=ctx.lengths_spec,
        config_specs=ctx.config_specs,
    )


def _restore_function_state(
    ctx: Any,
    *,
    transform_safe: bool = False,
) -> tuple[_FunctionSpec, KernelCall, tuple[Tensor, ...]]:
    saved = tuple(ctx.saved_tensors)
    if transform_safe:
        saved = tuple(
            unwrap_grad_tracking_tensor(tensor) for tensor in saved
        )
    flat_tensors = tuple(saved[: ctx.flat_tensor_count])
    state_tensors = tuple(saved[ctx.flat_tensor_count :])
    spec = _restore_function_spec(ctx)
    call = _restore_function_call(spec, flat_tensors)
    return spec, call, state_tensors


def _function_grad_index(tensor_index: int) -> int:
    # The static _FunctionSpec is the first argument to ``apply``.
    return tensor_index + 1


def _function_param_grad_indices(
    spec: _FunctionSpec,
) -> tuple[int | None, ...]:
    return tuple(
        _function_grad_index(value) if kind == "tensor" else None
        for kind, value in spec.param_specs
    )


def _function_needs_grad(
    spec: _FunctionSpec,
    flat_tensors: tuple[Tensor, ...],
) -> bool:
    differentiable_indices = [
        *spec.tensor_input_indices,
        *(
            value
            for kind, value in spec.param_specs
            if kind == "tensor"
        ),
    ]
    return torch.is_grad_enabled() and any(
        flat_tensors[index].requires_grad
        for index in differentiable_indices
    )


def _function_needs_transform_boundary(
    spec: _FunctionSpec,
    flat_tensors: tuple[Tensor, ...],
) -> bool:
    needs_grad = _function_needs_grad(spec, flat_tensors)
    if pt2_compiling():
        return needs_grad
    return needs_grad or functorch_transform_active()


def _use_jvp_function() -> bool:
    return (
        not pt2_compiling()
        and functorch_jvp_active()
        and not functorch_batching_active()
    )


def _expand_batch_grad(grad_value: Tensor, target: Tensor) -> Tensor:
    while grad_value.ndim < target.ndim:
        grad_value = grad_value.unsqueeze(-1)
    return grad_value


def _contract_tensor_direction(
    derivative: Tensor,
    tangent: Tensor,
    output: Tensor,
) -> Tensor:
    contribution = derivative * tangent
    if contribution.ndim < output.ndim:
        raise RuntimeError(
            "derivative direction has fewer dimensions than its output"
        )
    reduction_dims = tuple(range(output.ndim, contribution.ndim))
    if reduction_dims:
        contribution = contribution.sum(dim=reduction_dims)
    if contribution.shape != output.shape:
        contribution = contribution.expand_as(output)
    return contribution


def _accumulate_direction(
    total: Tensor | None,
    contribution: Tensor,
) -> Tensor:
    return contribution if total is None else total + contribution


def _function_input_tangents(
    spec: _FunctionSpec,
    flat_tangents: tuple[Tensor | None, ...],
) -> tuple[Tensor | None, ...]:
    return tuple(
        flat_tangents[index] for index in spec.tensor_input_indices
    )


def _function_param_tangents(
    spec: _FunctionSpec,
    flat_tangents: tuple[Tensor | None, ...],
) -> tuple[Tensor | None, ...]:
    return tuple(
        flat_tangents[value] if kind == "tensor" else None
        for kind, value in spec.param_specs
    )


def _contract_param_grad(
    grad_value: Tensor,
    derivative: Tensor,
    param: Param,
) -> Tensor | None:
    if not isinstance(param, Tensor):
        return None
    grad = grad_value * derivative
    return grad.sum_to_size(param.shape)


def _clone_function_state(
    tensors: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    # Dynamo's autograd.Function wiring requires distinct saved outputs.
    # Several adapters intentionally reuse a length/state tensor in both the
    # forward and backward pass records, so make each opaque state slot unique.
    return tuple(tensor.clone() for tensor in tensors)


def _to_backward_compute_dtype(tensor: Tensor) -> Tensor:
    """Promote floating backward/recompute inputs to the kernel dtype."""

    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(dtype=torch.float32)
    return tensor


def _cast_compute_dtype(
    tensor_inputs: tuple[Tensor, ...],
    params: tuple[Param, ...],
    dtype: torch.dtype | None,
) -> tuple[tuple[Tensor, ...], tuple[Param, ...]]:
    """Cast differentiable inputs through the FP32 accumulation escape hatch."""

    if dtype is None:
        return tensor_inputs, params
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or None")
    if dtype != torch.float32:
        raise ValueError(
            "d2p computes in float32; dtype= is an FP32-accumulation "
            "escape hatch - only torch.float32 is supported"
        )
    cast_inputs = tuple(
        value.to(dtype=dtype) if value.is_floating_point() else value
        for value in tensor_inputs
    )
    cast_params = tuple(
        value.to(dtype=dtype)
        if isinstance(value, Tensor) and value.is_floating_point()
        else value
        for value in params
    )
    return cast_inputs, cast_params


def _restore_function_grad_dtype(
    grad: Tensor | None,
    original: Tensor,
) -> Tensor | None:
    """Return a custom-Function gradient in its original input dtype."""

    if grad is None:
        return None
    return grad.to(device=original.device, dtype=original.dtype)


_ENTROPY_PRIMARY_INPUT_NAMES = {
    "sw": "pair_scores",
    "sw_affine": "pair_scores",
    "nw": "pair_scores",
    "nw_affine": "pair_scores",
    "dtw": "costs",
    "lcs": "match_scores",
    "lev": "substitution_costs",
    "osa": "substitution_costs",
    "damerau": "substitution_costs",
    "mas": "scores",
    "cky": "merge_scores",
    "eisner": "arc_scores",
}

_ENTROPY_PARAMETER_NAMES = {
    "gap": "gap_score",
    "gap_open": "gap_open_score",
    "gap_ext": "gap_extend_score",
    "ins": "insertion_cost",
    "del": "deletion_cost",
    "ins_cost": "insertion_cost",
    "del_cost": "deletion_cost",
    "trans_cost": "transposition_cost",
    "temp": "temperature",
}


def _entropy_unsupported_direction_error(
    spec: _FunctionSpec,
    direction_name: str,
) -> NotImplementedError:
    primary_name = _ENTROPY_PRIMARY_INPUT_NAMES.get(
        spec.name, spec.tensor_input_names[0]
    )
    public_direction_name = _ENTROPY_PARAMETER_NAMES.get(
        direction_name, direction_name
    )
    return NotImplementedError(
        f"{spec.name}_entropy is differentiable only w.r.t. its primary "
        f"{primary_name} input in 0.1.0; the entropy gradient w.r.t. "
        f"parameter {public_direction_name} (a second derivative involving "
        "temperature) is not available -- see DERIVATIVE_COVERAGE.md. "
        "Detach the parameter or use finite differences."
    )


def _entropy_unsupported_direction_indices(
    spec: _FunctionSpec,
) -> tuple[tuple[str, int], ...]:
    other_inputs = tuple(
        zip(
            spec.tensor_input_names[1:],
            spec.tensor_input_indices[1:],
            strict=True,
        )
    )
    tensor_params = tuple(
        (name, value)
        for name, (kind, value) in zip(
            spec.param_names, spec.param_specs, strict=True
        )
        if kind == "tensor"
    )
    return (*other_inputs, *tensor_params)


class _EntropyCompileFn(torch.autograd.Function):
    """Autograd registration for an operator's entropy observable.

    Entropy is the temperature parameter first-derivative of the value
    ``H = s * dV/dtemp`` with ``s = +1`` (score-native) or ``-1`` (cost-native).
    Its gradient w.r.t. the primary score input is the temperature
    cross-Jacobian of the marginals ``dH/d(primary) = s * dM/dtemp``, supplied
    by the existing ``marginals_grad_temp`` kernel (``param_backward`` with a
    ``None`` cotangent). No new kernel is needed; this is second-order-through-
    first-order autograd exactly as :class:`_MapFn`.

    Only the primary-input direction is differentiable. Requests for gradients
    w.r.t. another tensor input or scalar parameter raise ``NotImplementedError``
    because the corresponding second-derivative blocks are not shipped.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        spec: "_FunctionSpec",
        *flat_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        flat_tensors = tuple(flat_tensors)
        call = _restore_function_call(spec, flat_tensors)
        forward = _run_forward(spec, call)
        result = _run_backward(spec, call, forward)
        state_tensors = _clone_function_state(
            (
                forward.value,
                result.marginals,
                *result.input_grads[1:],
                *result.param_grads,
                *forward.saved_tensors,
                *result.saved_tensors,
            )
        )
        # Keep the differentiable public output distinct from the opaque
        # scalar-derivative primitive that produced it. AOTAutograd can
        # otherwise trace that primitive's PT2 mirror as the Function output
        # and materialize a zero cotangent for this custom backward.
        return (
            result.entropy.clone(),
            *state_tensors,
        )

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[Tensor, ...],
    ) -> None:
        spec, *flat_tensors = inputs
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*output[1:])
        _save_function_state(
            ctx, spec, tuple(flat_tensors), tuple(output)
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_entropy: Tensor | None,
        *unused_state_grads: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        spec = _restore_function_spec(ctx)
        for direction_name, flat_index in (
            _entropy_unsupported_direction_indices(spec)
        ):
            if ctx.needs_input_grad[_function_grad_index(flat_index)]:
                raise _entropy_unsupported_direction_error(
                    spec, direction_name
                )
        if grad_entropy is None:
            return (None, *(None for _ in range(ctx.flat_tensor_count)))
        primary_flat_index = spec.tensor_input_indices[0]
        if not ctx.needs_input_grad[_function_grad_index(primary_flat_index)]:
            return (None, *(None for _ in range(ctx.flat_tensor_count)))
        original_flat_tensors = tuple(
            ctx.saved_tensors[: ctx.flat_tensor_count]
        )
        native_flat_tensors = tuple(
            _to_backward_compute_dtype(
                unwrap_grad_tracking_tensor(tensor)
            )
            for tensor in original_flat_tensors
        )
        call = _restore_function_call(spec, native_flat_tensors)
        state_tensors = tuple(
            _to_backward_compute_dtype(
                unwrap_grad_tracking_tensor(tensor)
            )
            for tensor in ctx.saved_tensors[ctx.flat_tensor_count :]
        )
        input_grad_start = 3
        input_grad_end = (
            input_grad_start + len(spec.tensor_input_names) - 1
        )
        param_start = input_grad_end
        param_end = param_start + len(spec.param_names)
        state = BackwardPass(
            marginals=state_tensors[2],
            entropy=state_tensors[0],
            param_grads=tuple(state_tensors[param_start:param_end]),
            saved_tensors=tuple(state_tensors[param_end:]),
            input_grads=(
                state_tensors[2],
                *state_tensors[input_grad_start:input_grad_end],
            ),
        )
        # raw dM/dtemp (None cotangent => raw sensitivity field), shape = primary map shape
        d_marginals_d_temp = _FUNCTION_KERNELS[spec.name].param_backward(
            call, state, "temp", None
        )
        sign = -1.0 if spec.name in _COST_NATIVE else 1.0
        grad_primary = (
            sign
            * _expand_batch_grad(
                _to_backward_compute_dtype(grad_entropy),
                d_marginals_d_temp,
            )
            * d_marginals_d_temp
        )
        grad_primary = _restore_function_grad_dtype(
            grad_primary, original_flat_tensors[primary_flat_index]
        )
        flat_grads: list[Tensor | None] = [
            None for _ in range(ctx.flat_tensor_count)
        ]
        flat_grads[primary_flat_index] = grad_primary
        return (None, *flat_grads)


class _EntropyFn(torch.autograd.Function):
    """Forward-mode-enabled entropy Function used only by genuine JVP."""

    generate_vmap_rule = True
    forward = staticmethod(_EntropyCompileFn.forward)
    backward = staticmethod(_EntropyCompileFn.backward)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[Tensor, ...],
    ) -> None:
        spec, *flat_tensors = inputs
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*output[1:])
        _save_function_state(
            ctx,
            spec,
            tuple(flat_tensors),
            tuple(output),
            save_for_forward=True,
        )

    @staticmethod
    def jvp(
        ctx: Any,
        unused_spec_tangent: None,
        *flat_tangents: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        del unused_spec_tangent
        spec, call, state_tensors = _restore_function_state(
            ctx,
            transform_safe=True,
        )
        input_grad_start = 3
        input_grad_end = (
            input_grad_start + len(spec.tensor_input_names) - 1
        )
        param_start = input_grad_end
        param_end = param_start + len(spec.param_names)
        state = BackwardPass(
            marginals=state_tensors[2],
            entropy=state_tensors[0],
            param_grads=tuple(state_tensors[param_start:param_end]),
            saved_tensors=tuple(state_tensors[param_end:]),
            input_grads=(
                state_tensors[2],
                *state_tensors[input_grad_start:input_grad_end],
            ),
        )
        primary_index = spec.tensor_input_indices[0]
        primary_tangent = flat_tangents[primary_index]
        for direction_name, flat_index in (
            _entropy_unsupported_direction_indices(spec)
        ):
            if flat_tangents[flat_index] is not None:
                raise _entropy_unsupported_direction_error(
                    spec, direction_name
                )
        if primary_tangent is None:
            tangent_entropy = state.entropy.new_zeros(
                state.entropy.shape
            )
            return (
                tangent_entropy,
                *(None for _ in state_tensors[1:]),
            )
        primary_tangent = _to_backward_compute_dtype(
            unwrap_grad_tracking_tensor(primary_tangent)
        )
        d_marginals_d_temp = _FUNCTION_KERNELS[
            spec.name
        ].param_backward(call, state, "temp", None)
        tangent = _contract_tensor_direction(
            d_marginals_d_temp,
            primary_tangent,
            state.entropy,
        )
        sign = -1.0 if spec.name in _COST_NATIVE else 1.0
        tangent_entropy = (sign * tangent).to(
            dtype=state.entropy.dtype
        )
        return (
            tangent_entropy,
            *(None for _ in state_tensors[1:]),
        )


class _ValueCompileFn(torch.autograd.Function):
    """Autograd registration for an operator's scalar value."""

    # The generated Function rule composes the dispatcher rules registered for
    # each named primitive; dispatcher rules alone do not bypass this layer.
    generate_vmap_rule = True

    @staticmethod
    def forward(
        spec: _FunctionSpec,
        *flat_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        flat_tensors = tuple(flat_tensors)
        call = _restore_function_call(spec, flat_tensors)
        result = _run_forward(spec, call)
        return (
            result.value,
            *_clone_function_state(result.saved_tensors),
        )

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[Tensor, ...],
    ) -> None:
        spec, *flat_tensors = inputs
        ctx.set_materialize_grads(False)
        if len(output) > 1:
            ctx.mark_non_differentiable(*output[1:])
        _save_function_state(
            ctx, spec, tuple(flat_tensors), tuple(output)
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_value: Tensor | None,
        *unused_state_grads: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        spec = _restore_function_spec(ctx)
        input_indices = tuple(
            _function_grad_index(index)
            for index in spec.tensor_input_indices
        )
        param_indices = _function_param_grad_indices(spec)
        differentiable_indices = (
            *input_indices,
            *(index for index in param_indices if index is not None),
        )
        if grad_value is None or not any(
            ctx.needs_input_grad[index]
            for index in differentiable_indices
        ):
            return (
                None,
                *(None for _ in range(ctx.flat_tensor_count)),
            )

        saved_tensors = ctx.saved_tensors
        original_flat_tensors = tuple(
            saved_tensors[: ctx.flat_tensor_count]
        )
        backward_flat_tensors = tuple(
            _to_backward_compute_dtype(
                unwrap_grad_tracking_tensor(tensor)
            )
            for tensor in original_flat_tensors
        )
        state_tensors = tuple(
            _to_backward_compute_dtype(
                unwrap_grad_tracking_tensor(tensor)
            )
            for tensor in saved_tensors[ctx.flat_tensor_count :]
        )
        call = _restore_function_call(
            spec, backward_flat_tensors
        )
        grad_value = _to_backward_compute_dtype(
            unwrap_grad_tracking_tensor(grad_value)
        )
        forward = ForwardPass(
            value=state_tensors[0],
            saved_tensors=tuple(state_tensors[1:]),
        )
        needs_input = tuple(
            ctx.needs_input_grad[index] for index in input_indices
        )
        needs_param = tuple(
            index is not None and ctx.needs_input_grad[index]
            for index in param_indices
        )
        forward_has_primary_grad = len(forward.saved_tensors) > 0
        needs_backward_pass = (
            any(needs_param)
            or any(needs_input[1:])
            or (needs_input[0] and not forward_has_primary_grad)
        )
        result = (
            _run_backward(spec, call, forward)
            if needs_backward_pass
            else None
        )

        grad_inputs: list[Tensor | None] = []
        for index, needed in enumerate(needs_input):
            if needed:
                if result is not None:
                    derivative = result.input_grads[index]
                elif index == 0 and forward_has_primary_grad:
                    derivative = forward.saved_tensors[0]
                else:
                    raise RuntimeError(
                        f"{spec.name} score backward is missing "
                        "the gradient for "
                        f"{spec.tensor_input_names[index]}"
                    )
                grad_inputs.append(
                    derivative
                    * _expand_batch_grad(grad_value, derivative)
                )
            else:
                grad_inputs.append(None)

        grad_params: list[Tensor | None] = []
        for index, (param, needed) in enumerate(
            zip(call.params, needs_param, strict=True)
        ):
            if needed:
                if result is None:
                    raise RuntimeError(
                        f"{spec.name} score backward is missing "
                        f"the gradient for {spec.param_names[index]}"
                    )
                grad_params.append(
                    _contract_param_grad(
                        grad_value,
                        result.param_grads[index],
                        param,
                    )
                )
            else:
                grad_params.append(None)

        flat_grads: list[Tensor | None] = [
            None for _ in range(ctx.flat_tensor_count)
        ]
        for tensor_index, grad_input in zip(
            spec.tensor_input_indices, grad_inputs, strict=True
        ):
            flat_grads[tensor_index] = _restore_function_grad_dtype(
                grad_input, original_flat_tensors[tensor_index]
            )
        for param_spec, grad_param in zip(
            spec.param_specs, grad_params, strict=True
        ):
            kind, value = param_spec
            if kind == "tensor":
                flat_grads[value] = _restore_function_grad_dtype(
                    grad_param, original_flat_tensors[value]
                )
        return (None, *flat_grads)


class _ValueFn(torch.autograd.Function):
    """Forward-mode-enabled value Function used only by genuine JVP."""

    generate_vmap_rule = True
    forward = staticmethod(_ValueCompileFn.forward)
    backward = staticmethod(_ValueCompileFn.backward)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[Tensor, ...],
    ) -> None:
        spec, *flat_tensors = inputs
        ctx.set_materialize_grads(False)
        if len(output) > 1:
            ctx.mark_non_differentiable(*output[1:])
        _save_function_state(
            ctx,
            spec,
            tuple(flat_tensors),
            tuple(output),
            save_for_forward=True,
        )

    @staticmethod
    def jvp(
        ctx: Any,
        unused_spec_tangent: None,
        *flat_tangents: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        del unused_spec_tangent
        spec, call, state_tensors = _restore_function_state(
            ctx,
            transform_safe=True,
        )
        flat_tangents = tuple(
            unwrap_grad_tracking_tensor(tangent)
            if tangent is not None
            else None
            for tangent in flat_tangents
        )
        input_tangents = _function_input_tangents(
            spec, flat_tangents
        )
        param_tangents = _function_param_tangents(
            spec, flat_tangents
        )
        forward = ForwardPass(
            value=state_tensors[0],
            saved_tensors=tuple(state_tensors[1:]),
        )
        needs_input = tuple(
            tangent is not None for tangent in input_tangents
        )
        needs_param = tuple(
            tangent is not None for tangent in param_tangents
        )
        forward_has_primary_grad = len(forward.saved_tensors) > 0
        needs_backward_pass = (
            any(needs_param)
            or any(needs_input[1:])
            or (needs_input[0] and not forward_has_primary_grad)
        )
        backward = (
            _run_backward(spec, call, forward)
            if needs_backward_pass
            else None
        )

        tangent_value: Tensor | None = None
        for index, tangent in enumerate(input_tangents):
            if tangent is None:
                continue
            if backward is not None:
                derivative = backward.input_grads[index]
            elif index == 0 and forward_has_primary_grad:
                derivative = forward.saved_tensors[0]
            else:
                raise RuntimeError(
                    f"{spec.name} value JVP is missing the gradient for "
                    f"{spec.tensor_input_names[index]}"
                )
            tangent_value = _accumulate_direction(
                tangent_value,
                _contract_tensor_direction(
                    derivative,
                    tangent,
                    forward.value,
                ),
            )

        for index, tangent in enumerate(param_tangents):
            if tangent is None:
                continue
            if backward is None:
                raise RuntimeError(
                    f"{spec.name} value JVP is missing the gradient for "
                    f"{spec.param_names[index]}"
                )
            tangent_value = _accumulate_direction(
                tangent_value,
                _contract_tensor_direction(
                    backward.param_grads[index],
                    tangent,
                    forward.value,
                ),
            )

        if tangent_value is None:
            tangent_value = forward.value.new_zeros(forward.value.shape)
        tangent_value = tangent_value.to(dtype=forward.value.dtype)
        return (
            tangent_value,
            *(None for _ in state_tensors[1:]),
        )


class _MapCompileFn(torch.autograd.Function):
    """Autograd registration for an operator's marginals map."""

    # See _ValueFn: this enables Function-level composition of the real
    # dispatcher batching rules instead of advertising a kernel fallback.
    generate_vmap_rule = True

    @staticmethod
    def forward(
        spec: _FunctionSpec,
        *flat_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        flat_tensors = tuple(flat_tensors)
        call = _restore_function_call(spec, flat_tensors)
        forward = _run_forward(spec, call)
        result = _run_backward(spec, call, forward)
        state_tensors = _clone_function_state(
            (
                forward.value,
                result.entropy,
                *result.input_grads[1:],
                *result.param_grads,
                *forward.saved_tensors,
                *result.saved_tensors,
            )
        )
        return (
            result.marginals,
            *state_tensors,
        )

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[Tensor, ...],
    ) -> None:
        spec, *flat_tensors = inputs
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*output[1:])
        _save_function_state(
            ctx, spec, tuple(flat_tensors), tuple(output)
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_map: Tensor | None,
        *unused_state_grads: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        spec = _restore_function_spec(ctx)
        input_indices = tuple(
            _function_grad_index(index)
            for index in spec.tensor_input_indices
        )
        param_indices = _function_param_grad_indices(spec)
        differentiable_indices = (
            *input_indices,
            *(index for index in param_indices if index is not None),
        )
        if grad_map is None or not any(
            ctx.needs_input_grad[index]
            for index in differentiable_indices
        ):
            return (
                None,
                *(None for _ in range(ctx.flat_tensor_count)),
            )

        saved_tensors = ctx.saved_tensors
        original_flat_tensors = tuple(
            saved_tensors[: ctx.flat_tensor_count]
        )
        map_backward_flat_tensors = tuple(
            _to_backward_compute_dtype(tensor)
            for tensor in original_flat_tensors
        )
        state_tensors = tuple(
            _to_backward_compute_dtype(tensor)
            for tensor in saved_tensors[ctx.flat_tensor_count :]
        )
        spec = _restore_function_spec(ctx)
        call = _restore_function_call(
            spec, map_backward_flat_tensors
        )
        grad_map = _to_backward_compute_dtype(grad_map)
        input_grad_start = 3
        input_grad_end = (
            input_grad_start + len(spec.tensor_input_names) - 1
        )
        param_start = input_grad_end
        param_end = param_start + len(spec.param_names)
        state = BackwardPass(
            marginals=state_tensors[0],
            entropy=state_tensors[2],
            param_grads=tuple(state_tensors[param_start:param_end]),
            saved_tensors=tuple(state_tensors[param_end:]),
            input_grads=(
                state_tensors[0],
                *state_tensors[input_grad_start:input_grad_end],
            ),
        )

        grad_inputs: list[Tensor | None] = [
            None for _ in spec.tensor_input_names
        ]
        if any(ctx.needs_input_grad[index] for index in input_indices):
            all_grad_inputs = _run_map_backward(spec, call, state, grad_map)
            for index, grad_input in enumerate(all_grad_inputs):
                if ctx.needs_input_grad[input_indices[index]]:
                    grad_inputs[index] = grad_input

        grad_params: list[Tensor | None] = []
        for index, param_name in enumerate(spec.param_names):
            param_index = param_indices[index]
            if (
                param_index is not None
                and ctx.needs_input_grad[param_index]
            ):
                grad_params.append(
                    _FUNCTION_KERNELS[spec.name].param_backward(
                        call, state, param_name, grad_map
                    )
                )
            else:
                grad_params.append(None)

        flat_grads: list[Tensor | None] = [
            None for _ in range(ctx.flat_tensor_count)
        ]
        for tensor_index, grad_input in zip(
            spec.tensor_input_indices, grad_inputs, strict=True
        ):
            flat_grads[tensor_index] = _restore_function_grad_dtype(
                grad_input, original_flat_tensors[tensor_index]
            )
        for param_spec, grad_param in zip(
            spec.param_specs, grad_params, strict=True
        ):
            kind, value = param_spec
            if kind == "tensor":
                flat_grads[value] = _restore_function_grad_dtype(
                    grad_param, original_flat_tensors[value]
                )
        return (None, *flat_grads)


class _MapFn(torch.autograd.Function):
    """Forward-mode-enabled map Function used only by genuine JVP."""

    generate_vmap_rule = True
    forward = staticmethod(_MapCompileFn.forward)
    backward = staticmethod(_MapCompileFn.backward)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[Tensor, ...],
    ) -> None:
        spec, *flat_tensors = inputs
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*output[1:])
        _save_function_state(
            ctx,
            spec,
            tuple(flat_tensors),
            tuple(output),
            save_for_forward=True,
        )

    @staticmethod
    def jvp(
        ctx: Any,
        unused_spec_tangent: None,
        *flat_tangents: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        del unused_spec_tangent
        spec, call, state_tensors = _restore_function_state(
            ctx,
            transform_safe=True,
        )
        call = _restore_function_call(
            spec,
            tuple(
                _to_backward_compute_dtype(
                    unwrap_grad_tracking_tensor(tensor)
                )
                for tensor in ctx.saved_tensors[: ctx.flat_tensor_count]
            ),
        )
        state_tensors = tuple(
            _to_backward_compute_dtype(tensor)
            for tensor in state_tensors
        )
        flat_tangents = tuple(
            _to_backward_compute_dtype(
                unwrap_grad_tracking_tensor(tangent)
            )
            if tangent is not None
            else None
            for tangent in flat_tangents
        )
        input_grad_start = 3
        input_grad_end = (
            input_grad_start + len(spec.tensor_input_names) - 1
        )
        param_start = input_grad_end
        param_end = param_start + len(spec.param_names)
        state = BackwardPass(
            marginals=state_tensors[0],
            entropy=state_tensors[2],
            param_grads=tuple(state_tensors[param_start:param_end]),
            saved_tensors=tuple(state_tensors[param_end:]),
            input_grads=(
                state_tensors[0],
                *state_tensors[input_grad_start:input_grad_end],
            ),
        )
        input_tangents = _function_input_tangents(
            spec, flat_tangents
        )
        tangent_map: Tensor | None = None
        if any(tangent is not None for tangent in input_tangents):
            tangent_map = _run_map_jvp(
                spec,
                call,
                state,
                input_tangents,
            )

        param_tangents = _function_param_tangents(
            spec, flat_tangents
        )
        for param_name, tangent in zip(
            spec.param_names,
            param_tangents,
            strict=True,
        ):
            if tangent is None:
                continue
            sensitivity = _FUNCTION_KERNELS[
                spec.name
            ].param_backward(call, state, param_name, None)
            tangent_map = _accumulate_direction(
                tangent_map,
                _contract_tensor_direction(
                    sensitivity,
                    tangent,
                    state.marginals,
                ),
            )

        if tangent_map is None:
            tangent_map = state.marginals.new_zeros(
                state.marginals.shape
            )
        tangent_map = tangent_map.to(dtype=state.marginals.dtype)
        return (
            tangent_map,
            *(None for _ in state_tensors[1:]),
        )


class Operator:
    """Private autograd engine used by explicit per-operator functions."""

    def __init__(
        self,
        name: str,
        params: Sequence[str],
        *,
        kernels: OperatorKernels | Mapping[str, Any],
        tensor_inputs: Sequence[str] = ("scores",),
        tensor_shapes: Sequence[str],
        length_axes: Sequence[int] | None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        param_names = tuple(params)
        if not all(isinstance(param, str) and param for param in param_names):
            raise ValueError("params must contain only non-empty strings")
        if len(set(param_names)) != len(param_names):
            raise ValueError("params must not contain duplicates")
        tensor_input_names = tuple(tensor_inputs)
        if not tensor_input_names:
            raise ValueError("tensor_inputs must contain at least one name")
        if not all(
            isinstance(input_name, str) and input_name
            for input_name in tensor_input_names
        ):
            raise ValueError(
                "tensor_inputs must contain only non-empty strings"
            )
        if len(set(tensor_input_names)) != len(tensor_input_names):
            raise ValueError("tensor_inputs must not contain duplicates")
        tensor_shape_specs = tuple(
            tuple(part.strip() for part in shape.split(","))
            for shape in tensor_shapes
        )
        if len(tensor_shape_specs) != len(tensor_input_names):
            raise ValueError(
                "tensor_shapes must provide one shape for each tensor input"
            )
        if not all(spec and all(spec) for spec in tensor_shape_specs):
            raise ValueError("tensor_shapes must contain named dimensions")
        normalized_length_axes = (
            None if length_axes is None else tuple(length_axes)
        )
        if normalized_length_axes is not None:
            primary_rank = len(tensor_shape_specs[0])
            if not normalized_length_axes:
                raise ValueError("length_axes must not be empty")
            if any(
                not isinstance(axis, int)
                or axis >= primary_rank
                or axis < -primary_rank
                for axis in normalized_length_axes
            ):
                raise ValueError(
                    "length_axes contains an invalid primary-input axis"
                )
        overlap = set(param_names) & set(tensor_input_names)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"tensor input and parameter names must be disjoint: {names}"
            )

        self.name = name
        self._params = param_names
        self._tensor_inputs = tensor_input_names
        self._tensor_shapes = tensor_shape_specs
        self._length_axes = normalized_length_axes
        self._kernels = _normalize_kernels(kernels)
        _FUNCTION_KERNELS[self.name] = self._kernels

    @property
    def params(self) -> tuple[str, ...]:
        """Names of the operator's positional scoring parameters."""

        return self._params

    @property
    def tensor_inputs(self) -> tuple[str, ...]:
        """Names of the operator's positional tensor-valued inputs."""

        return self._tensor_inputs

    def _check_call(
        self,
        tensor_inputs: tuple[Tensor, ...],
        params: tuple[Param, ...],
    ) -> None:
        if len(tensor_inputs) != len(self.tensor_inputs):
            raise TypeError(
                f"{self.name} expected {len(self.tensor_inputs)} tensor inputs "
                f"({', '.join(self.tensor_inputs)}), got {len(tensor_inputs)}"
            )
        for name, value in zip(
            self.tensor_inputs, tensor_inputs, strict=True
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
        if len(params) != len(self.params):
            raise TypeError(
                f"{self.name} expected {len(self.params)} parameters "
                f"({', '.join(self.params)}), got {len(params)}"
            )
        if not all(
            isinstance(param, (float, int, Tensor))
            and not isinstance(param, bool)
            for param in params
        ):
            raise TypeError("scoring parameters must be real numbers or tensors")

    def _validate_call_contract(
        self,
        tensor_inputs: tuple[Tensor, ...],
        params: tuple[Param, ...],
        lengths: Tensor | None,
    ) -> None:
        """Enforce the shared eager/transform/compile input contract."""

        dimensions: dict[str, tuple[int, str]] = {}
        primary = tensor_inputs[0]
        for name, value, shape_spec in zip(
            self.tensor_inputs,
            tensor_inputs,
            self._tensor_shapes,
            strict=True,
        ):
            expected = "[" + ", ".join(shape_spec) + "]"
            if value.ndim != len(shape_spec):
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(value.shape)}"
                )
            if value.shape[0] == 0:
                raise ValueError(
                    f"{self.name} does not support an empty leading batch (B=0)"
                )
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.device != primary.device:
                raise ValueError(
                    f"{name} must be on the same device as {self.tensor_inputs[0]}, "
                    f"got {value.device} and {primary.device}"
                )
            if value.dtype != torch.float32:
                raise TypeError(
                    f"{name} must have dtype torch.float32 outside autocast; "
                    f"got {value.dtype}"
                )
            for axis, dimension_name in enumerate(shape_spec):
                size = value.shape[axis]
                previous = dimensions.get(dimension_name)
                if previous is not None and size != previous[0]:
                    raise ValueError(
                        f"{name} dimension {dimension_name} must equal "
                        f"{previous[1]} dimension {dimension_name} "
                        f"({previous[0]}), got {size}"
                    )
                dimensions.setdefault(dimension_name, (size, name))

        for name, value in zip(self.params, params, strict=True):
            _validate_scalar_parameter_shape(value, name)
            if not isinstance(value, Tensor):
                continue
            if not value.is_floating_point():
                raise TypeError(
                    f"{name} tensor must have a floating-point dtype"
                )
            if value.device != primary.device:
                raise ValueError(
                    f"{name} tensor must be on the same device as "
                    f"{self.tensor_inputs[0]}, got {value.device} and "
                    f"{primary.device}"
                )
            if value.dtype != torch.float32:
                raise TypeError(
                    f"{name} tensor must have dtype torch.float32 outside "
                    "autocast; "
                    f"got {value.dtype}"
                )

        self._validate_lengths_contract(primary, lengths)

    def _validate_lengths_contract(
        self,
        primary: Tensor,
        lengths: Tensor | None,
    ) -> Tensor | None:
        """Validate and return the discrete active-length tensor."""

        if lengths is None:
            return None
        if self._length_axes is None:
            raise TypeError(f"{self.name} does not accept lengths")
        if not isinstance(lengths, Tensor):
            raise TypeError("lengths must be a tensor or None")
        expected_shape = (
            (primary.shape[0],)
            if len(self._length_axes) == 1
            else (primary.shape[0], len(self._length_axes))
        )
        if tuple(lengths.shape) != expected_shape:
            rendered = (
                "[B]"
                if len(self._length_axes) == 1
                else f"[B, {len(self._length_axes)}]"
            )
            raise ValueError(
                f"lengths must have shape {rendered}, got {tuple(lengths.shape)}"
            )
        if lengths.dtype != torch.int32:
            raise TypeError(
                f"lengths must have dtype torch.int32, got {lengths.dtype}"
            )
        if lengths.device != primary.device:
            raise ValueError(
                "lengths must be on the same device as the primary input, got "
                f"{lengths.device} and {primary.device}"
            )
        if not lengths.is_contiguous():
            raise ValueError("lengths must be contiguous")
        if lengths.ndim == 1:
            valid_lengths = (lengths >= 0) & (
                lengths <= primary.shape[self._length_axes[0]]
            )
        else:
            valid_lengths = torch.stack(
                tuple(
                    (lengths[:, index] >= 0)
                    & (lengths[:, index] <= primary.shape[axis])
                    for index, axis in enumerate(self._length_axes)
                ),
                dim=-1,
            )
        _graph_safe_assert(
            valid_lengths,
            "lengths entries must lie within the padded input shape",
        )
        return lengths

    def _split_call_inputs(
        self,
        primary: Tensor,
        inputs_and_params: tuple[Tensor | Param, ...],
        config: Mapping[str, Any],
        dtype: torch.dtype | None,
        lengths: Tensor | None,
    ) -> tuple[
        tuple[Tensor, ...],
        tuple[Param, ...],
        dict[str, Any],
    ]:
        extra_input_count = len(self.tensor_inputs) - 1
        tensor_inputs = (
            primary,
            *inputs_and_params[:extra_input_count],
        )
        params = tuple(inputs_and_params[extra_input_count:])
        normalized_config = dict(config)

        # The shipped DTW adapter uses zero for an unrestricted band.
        # Frozen v3 names that public convention ``None``.
        if (
            self.name == "dtw"
            and normalized_config.get("bandwidth", 0) is None
        ):
            normalized_config["bandwidth"] = 0

        self._check_call(tensor_inputs, params)
        if dtype is None and torch.is_autocast_enabled(primary.device.type):
            dtype = torch.float32
        tensor_inputs, params = _cast_compute_dtype(
            tensor_inputs,
            params,
            dtype,
        )
        self._validate_call_contract(tensor_inputs, params, lengths)
        _validate_numerical_domain(
            self.tensor_inputs,
            tensor_inputs,
            self.params,
            params,
            cost_native=self.name in _COST_NATIVE,
        )
        tensor_inputs = _normalize_masked_tensor_inputs(
            tensor_inputs,
            cost_native=self.name in _COST_NATIVE,
            temperature=params[self.params.index("temp")],
        )
        return tensor_inputs, params, normalized_config

    def _config_items(self, config: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        unknown = set(config) - set(self._kernels.config)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"{self.name} got unexpected configuration: {names}")
        return tuple(
            (
                key,
                config[key] if key in config else default,
            )
            for key, default in self._kernels.config.items()
        )

    def _run_forward(self, call: KernelCall) -> ForwardPass:
        return _normalize_forward(self._kernels.forward(call))

    def _run_backward(
        self,
        call: KernelCall,
        forward: ForwardPass,
    ) -> BackwardPass:
        return _normalize_backward(
            self._kernels.backward(call, forward),
            len(self.params),
            len(self.tensor_inputs),
        )

    def _run_map_backward(
        self,
        call: KernelCall,
        state: ForwardPass | BackwardPass,
        grad_map: Tensor,
    ) -> tuple[Tensor, ...]:
        return _normalize_map_backward(
            self._kernels.map_backward(call, state, grad_map),
            self.tensor_inputs,
        )

    def _prepare(
        self,
        tensor_inputs: tuple[Tensor, ...],
        params: tuple[Param, ...],
        lengths: Tensor | None,
        config: Mapping[str, Any],
    ) -> tuple[KernelCall, ForwardPass]:
        self._check_call(tensor_inputs, params)
        call = _make_call(
            tensor_inputs, params, lengths, self._config_items(config)
        )
        return call, self._run_forward(call)

    def _mask_primary(self, primary: Tensor, mask: Tensor | None) -> Tensor:
        """Exclude cells flagged ``True`` in ``mask`` from the alignment/parse.

        Callers mark which cells to exclude; the orientation-correct infinity is
        applied internally (``-inf`` for score-native soft-max operators, ``+inf``
        for cost-native soft-min operators) and normalized to a finite sentinel
        before dispatch, so callers never handle infinities.
        """
        if not isinstance(primary, Tensor):
            raise TypeError(f"{self.tensor_inputs[0]} must be a tensor")
        if mask is None:
            return primary
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            got = mask.dtype if isinstance(mask, Tensor) else type(mask).__name__
            raise TypeError(
                f"mask must be a boolean tensor flagging excluded cells; got {got}"
            )
        if tuple(mask.shape) != tuple(primary.shape):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match "
                f"{self.tensor_inputs[0]} shape {tuple(primary.shape)}"
            )
        if mask.device != primary.device:
            raise ValueError(
                f"mask must be on the same device as {self.tensor_inputs[0]}, "
                f"got {mask.device} and {primary.device}"
            )
        fill = float("inf") if self.name in _COST_NATIVE else float("-inf")
        return primary.masked_fill(mask, fill)

    def __call__(
        self,
        primary: Tensor,
        *inputs_and_params: Tensor | Param,
        lengths: Tensor | None = None,
        dtype: torch.dtype | None = None,
        mask: Tensor | None = None,
        **config: Any,
    ) -> Tensor:
        """Return the marginals map."""

        primary = self._mask_primary(primary, mask)
        tensor_inputs, params, config = self._split_call_inputs(
            primary, tuple(inputs_and_params), config, dtype, lengths
        )
        spec, flat_tensors = _flatten_function_inputs(
            name=self.name,
            tensor_input_names=self.tensor_inputs,
            param_names=self.params,
            kernels=self._kernels,
            tensor_inputs=tensor_inputs,
            params=params,
            lengths=lengths,
            config_items=self._config_items(config),
        )
        if not _function_needs_transform_boundary(spec, flat_tensors):
            call = _restore_function_call(spec, flat_tensors)
            forward = _run_forward(spec, call)
            return _run_backward(spec, call, forward).marginals
        if _use_jvp_function():
            outputs = _MapFn.apply(spec, *flat_tensors)
        else:
            outputs = _MapCompileFn.apply(spec, *flat_tensors)
        return outputs[0]

    def value(
        self,
        primary: Tensor,
        *inputs_and_params: Tensor | Param,
        lengths: Tensor | None = None,
        dtype: torch.dtype | None = None,
        mask: Tensor | None = None,
        **config: Any,
    ) -> Tensor:
        """Return the per-batch structured value."""

        primary = self._mask_primary(primary, mask)
        tensor_inputs, params, config = self._split_call_inputs(
            primary, tuple(inputs_and_params), config, dtype, lengths
        )
        spec, flat_tensors = _flatten_function_inputs(
            name=self.name,
            tensor_input_names=self.tensor_inputs,
            param_names=self.params,
            kernels=self._kernels,
            tensor_inputs=tensor_inputs,
            params=params,
            lengths=lengths,
            config_items=self._config_items(config),
        )
        if not _function_needs_transform_boundary(spec, flat_tensors):
            call = _restore_function_call(spec, flat_tensors)
            return _run_forward(spec, call).value
        if _use_jvp_function():
            outputs = _ValueFn.apply(spec, *flat_tensors)
        else:
            outputs = _ValueCompileFn.apply(spec, *flat_tensors)
        return outputs[0]

    def entropy(
        self,
        primary: Tensor,
        *inputs_and_params: Tensor | Param,
        lengths: Tensor | None = None,
        dtype: torch.dtype | None = None,
        mask: Tensor | None = None,
        **config: Any,
    ) -> Tensor:
        """Return the entropy observable."""

        primary = self._mask_primary(primary, mask)
        tensor_inputs, params, config = self._split_call_inputs(
            primary, tuple(inputs_and_params), config, dtype, lengths
        )
        spec, flat_tensors = _flatten_function_inputs(
            name=self.name,
            tensor_input_names=self.tensor_inputs,
            param_names=self.params,
            kernels=self._kernels,
            tensor_inputs=tensor_inputs,
            params=params,
            lengths=lengths,
            config_items=self._config_items(config),
        )
        if not _function_needs_transform_boundary(spec, flat_tensors):
            call = _restore_function_call(spec, flat_tensors)
            forward = _run_forward(spec, call)
            return _run_backward(spec, call, forward).entropy
        if _use_jvp_function():
            outputs = _EntropyFn.apply(spec, *flat_tensors)
        else:
            outputs = _EntropyCompileFn.apply(spec, *flat_tensors)
        return outputs[0]

    def _vjp_one(
        self,
        primary: Tensor,
        *inputs_and_params: Tensor | Param,
        grad_map: Tensor,
        wrt: str,
        lengths: Tensor | None = None,
        dtype: torch.dtype | None = None,
        **config: Any,
    ) -> Tensor:
        """Contract one selected map derivative with ``grad_map``."""

        valid = (*self.tensor_inputs, *self.params)
        if not isinstance(wrt, str):
            raise TypeError("wrt must be a string")
        if wrt not in valid:
            raise ValueError(
                f"invalid wrt field {wrt!r}; expected one of {valid!r}"
            )
        if not isinstance(grad_map, Tensor):
            raise TypeError("grad_map (cotangent) must be a tensor")
        if grad_map.shape != primary.shape:
            raise ValueError(
                f"grad_map (cotangent) shape {tuple(grad_map.shape)} must match "
                f"the {self.tensor_inputs[0]} map shape {tuple(primary.shape)}"
            )
        if grad_map.device != primary.device:
            raise ValueError(
                "grad_map (cotangent) must be on the same device as the map, "
                f"got {grad_map.device} and {primary.device}"
            )
        if grad_map.dtype != torch.float32:
            raise TypeError(
                "grad_map (cotangent) must have dtype torch.float32, "
                f"got {grad_map.dtype}"
            )
        if not grad_map.is_contiguous():
            raise ValueError("grad_map (cotangent) must be contiguous")
        tensor_inputs, params, config = self._split_call_inputs(
            primary, tuple(inputs_and_params), config, dtype, lengths
        )
        call, state = self._prepare(
            tensor_inputs, params, lengths, config
        )
        if wrt in self.tensor_inputs:
            input_values = self._run_map_backward(call, state, grad_map)
            return input_values[self.tensor_inputs.index(wrt)]
        return self._kernels.param_backward(
            call,
            state,
            wrt,
            grad_map,
        )


__all__ = [
    "BackwardPass",
    "ForwardPass",
    "KernelCall",
    "Operator",
    "OperatorKernels",
]
