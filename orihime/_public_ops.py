# SPDX-License-Identifier: Apache-2.0
"""Finite public operation objects built on Orihime's kernel adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, Signature, signature
from types import MappingProxyType, ModuleType
from typing import Any

import torch
from torch import Tensor

from .operator import Operator


_FORWARD_FIELDS = ("map", "value", "entropy")


@dataclass(frozen=True)
class _PreparedCall:
    primary: Tensor
    inputs_and_params: tuple[Tensor | float | int, ...]
    lengths: Tensor | None
    dtype: torch.dtype | None
    mask: Tensor | None
    config: Mapping[str, Any]
    field_values: Mapping[str, Any]


def _selected_fields(
    output: str | Sequence[str] | None,
    allowed: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    if output is None:
        return allowed, False
    if isinstance(output, str):
        fields = (output,)
        single = True
    elif isinstance(output, Sequence):
        fields = tuple(output)
        single = False
    else:
        raise TypeError("output must be None, a field name, or a sequence of names")

    if not fields:
        raise ValueError("output must select at least one field")
    if not all(isinstance(field, str) for field in fields):
        raise TypeError("output field names must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("output must not contain duplicate field names")
    invalid = tuple(field for field in fields if field not in allowed)
    if invalid:
        raise ValueError(
            f"unknown output field {invalid[0]!r}; expected one of {allowed!r}"
        )
    return fields, single


def _selected_result(
    fields: tuple[str, ...],
    single: bool,
    evaluate: Callable[[str], Tensor],
) -> Tensor | dict[str, Tensor]:
    if single:
        return evaluate(fields[0])
    return {field: evaluate(field) for field in fields}


def _with_keyword_parameters(
    base: Signature,
    *parameters: Parameter,
) -> Signature:
    return base.replace(parameters=(*base.parameters.values(), *parameters))


class PublicOperation:
    """One algorithm's finite forward, backward, and sensitivity surface."""

    def __init__(
        self,
        *,
        name: str,
        operator: Operator,
        map_function: Callable[..., Tensor],
        value_function: Callable[..., Tensor],
        entropy_function: Callable[..., Tensor],
    ) -> None:
        self.name = name
        self._operator = operator
        self._functions = MappingProxyType(
            {
                "map": map_function,
                "value": value_function,
                "entropy": entropy_function,
            }
        )
        self._call_signature = signature(map_function)
        ordered_names = tuple(self._call_signature.parameters)
        input_count = len(operator.tensor_inputs)
        parameter_count = len(operator.params)
        self.input_fields = ordered_names[:input_count]
        self.parameter_fields = ordered_names[
            input_count : input_count + parameter_count
        ]
        self.backward_fields = (*self.input_fields, *self.parameter_fields)
        self.sensitivity_fields = self.parameter_fields
        self._internal_fields = MappingProxyType(
            dict(
                zip(
                    self.backward_fields,
                    (*operator.tensor_inputs, *operator.params),
                    strict=True,
                )
            )
        )

        def forward(
            *args: Any,
            output: str | Sequence[str] | None = None,
            **kwargs: Any,
        ) -> Tensor | dict[str, Tensor]:
            """Return selected map, value, and entropy outputs."""

            fields, single = _selected_fields(output, _FORWARD_FIELDS)
            return _selected_result(
                fields,
                single,
                lambda field: self._functions[field](*args, **kwargs),
            )

        def backward(
            *args: Any,
            grad_map: Tensor,
            output: str | Sequence[str] | None = None,
            **kwargs: Any,
        ) -> Tensor | dict[str, Tensor]:
            """Contract a map cotangent against selected inputs and parameters."""

            fields, single = _selected_fields(output, self.backward_fields)
            prepared = self._prepare(args, kwargs)

            def evaluate(field: str) -> Tensor:
                with torch.no_grad():
                    result = self._operator._vjp_one(
                        prepared.primary,
                        *prepared.inputs_and_params,
                        grad_map=grad_map,
                        wrt=self._internal_fields[field],
                        lengths=prepared.lengths,
                        dtype=prepared.dtype,
                        mask=prepared.mask,
                        **prepared.config,
                    )
                if field not in self.parameter_fields:
                    return result
                original = prepared.field_values[field]
                if isinstance(original, Tensor):
                    return result.to(
                        device=original.device,
                        dtype=original.dtype,
                    ).reshape(original.shape)
                return result.reshape(())

            return _selected_result(fields, single, evaluate)

        def sensitivity(
            *args: Any,
            output: str | Sequence[str] | None = None,
            **kwargs: Any,
        ) -> Tensor | dict[str, Tensor]:
            """Return selected map sensitivities to scalar scoring parameters."""

            fields, single = _selected_fields(output, self.sensitivity_fields)
            prepared = self._prepare(args, kwargs)

            def evaluate(field: str) -> Tensor:
                with torch.no_grad():
                    return self._operator._sensitivity_one(
                        prepared.primary,
                        *prepared.inputs_and_params,
                        wrt=self._internal_fields[field],
                        lengths=prepared.lengths,
                        dtype=prepared.dtype,
                        mask=prepared.mask,
                        **prepared.config,
                    )

            return _selected_result(fields, single, evaluate)

        forward.__name__ = "forward"
        forward.__qualname__ = f"ops.{name}.forward"
        forward.__signature__ = _with_keyword_parameters(  # type: ignore[attr-defined]
            self._call_signature,
            Parameter("output", Parameter.KEYWORD_ONLY, default=None),
        )
        backward.__name__ = "backward"
        backward.__qualname__ = f"ops.{name}.backward"
        backward.__signature__ = _with_keyword_parameters(  # type: ignore[attr-defined]
            self._call_signature,
            Parameter("grad_map", Parameter.KEYWORD_ONLY),
            Parameter("output", Parameter.KEYWORD_ONLY, default=None),
        )
        sensitivity.__name__ = "sensitivity"
        sensitivity.__qualname__ = f"ops.{name}.sensitivity"
        sensitivity.__signature__ = _with_keyword_parameters(  # type: ignore[attr-defined]
            self._call_signature,
            Parameter("output", Parameter.KEYWORD_ONLY, default=None),
        )
        self.forward = forward
        self.backward = backward
        self.sensitivity = sensitivity

    def _prepare(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> _PreparedCall:
        bound = self._call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        primary = values[self.input_fields[0]]
        if not isinstance(primary, Tensor):
            raise TypeError(f"{self.input_fields[0]} must be a tensor")

        tensor_inputs = tuple(values[field] for field in self.input_fields[1:])
        params = tuple(values[field] for field in self.parameter_fields)
        lengths = values.get("lengths")
        dtype = values.get("dtype")
        mask = values.get("mask")
        config: dict[str, Any] = {}

        if self.name == "dtw":
            module = import_module(".dtw", package=__package__)
            bandwidth = values["bandwidth"]
            module._validate_structural_inputs(lengths, bandwidth)
            config["bandwidth"] = bandwidth
        elif self.name == "osa":
            module = import_module(".osa", package=__package__)
            config["trans_mask"] = module._kernel_allowed_transpositions(
                primary,
                lengths,
                values["allowed_transpositions"],
            )
        elif self.name == "damerau":
            module = import_module(".damerau", package=__package__)
            lengths, transposition_sources = module._validate_structural_inputs(
                primary,
                lengths,
                values["transposition_sources"],
            )
            config["trans_src"] = transposition_sources

        return _PreparedCall(
            primary=primary,
            inputs_and_params=(*tensor_inputs, *params),
            lengths=lengths,
            dtype=dtype,
            mask=mask,
            config=MappingProxyType(config),
            field_values=MappingProxyType(dict(values)),
        )

    def __repr__(self) -> str:
        return f"<orihime.ops.{self.name}>"


def install_public_operations(namespace: ModuleType) -> None:
    """Populate :mod:`orihime.ops` after all high-level modules are loaded."""

    definitions = (
        ("sw", "sw", "_sw_operator", "sw"),
        ("sw_affine", "sw", "_sw_affine_operator", "sw_affine"),
        ("sv", "sv", "_sv_linear_operator", "sv"),
        ("sv_affine", "sv", "_sv_affine_operator", "sv_affine"),
        ("nw", "nw", "_nw_operator", "nw"),
        ("nw_affine", "nw", "_nw_affine_operator", "nw_affine"),
        ("dtw", "dtw", "_dtw_operator", "dtw"),
        ("lcs", "lcs", "_lcs_operator", "lcs"),
        ("lev", "edit_distance", "_lev_operator", "lev"),
        ("osa", "osa", "_osa_operator", "osa"),
        ("damerau", "damerau", "_damerau_operator", "damerau"),
        ("mas", "mas", "_mas_operator", "mas"),
        ("cky", "cky", "_cky_operator", "cky"),
        ("eisner", "eisner", "_eisner_operator", "eisner"),
    )
    operations: dict[str, PublicOperation] = {}
    kernel_modules: dict[str, ModuleType] = {}
    for public_name, module_name, operator_name, function_prefix in definitions:
        module = import_module(f".{module_name}", package=__package__)
        operations[public_name] = PublicOperation(
            name=public_name,
            operator=getattr(module, operator_name),
            map_function=getattr(module, function_prefix),
            value_function=getattr(module, f"{function_prefix}_value"),
            entropy_function=getattr(module, f"{function_prefix}_entropy"),
        )
        internal_name = "sv_linear" if public_name == "sv" else public_name
        kernel_modules[public_name] = import_module(
            f".ops.{internal_name}", package=__package__
        )

    namespace._kernels = MappingProxyType(kernel_modules)
    for name, operation in operations.items():
        setattr(namespace, name, operation)
    namespace.__dict__.pop("sv_linear", None)
    namespace.__all__ = list(operations)


__all__ = ["PublicOperation"]
