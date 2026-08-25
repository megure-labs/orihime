# SPDX-License-Identifier: Apache-2.0
"""Shared type-stable VJP machinery for the raw operator tier."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, TypeAlias

import torch
from torch import Tensor

from ..operator import _validate_derivative_vector


VJPFields: TypeAlias = tuple[str, ...]
_VJPOne = Callable[..., Tensor]


def _validate_vjp_fields(fields: VJPFields) -> VJPFields:
    if not isinstance(fields, tuple):
        raise TypeError("vjp_fields must be a tuple")
    if not fields:
        raise ValueError("vjp_fields must not be empty")
    if not all(isinstance(field, str) and field for field in fields):
        raise TypeError("vjp_fields must contain non-empty strings")
    if len(set(fields)) != len(fields):
        raise ValueError("vjp_fields must not contain duplicates")
    return fields


class _RawVJP:
    """Validate and dispatch one operator's no-graph VJP calls.

    Per-operator modules keep explicit public signatures and delegate to this
    class. ``vjp_one`` dispatches exactly one requested field; ``vjp`` returns
    a dictionary even for one field and never treats ``None`` as "all". Both
    entrypoints require a contiguous FP32 cotangent with the exact primary-map
    shape and device; invalid cotangents are rejected before native dispatch.
    """

    def __init__(
        self,
        *,
        vjp_fields: VJPFields,
        vjp_one: _VJPOne,
    ) -> None:
        if not callable(vjp_one):
            raise TypeError("vjp_one must be callable")
        self.vjp_fields = _validate_vjp_fields(vjp_fields)
        self._vjp_one = vjp_one

    def _validate_field(self, wrt: str) -> str:
        if not isinstance(wrt, str):
            raise TypeError("wrt must be a string")
        if wrt not in self.vjp_fields:
            raise ValueError(
                f"invalid wrt field {wrt!r}; expected one of "
                f"{self.vjp_fields!r}"
            )
        return wrt

    @staticmethod
    def _validate_cotangent(primary: Tensor, cotangent: Tensor) -> Tensor:
        return _validate_derivative_vector(
            primary,
            cotangent,
            name="cotangent",
            primary_name="map",
            error_type=None,
        )

    def vjp_one(
        self,
        *inputs: Any,
        wrt: str,
        cotangent: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Return one selected VJP field as a tensor.

        ``cotangent`` must already be a contiguous FP32 tensor with the exact
        primary-map shape and device. No implicit layout normalization occurs.
        """

        field = self._validate_field(wrt)
        cotangent = self._validate_cotangent(inputs[0], cotangent)
        with torch.no_grad():
            result = self._vjp_one(
                *inputs,
                wrt=field,
                cotangent=cotangent,
                **kwargs,
            )
        if not isinstance(result, Tensor):
            raise TypeError(
                "raw vjp_one implementation must return a tensor"
            )
        return result

    def vjp(
        self,
        *inputs: Any,
        wrt: VJPFields,
        cotangent: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Return selected VJP fields in a type-stable dictionary.

        ``cotangent`` must already be a contiguous FP32 tensor with the exact
        primary-map shape and device. No implicit layout normalization occurs.
        """

        if not isinstance(wrt, tuple):
            raise TypeError("wrt must be a tuple of field names")
        if len(set(wrt)) != len(wrt):
            raise ValueError("wrt must not contain duplicates")
        fields = tuple(self._validate_field(field) for field in wrt)
        cotangent = self._validate_cotangent(inputs[0], cotangent)
        return {
            field: self.vjp_one(
                *inputs,
                wrt=field,
                cotangent=cotangent,
                **kwargs,
            )
            for field in fields
        }


def _export_raw_vjp(raw: _RawVJP) -> Mapping[str, Any]:
    """Return immutable module globals for a per-operator raw wrapper."""

    if not isinstance(raw, _RawVJP):
        raise TypeError("raw must be a _RawVJP instance")
    return MappingProxyType(
        {
            "vjp_fields": raw.vjp_fields,
            "vjp_one": raw.vjp_one,
            "vjp": raw.vjp,
        }
    )


__all__ = ["VJPFields"]
