# SPDX-License-Identifier: Apache-2.0
"""Module foundation for learnable d2p operator parameters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from .cky import cky, cky_entropy, cky_value
from .damerau import damerau, damerau_entropy, damerau_value
from .dtw import dtw, dtw_entropy, dtw_value
from .edit_distance import lev, lev_entropy, lev_value
from .eisner import eisner, eisner_entropy, eisner_value
from .lcs import lcs, lcs_entropy, lcs_value
from .mas import mas, mas_entropy, mas_value
from .nw import (
    nw,
    nw_affine,
    nw_affine_entropy,
    nw_affine_value,
    nw_entropy,
    nw_value,
)
from .osa import osa, osa_entropy, osa_value
from .operator import (
    _validate_scalar_parameter_shape,
    _validate_temperature,
)
from .sw import (
    sw,
    sw_affine,
    sw_affine_entropy,
    sw_affine_value,
    sw_entropy,
    sw_value,
)


ParamValue = float | Tensor


class _OperatorModule(nn.Module):
    """Base for thin stateful wrappers around the plain v3 functions.

    Subclasses declare ``_param_names`` and pass the three operator functions
    plus a complete parameter mapping to this constructor. Selected names are
    registered as ``nn.Parameter`` objects; every other model parameter is a
    persistent buffer.
    """

    _param_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        params: Mapping[str, ParamValue],
        map_function: Callable[..., Tensor],
        value_function: Callable[..., Tensor],
        entropy_function: Callable[..., Tensor],
        learnable: Sequence[str] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not self._param_names:
            raise TypeError(
                f"{type(self).__name__} must declare _param_names"
            )
        if len(set(self._param_names)) != len(self._param_names):
            raise TypeError("_param_names must not contain duplicates")
        if set(params) != set(self._param_names):
            missing = set(self._param_names) - set(params)
            unknown = set(params) - set(self._param_names)
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise TypeError("invalid parameter mapping: " + "; ".join(details))
        for label, function in (
            ("map_function", map_function),
            ("value_function", value_function),
            ("entropy_function", entropy_function),
        ):
            if not callable(function):
                raise TypeError(f"{label} must be callable")

        if isinstance(learnable, (str, bytes)):
            raise TypeError("learnable must be a sequence of parameter names")
        learnable_names = tuple(learnable)
        if not all(
            isinstance(name, str) and name for name in learnable_names
        ):
            raise TypeError("learnable must contain non-empty strings")
        if len(set(learnable_names)) != len(learnable_names):
            raise ValueError("learnable must not contain duplicates")
        unknown_learnable = set(learnable_names) - set(self._param_names)
        if unknown_learnable:
            raise ValueError(
                "unknown learnable parameter(s): "
                + ", ".join(sorted(unknown_learnable))
            )

        object.__setattr__(self, "_map_function", map_function)
        object.__setattr__(self, "_value_function", value_function)
        object.__setattr__(self, "_entropy_function", entropy_function)

        learnable_set = set(learnable_names)
        for name in self._param_names:
            tensor = self._parameter_tensor(
                params[name],
                device=device,
                dtype=dtype,
            )
            _validate_scalar_parameter_shape(tensor, name)
            if name in learnable_set:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor, persistent=True)

        self._validate_constraints()

    @staticmethod
    def _parameter_tensor(
        value: ParamValue,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> Tensor:
        if isinstance(value, Tensor):
            if not value.is_floating_point() and not value.is_complex():
                if dtype is None:
                    dtype = torch.get_default_dtype()
            tensor = value.detach().to(device=device, dtype=dtype).clone()
        elif isinstance(value, (int, float)):
            tensor = torch.tensor(
                value,
                device=device,
                dtype=dtype or torch.get_default_dtype(),
            )
        else:
            raise TypeError("operator parameters must be numbers or tensors")
        if not tensor.is_floating_point():
            raise TypeError(
                "operator parameters must have a floating-point dtype"
            )
        return tensor

    def _validate_constraints(self) -> None:
        temperature = getattr(self, "temperature", None)
        if temperature is None:
            return
        _validate_temperature(temperature)

    def _parameter_kwargs(self) -> dict[str, Tensor]:
        self._validate_constraints()
        return {name: getattr(self, name) for name in self._param_names}

    def forward(self, *inputs: Tensor, **structural: Any) -> Tensor:
        """Return the operator map."""

        return self._map_function(
            *inputs,
            **self._parameter_kwargs(),
            **structural,
        )

    def value(self, *inputs: Tensor, **structural: Any) -> Tensor:
        """Return the operator value."""

        return self._value_function(
            *inputs,
            **self._parameter_kwargs(),
            **structural,
        )

    def entropy(self, *inputs: Tensor, **structural: Any) -> Tensor:
        """Return the operator entropy."""

        return self._entropy_function(
            *inputs,
            **self._parameter_kwargs(),
            **structural,
        )

    def extra_repr(self) -> str:
        learnable = tuple(
            name for name in self._param_names if name in self._parameters
        )
        return f"learnable={learnable!r}"


class SmithWaterman(_OperatorModule):
    """Stateful Smith-Waterman layer with per-parameter learnability."""

    _param_names = ("gap_score", "temperature")

    def __init__(
        self,
        *,
        gap_score: float | Tensor = 0.0,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "gap_score": gap_score,
                "temperature": temperature,
            },
            map_function=sw,
            value_function=sw_value,
            entropy_function=sw_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class SmithWatermanAffine(_OperatorModule):
    """Affine Smith-Waterman layer with independently learnable parameters."""

    _param_names = (
        "gap_open_score",
        "gap_extend_score",
        "temperature",
    )

    def __init__(
        self,
        *,
        gap_open_score: float | Tensor = 0.0,
        gap_extend_score: float | Tensor = 0.0,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "gap_open_score": gap_open_score,
                "gap_extend_score": gap_extend_score,
                "temperature": temperature,
            },
            map_function=sw_affine,
            value_function=sw_affine_value,
            entropy_function=sw_affine_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class NeedlemanWunsch(_OperatorModule):
    """Score-native soft Needleman-Wunsch global alignment."""

    _param_names = ("gap_score", "temperature")

    def __init__(
        self,
        *,
        gap_score: float | Tensor = 0.0,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "gap_score": gap_score,
                "temperature": temperature,
            },
            map_function=nw,
            value_function=nw_value,
            entropy_function=nw_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class NeedlemanWunschAffine(_OperatorModule):
    """Stateful affine Needleman-Wunsch layer."""

    _param_names = (
        "gap_open_score",
        "gap_extend_score",
        "temperature",
    )

    def __init__(
        self,
        *,
        gap_open_score: float | Tensor = 0.0,
        gap_extend_score: float | Tensor = 0.0,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "gap_open_score": gap_open_score,
                "gap_extend_score": gap_extend_score,
                "temperature": temperature,
            },
            map_function=nw_affine,
            value_function=nw_affine_value,
            entropy_function=nw_affine_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class DynamicTimeWarping(_OperatorModule):
    """Stateful cost-native Dynamic Time Warping layer."""

    _param_names = ("temperature",)

    def __init__(
        self,
        *,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={"temperature": temperature},
            map_function=dtw,
            value_function=dtw_value,
            entropy_function=dtw_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class LongestCommonSubsequence(_OperatorModule):
    """Stateful score-native Longest Common Subsequence layer."""

    _param_names = ("temperature",)

    def __init__(
        self,
        *,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={"temperature": temperature},
            map_function=lcs,
            value_function=lcs_value,
            entropy_function=lcs_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class Levenshtein(_OperatorModule):
    """Soft Levenshtein layer with independently learnable parameters."""

    _param_names = (
        "insertion_cost",
        "deletion_cost",
        "temperature",
    )

    def __init__(
        self,
        *,
        insertion_cost: float | Tensor = 1.0,
        deletion_cost: float | Tensor = 1.0,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "insertion_cost": insertion_cost,
                "deletion_cost": deletion_cost,
                "temperature": temperature,
            },
            map_function=lev,
            value_function=lev_value,
            entropy_function=lev_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class OptimalStringAlignment(_OperatorModule):
    """Stateful cost-native Optimal String Alignment layer."""

    _param_names = (
        "insertion_cost",
        "deletion_cost",
        "transposition_cost",
        "temperature",
    )

    def __init__(
        self,
        *,
        insertion_cost: float | Tensor = 1.0,
        deletion_cost: float | Tensor = 1.0,
        transposition_cost: float | Tensor = 1.0,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "insertion_cost": insertion_cost,
                "deletion_cost": deletion_cost,
                "transposition_cost": transposition_cost,
                "temperature": temperature,
            },
            map_function=osa,
            value_function=osa_value,
            entropy_function=osa_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class DamerauLevenshtein(_OperatorModule):
    """Damerau-Levenshtein layer with per-parameter learnability."""

    _param_names = (
        "insertion_cost",
        "deletion_cost",
        "transposition_cost",
        "temperature",
    )

    def __init__(
        self,
        *,
        insertion_cost: ParamValue = 1.0,
        deletion_cost: ParamValue = 1.0,
        transposition_cost: ParamValue = 1.0,
        temperature: ParamValue = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={
                "insertion_cost": insertion_cost,
                "deletion_cost": deletion_cost,
                "transposition_cost": transposition_cost,
                "temperature": temperature,
            },
            map_function=damerau,
            value_function=damerau_value,
            entropy_function=damerau_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class MonotonicAlignmentSearch(_OperatorModule):
    """Stateful score-native Monotonic Alignment Search layer."""

    _param_names = ("temperature",)

    def __init__(
        self,
        *,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={"temperature": temperature},
            map_function=mas,
            value_function=mas_value,
            entropy_function=mas_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class CKY(_OperatorModule):
    """CKY layer with optionally learnable temperature."""

    _param_names = ("temperature",)

    def __init__(
        self,
        *,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={"temperature": temperature},
            map_function=cky,
            value_function=cky_value,
            entropy_function=cky_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


class Eisner(_OperatorModule):
    """Eisner dependency-parsing layer with learnable temperature."""

    _param_names = ("temperature",)

    def __init__(
        self,
        *,
        temperature: float | Tensor = 1.0,
        learnable: tuple[str, ...] = (),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params={"temperature": temperature},
            map_function=eisner,
            value_function=eisner_value,
            entropy_function=eisner_entropy,
            learnable=learnable,
            device=device,
            dtype=dtype,
        )


SW = SmithWaterman
SWAffine = SmithWatermanAffine
NW = NeedlemanWunsch
NWAffine = NeedlemanWunschAffine
DTW = DynamicTimeWarping
LCS = LongestCommonSubsequence
OSA = OptimalStringAlignment
Damerau = DamerauLevenshtein
MAS = MonotonicAlignmentSearch


__all__ = [
    "SmithWaterman",
    "SW",
    "SmithWatermanAffine",
    "SWAffine",
    "NeedlemanWunsch",
    "NW",
    "NeedlemanWunschAffine",
    "NWAffine",
    "DynamicTimeWarping",
    "DTW",
    "LongestCommonSubsequence",
    "LCS",
    "Levenshtein",
    "OptimalStringAlignment",
    "OSA",
    "DamerauLevenshtein",
    "Damerau",
    "MonotonicAlignmentSearch",
    "MAS",
    "CKY",
    "Eisner",
]
