# SPDX-License-Identifier: Apache-2.0
"""Canonical metadata for the cross-operator test matrices.

This module is deliberately test-only.  It describes public inputs, parameter
directions, structural arguments, and test capabilities; it contains no
expected numerical values or implementation-derived oracles.  Individual
test suites keep their own fixture generation and assertion/oracle logic and
adapt this catalog to the schema they already expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Shape = tuple[int, ...]
ParameterValues = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ParameterSpec:
    """One public scalar parameter and its suite fixture values."""

    name: str
    contract_default: float
    matrix_value: float
    mask_value: float
    raw_vmap_name: str


@dataclass(frozen=True)
class OperatorCase:
    """All metadata needed to adapt one implemented operator into a suite."""

    name: str
    input_names: tuple[str, ...]
    profiles: tuple[tuple[str, tuple[Shape, ...]], ...]
    parameters: tuple[ParameterSpec, ...]
    structural_defaults: tuple[tuple[str, Any], ...]
    vjp_fields: tuple[str, ...]
    orientation: str
    nn_class: str
    operator_module: str
    operator_name: str
    nonnegative: bool
    capabilities: frozenset[str]
    off_optimal_index: tuple[int, ...]
    padded_index: tuple[int, ...]
    invariance_index: tuple[int, ...]

    @property
    def cost_native(self) -> bool:
        return self.orientation == "cost-native"

    @property
    def public_entrypoints(self) -> tuple[str, ...]:
        return (self.name, f"{self.name}_value", f"{self.name}_entropy")

    @property
    def structural_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.structural_defaults)

    @property
    def contract_params(self) -> ParameterValues:
        return tuple(
            (parameter.name, parameter.contract_default)
            for parameter in self.parameters
        )

    @property
    def matrix_params(self) -> ParameterValues:
        return tuple(
            (parameter.name, parameter.matrix_value)
            for parameter in self.parameters
        )

    @property
    def mask_params(self) -> ParameterValues:
        return tuple(
            (parameter.name, parameter.mask_value)
            for parameter in self.parameters
        )

    @property
    def raw_vmap_names(self) -> tuple[str, ...]:
        return tuple(
            parameter.raw_vmap_name for parameter in self.parameters
        )

    def shapes(self, profile: str) -> tuple[Shape, ...]:
        for name, shapes in self.profiles:
            if name == profile:
                return shapes
        raise KeyError(f"unknown {self.name} input-shape profile: {profile}")


def _parameter_specs(
    *values: tuple[str, float, float, float, str],
) -> tuple[ParameterSpec, ...]:
    return tuple(ParameterSpec(*value) for value in values)


_BATCH = 2


OPERATOR_CASES = (
    OperatorCase(
        name="sw",
        input_names=("pair_scores",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("gap_score", 0.0, -0.7, -0.7, "gap"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("gap_score", "temperature"),
        orientation="score-native",
        nn_class="SmithWaterman",
        operator_module="orihime.sw",
        operator_name="_sw_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "score-alignment"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="sw_affine",
        input_names=("pair_scores",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("gap_open_score", 0.0, -1.0, -1.0, "gap_open"),
            ("gap_extend_score", 0.0, -0.3, -0.3, "gap_ext"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("gap_open_score", "gap_extend_score", "temperature"),
        orientation="score-native",
        nn_class="SmithWatermanAffine",
        operator_module="orihime.sw",
        operator_name="_sw_affine_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "affine", "score-alignment"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="nw",
        input_names=("pair_scores",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("gap_score", 0.0, -0.7, -0.7, "gap"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("gap_score", "temperature"),
        orientation="score-native",
        nn_class="NeedlemanWunsch",
        operator_module="orihime.nw",
        operator_name="_nw_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "score-alignment"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="nw_affine",
        input_names=("pair_scores",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("gap_open_score", 0.0, -1.0, -1.0, "gap_open"),
            ("gap_extend_score", 0.0, -0.3, -0.3, "gap_ext"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("gap_open_score", "gap_extend_score", "temperature"),
        orientation="score-native",
        nn_class="NeedlemanWunschAffine",
        operator_module="orihime.nw",
        operator_name="_nw_affine_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "affine", "score-alignment"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="dtw",
        input_names=("costs",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None), ("bandwidth", None)),
        vjp_fields=("temperature",),
        orientation="cost-native",
        nn_class="DynamicTimeWarping",
        operator_module="orihime.dtw",
        operator_name="_dtw_operator",
        nonnegative=True,
        capabilities=frozenset({"lengths", "bandwidth", "cost-alignment"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="lcs",
        input_names=("match_scores",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("temperature",),
        orientation="score-native",
        nn_class="LongestCommonSubsequence",
        operator_module="orihime.lcs",
        operator_name="_lcs_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "edit-match"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="lev",
        input_names=("substitution_costs",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("insertion_cost", 1.0, 0.8, 0.8, "ins_cost"),
            ("deletion_cost", 1.0, 1.1, 1.1, "del_cost"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("insertion_cost", "deletion_cost", "temperature"),
        orientation="cost-native",
        nn_class="Levenshtein",
        operator_module="orihime.edit_distance",
        operator_name="_lev_operator",
        nonnegative=True,
        capabilities=frozenset({"lengths", "edit-distance"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="osa",
        input_names=("substitution_costs",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("insertion_cost", 1.0, 0.8, 0.8, "ins_cost"),
            ("deletion_cost", 1.0, 1.1, 1.1, "del_cost"),
            ("transposition_cost", 1.0, 0.7, 0.7, "trans_cost"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(
            ("lengths", None),
            ("allowed_transpositions", None),
        ),
        vjp_fields=(
            "insertion_cost",
            "deletion_cost",
            "transposition_cost",
            "temperature",
        ),
        orientation="cost-native",
        nn_class="OptimalStringAlignment",
        operator_module="orihime.osa",
        operator_name="_osa_operator",
        nonnegative=True,
        capabilities=frozenset({
            "lengths",
            "edit-distance",
            "allowed-transpositions",
        }),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="damerau",
        input_names=("substitution_costs",),
        profiles=(
            ("contract", ((_BATCH, 4, 5),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("insertion_cost", 1.0, 0.8, 0.8, "ins_cost"),
            ("deletion_cost", 1.0, 1.1, 1.1, "del_cost"),
            ("transposition_cost", 1.0, 0.7, 0.7, "trans_cost"),
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(
            ("lengths", None),
            ("transposition_sources", None),
        ),
        vjp_fields=(
            "insertion_cost",
            "deletion_cost",
            "transposition_cost",
            "temperature",
        ),
        orientation="cost-native",
        nn_class="DamerauLevenshtein",
        operator_module="orihime.damerau",
        operator_name="_damerau_operator",
        nonnegative=True,
        capabilities=frozenset({
            "lengths",
            "edit-distance",
            "transposition-sources",
        }),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 2, 2),
    ),
    OperatorCase(
        name="mas",
        input_names=("scores",),
        profiles=(
            ("contract", ((_BATCH, 5, 3),)),
            ("validation", ((1, 4, 3),)),
            ("func", ((2, 4, 3),)),
            ("compile", ((1, 5, 3),)),
            ("masking", ((1, 5, 4),)),
            ("mask_kwarg", ((1, 5, 3),)),
            ("invariance", ((1, 5, 3),)),
            ("autocast", ((1, 5, 3),)),
        ),
        parameters=_parameter_specs(
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("temperature",),
        orientation="score-native",
        nn_class="MonotonicAlignmentSearch",
        operator_module="orihime.mas",
        operator_name="_mas_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "monotonic-alignment"}),
        off_optimal_index=(0, 1, 0),
        padded_index=(0, 4, 0),
        invariance_index=(0, 2, 1),
    ),
    OperatorCase(
        name="cky",
        input_names=("merge_scores", "leaf_scores"),
        profiles=(
            ("contract", ((_BATCH, 4, 4, 4), (_BATCH, 4))),
            ("validation", ((1, 3, 3, 3), (1, 3))),
            ("func", ((2, 3, 3, 3), (2, 3))),
            ("compile", ((1, 4, 4, 4), (1, 4))),
            ("masking", ((1, 4, 4, 4), (1, 4))),
            ("mask_kwarg", ((1, 4, 4, 4), (1, 4))),
            ("invariance", ((1, 4, 4, 4), (1, 4))),
            ("autocast", ((1, 4, 4, 4), (1, 4))),
        ),
        parameters=_parameter_specs(
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(),
        vjp_fields=("leaf_scores", "temperature"),
        orientation="score-native",
        nn_class="CKY",
        operator_module="orihime.cky",
        operator_name="_cky_operator",
        nonnegative=False,
        capabilities=frozenset({"leaf-scores", "chart-parser"}),
        off_optimal_index=(0, 0, 1, 3),
        padded_index=(0, 1, 2, 3),
        invariance_index=(0, 0, 1, 3),
    ),
    OperatorCase(
        name="eisner",
        input_names=("arc_scores",),
        profiles=(
            ("contract", ((_BATCH, 4, 4),)),
            ("validation", ((1, 3, 3),)),
            ("func", ((2, 3, 3),)),
            ("compile", ((1, 4, 4),)),
            ("masking", ((1, 4, 4),)),
            ("mask_kwarg", ((1, 4, 4),)),
            ("invariance", ((1, 5, 5),)),
            ("autocast", ((1, 4, 4),)),
        ),
        parameters=_parameter_specs(
            ("temperature", 1.0, 0.9, 1.0, "temp"),
        ),
        structural_defaults=(("lengths", None),),
        vjp_fields=("temperature",),
        orientation="score-native",
        nn_class="Eisner",
        operator_module="orihime.eisner",
        operator_name="_eisner_operator",
        nonnegative=False,
        capabilities=frozenset({"lengths", "chart-parser", "projective"}),
        off_optimal_index=(0, 0, 2),
        padded_index=(0, 3, 0),
        invariance_index=(0, 0, 2),
    ),
)


OPERATOR_NAMES = tuple(case.name for case in OPERATOR_CASES)
AUTOCAST_DTYPE_IDS = ("fp16", "bf16")
