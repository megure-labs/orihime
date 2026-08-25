# SPDX-License-Identifier: Apache-2.0
"""
FakeTensor registrations for torch.compile support.

This module registers "fake" (meta) implementations for all orihime operators,
enabling torch.compile, torch.export, and FX tracing to work correctly.

The fake implementations describe output tensor shapes without running
the actual CUDA kernels.
"""

import torch
from torch.library import register_fake

# Ensure extension is loaded
from . import _ops  # noqa: F401
from ._pt2_ops import install_pt2_dispatch


# =============================================================================
# NEW API: Smith-Waterman (Regular - Linear Gap)
# =============================================================================


@register_fake("orihime::sw_forward")
def sw_forward_fake(scores, gap, temp, lengths):
    """sw_forward returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("orihime::sw_forward_t")
def sw_forward_t_fake(scores, gap, temp, lengths):
    """sw_forward_t (tensor params) returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("orihime::sw_value_grad_params")
def sw_value_grad_params_fake(scores, gap, temp, lengths):
    """Returns (grad_gap, grad_temp) per batch."""
    B = scores.size(0)
    return (scores.new_empty([B]), scores.new_empty([B]))


@register_fake("orihime::sw_marginals_backward")
def sw_marginals_backward_fake(scores, grad_marginals, gap, temp, lengths):
    """Returns (grad_scores, grad_gap, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@register_fake("orihime::sw_marginals_hvp")
def sw_marginals_hvp_fake(scores, v, gap, temp, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("orihime::sw_marginals_grad_gap")
def sw_marginals_grad_gap_fake(scores, gap, temp, lengths):
    """Returns d(marginals)/d(gap) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("orihime::sw_marginals_grad_temp")
def sw_marginals_grad_temp_fake(scores, gap, temp, lengths):
    """Returns d(marginals)/d(temperature) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


# =============================================================================
# NEW API: Smith-Waterman (Affine Gap)
# =============================================================================


@register_fake("orihime::sw_affine_forward")
def sw_affine_forward_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("orihime::sw_affine_forward_t")
def sw_affine_forward_t_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("orihime::sw_affine_value_grad_params")
def sw_affine_value_grad_params_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns (grad_gap_open, grad_gap_ext, grad_temp) per batch."""
    B = scores.size(0)
    return (scores.new_empty([B]), scores.new_empty([B]), scores.new_empty([B]))


@register_fake("orihime::sw_affine_marginals_backward")
def sw_affine_marginals_backward_fake(
    scores, grad_marginals, gap_open, gap_ext, temp, lengths
):
    """Returns (grad_scores, grad_gap_open, grad_gap_ext, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@register_fake("orihime::sw_affine_marginals_hvp")
def sw_affine_marginals_hvp_fake(scores, v, gap_open, gap_ext, temp, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("orihime::sw_affine_marginals_grad_gap_open")
def sw_affine_marginals_grad_gap_open_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns d(marginals)/d(gap_open) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("orihime::sw_affine_marginals_grad_gap_ext")
def sw_affine_marginals_grad_gap_ext_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns d(marginals)/d(gap_ext) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("orihime::sw_affine_marginals_grad_temp")
def sw_affine_marginals_grad_temp_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns d(marginals)/d(temperature) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


# =============================================================================
# Namespaced API
# =============================================================================


def _named_forward_fake(primary, *unused_args):
    """Return a per-batch value and a map shaped like the primary input."""
    return [
        primary.new_empty([primary.size(0)]),
        primary.new_empty(primary.shape),
    ]


def _named_batch_fake(primary, *unused_args):
    """Return one per-batch scalar field."""
    return primary.new_empty([primary.size(0)])


def _named_batch_tuple_fake(count):
    """Build a fake returning ``count`` per-batch scalar fields."""

    def fake(primary, *unused_args):
        return tuple(
            primary.new_empty([primary.size(0)]) for _ in range(count)
        )

    return fake


def _named_map_backward_fake(param_count):
    """Build a map backward fake with contracted scalar parameter gradients."""

    def fake(primary, *unused_args):
        return (
            primary.new_empty(primary.shape),
            *(primary.new_empty([1]) for _ in range(param_count)),
        )

    return fake


def _named_map_fake(primary, *unused_args):
    """Return a map-shaped derivative field."""
    return primary.new_empty(primary.shape)


def _cky_value_grad_params_fake(merge_scores, leaf_scores, *unused_args):
    return (
        leaf_scores.new_empty(leaf_scores.shape),
        merge_scores.new_empty([merge_scores.size(0)]),
    )


def _cky_marginals_backward_fake(
    merge_scores, leaf_scores, *unused_args
):
    return (
        merge_scores.new_empty(merge_scores.shape),
        leaf_scores.new_empty(leaf_scores.shape),
        merge_scores.new_empty([1]),
    )


for _name in (
    "nw",
    "nw_affine",
    "dtw",
    "cky",
    "mas",
    "eisner",
    "lev",
    "lcs",
    "osa",
    "damerau",
):
    register_fake(f"orihime::{_name}_forward")(_named_forward_fake)
    register_fake(f"orihime::{_name}_forward_t")(_named_forward_fake)


for _name in ("dtw", "mas", "eisner", "lcs"):
    register_fake(f"orihime::{_name}_value_grad_params")(_named_batch_fake)


for _name, _count in {
    "nw": 2,
    "nw_affine": 4,
    "lev": 3,
    "osa": 4,
    "damerau": 4,
}.items():
    register_fake(f"orihime::{_name}_value_grad_params")(
        _named_batch_tuple_fake(_count)
    )


register_fake("orihime::cky_value_grad_params")(_cky_value_grad_params_fake)


for _name, _param_count in {
    "nw": 2,
    "nw_affine": 3,
    "dtw": 1,
    "mas": 1,
    "eisner": 1,
    "lev": 3,
    "lcs": 1,
    "osa": 4,
    "damerau": 4,
}.items():
    register_fake(f"orihime::{_name}_marginals_backward")(
        _named_map_backward_fake(_param_count)
    )


register_fake("orihime::cky_marginals_backward")(
    _cky_marginals_backward_fake
)


for _name in (
    "nw",
    "nw_affine",
    "dtw",
    "cky",
    "mas",
    "eisner",
    "lev",
    "lcs",
    "osa",
    "damerau",
):
    register_fake(f"orihime::{_name}_marginals_hvp")(_named_map_fake)


for _name in (
    "nw_marginals_grad_gap",
    "nw_marginals_grad_temp",
    "nw_affine_marginals_grad_gap_open",
    "nw_affine_marginals_grad_gap_ext",
    "nw_affine_marginals_grad_temp",
    "dtw_marginals_grad_temp",
    "cky_marginals_grad_leaf",
    "cky_marginals_grad_temp",
    "mas_marginals_grad_temp",
    "eisner_marginals_grad_temp",
    "lev_marginals_grad_ins_cost",
    "lev_marginals_grad_del_cost",
    "lev_marginals_grad_temp",
    "lcs_marginals_grad_temp",
    "osa_marginals_grad_ins_cost",
    "osa_marginals_grad_del_cost",
    "osa_marginals_grad_trans_cost",
    "osa_marginals_grad_temp",
    "damerau_marginals_grad_ins_cost",
    "damerau_marginals_grad_del_cost",
    "damerau_marginals_grad_trans_cost",
    "damerau_marginals_grad_temp",
):
    register_fake(f"orihime::{_name}")(_named_map_fake)


# =============================================================================
# Autocast (AMP) Support
# =============================================================================
#
# Register autocast behavior for all orihime operators. DP algorithms need FP32 for
# numerical stability due to long sequential dependency chains and logsumexp
# operations that accumulate error in reduced precision (FP16/BF16).
#
# When torch.autocast is enabled, inputs are automatically cast to FP32.

_SHIPPED_OPS = (
    # Smith-Waterman
    "sw_forward",
    "sw_forward_t",
    "sw_value_grad_params",
    "sw_marginals_backward",
    "sw_marginals_hvp",
    "sw_marginals_grad_gap",
    "sw_marginals_grad_temp",
    # Smith-Waterman affine
    "sw_affine_forward",
    "sw_affine_forward_t",
    "sw_affine_value_grad_params",
    "sw_affine_marginals_backward",
    "sw_affine_marginals_hvp",
    "sw_affine_marginals_grad_gap_open",
    "sw_affine_marginals_grad_gap_ext",
    "sw_affine_marginals_grad_temp",
    # Needleman-Wunsch
    "nw_forward",
    "nw_forward_t",
    "nw_value_grad_params",
    "nw_marginals_backward",
    "nw_marginals_hvp",
    "nw_marginals_grad_gap",
    "nw_marginals_grad_temp",
    # Needleman-Wunsch affine
    "nw_affine_forward",
    "nw_affine_forward_t",
    "nw_affine_value_grad_params",
    "nw_affine_marginals_backward",
    "nw_affine_marginals_hvp",
    "nw_affine_marginals_grad_gap_open",
    "nw_affine_marginals_grad_gap_ext",
    "nw_affine_marginals_grad_temp",
    # Dynamic time warping
    "dtw_forward",
    "dtw_forward_t",
    "dtw_value_grad_params",
    "dtw_marginals_backward",
    "dtw_marginals_hvp",
    "dtw_marginals_grad_temp",
    # CKY
    "cky_forward",
    "cky_forward_t",
    "cky_value_grad_params",
    "cky_marginals_backward",
    "cky_marginals_hvp",
    "cky_marginals_grad_leaf",
    "cky_marginals_grad_temp",
    # Monotonic alignment search
    "mas_forward",
    "mas_forward_t",
    "mas_value_grad_params",
    "mas_marginals_backward",
    "mas_marginals_hvp",
    "mas_marginals_grad_temp",
    # Eisner
    "eisner_forward",
    "eisner_forward_t",
    "eisner_value_grad_params",
    "eisner_marginals_backward",
    "eisner_marginals_hvp",
    "eisner_marginals_grad_temp",
    # Levenshtein
    "lev_forward",
    "lev_forward_t",
    "lev_value_grad_params",
    "lev_marginals_backward",
    "lev_marginals_hvp",
    "lev_marginals_grad_ins_cost",
    "lev_marginals_grad_del_cost",
    "lev_marginals_grad_temp",
    # Longest common subsequence
    "lcs_forward",
    "lcs_forward_t",
    "lcs_value_grad_params",
    "lcs_marginals_backward",
    "lcs_marginals_hvp",
    "lcs_marginals_grad_temp",
    # Optimal string alignment
    "osa_forward",
    "osa_forward_t",
    "osa_value_grad_params",
    "osa_marginals_backward",
    "osa_marginals_hvp",
    "osa_marginals_grad_ins_cost",
    "osa_marginals_grad_del_cost",
    "osa_marginals_grad_trans_cost",
    "osa_marginals_grad_temp",
    # Damerau-Levenshtein
    "damerau_forward",
    "damerau_forward_t",
    "damerau_value_grad_params",
    "damerau_marginals_backward",
    "damerau_marginals_hvp",
    "damerau_marginals_grad_ins_cost",
    "damerau_marginals_grad_del_cost",
    "damerau_marginals_grad_trans_cost",
    "damerau_marginals_grad_temp",
)

try:
    from torch.library import register_autocast
except ImportError:
    # register_autocast not available in older PyTorch versions
    pass
else:
    # The namespaced operators are the shipped low-level surface used by
    # orihime/ops/*.py. Keep every forward, backward, HVP, and sensitivity
    # primitive on the same FP32 autocast policy.
    for _op_name in _SHIPPED_OPS:
        register_autocast(f"orihime::{_op_name}", "cuda", torch.float32)


# Inductor must retain an opaque target all the way through lowering. The
# mirrored custom operators delegate FakeTensor calls to the Meta registrations
# above and runtime calls to the unchanged native CPU/CUDA implementations.
install_pt2_dispatch(_ops, _SHIPPED_OPS)
