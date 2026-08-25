# SPDX-License-Identifier: Apache-2.0
"""Differentiable dynamic-programming operators for PyTorch.

The high-level v3 API consists of plain map, value, and entropy functions.
The shipped kernels accept exactly one leading batch dimension: alignment
and edit-distance inputs use ``[B, L1, L2]``; Eisner uses ``[B, N, N]``;
CKY uses merge scores ``[B, N, N, N]`` plus leaf scores ``[B, N]``. This is
the documented narrower alternative to arbitrary leading batch dimensions.

For two-sequence operators, ``lengths`` is a contiguous ``torch.int32``
tensor shaped ``[B, 2]`` on the input device. Eisner lengths are ``[B]``.
Maps are zero outside declared active lengths. CKY has no lengths argument,
so callers must mask padded charts in their scores. To exclude cells, pass a boolean ``mask=`` (``True`` marks excluded cells)
to any map/value/entropy function; the operator applies the orientation-
correct infinity internally and normalizes it to an answer-preserving finite
sentinel, so excluded cells never produce ``NaN`` entropy and callers never
handle infinities. Equivalently, score-native operators accept ``-inf`` score
masks and cost-native operators accept ``+inf`` cost masks written directly. The opposite infinity orientation and all ``NaN`` inputs are rejected,
including under ``torch.compile``. Model parameters are scalar numbers, 0-D
tensors, or one-element vectors; per-batch parameter broadcasting and other
singleton shapes are not provided. Temperature must be finite and strictly
positive.

The supported FP32 numerical domain requires every finite score, cost, and
scoring parameter to satisfy ``abs(value) / temperature <= 80``. Inputs passed
to the native kernels must be contiguous. Every map/value/entropy function
accepts ``dtype=torch.float32`` as an explicit FP32-accumulation escape hatch;
all other explicit ``dtype=`` values are rejected. CUDA autocast runs the
native dynamic programs in FP32, returns finite FP32 outputs for FP16/BF16
inputs, and restores gradients to the original input dtype across the
autocast boundary.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("py-d2p")
except PackageNotFoundError:
    __version__ = "0.1.0+source"

# Load the extension before importing any kernel adapters.
from . import _ops as _ops

# Register fake tensor implementations for torch.compile support.
from . import _compile as _compile

from .cky import cky, cky_entropy, cky_leaf_map, cky_value
from .damerau import (
    build_damerau_transposition_sources,
    damerau,
    damerau_entropy,
    damerau_value,
)
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
from .sw import (
    sw,
    sw_affine,
    sw_affine_entropy,
    sw_affine_value,
    sw_entropy,
    sw_value,
)

# Public module and no-graph tiers.
from . import nn as nn
from . import raw as raw

__all__ = [
    "__version__",
    "sw",
    "sw_value",
    "sw_entropy",
    "sw_affine",
    "sw_affine_value",
    "sw_affine_entropy",
    "nw",
    "nw_value",
    "nw_entropy",
    "nw_affine",
    "nw_affine_value",
    "nw_affine_entropy",
    "dtw",
    "dtw_value",
    "dtw_entropy",
    "lcs",
    "lcs_value",
    "lcs_entropy",
    "lev",
    "lev_value",
    "lev_entropy",
    "osa",
    "osa_value",
    "osa_entropy",
    "damerau",
    "damerau_value",
    "damerau_entropy",
    "mas",
    "mas_value",
    "mas_entropy",
    "cky",
    "cky_value",
    "cky_entropy",
    "cky_leaf_map",
    "eisner",
    "eisner_value",
    "eisner_entropy",
    "build_damerau_transposition_sources",
    "nn",
    "raw",
]
