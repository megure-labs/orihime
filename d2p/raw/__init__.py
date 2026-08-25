# SPDX-License-Identifier: Apache-2.0
"""The low-level tier: no-graph VJPs plus the raw kernel bindings.

Every operator module exposes the type-stable VJP contract (``vjp_fields``,
``vjp_one``, ``vjp``; see :mod:`d2p.raw._base`) alongside the named kernel
bindings (``forward``, ``forward_t``, ``value_grad_params``,
``marginals_backward``, ``marginals_hvp``, ``marginals_grad_*``). This is the
single supported low-level surface. Raw VJPs require a contiguous FP32
cotangent matching the primary map's shape and device. Named
``marginals_hvp`` and ``marginals_backward`` bindings require the same strict
layout, dtype, shape, and device contract for their tangent or cotangent
vector, respectively; invalid user vectors are rejected before dispatch.
Private autograd paths may normalize framework-created gradients before they
reach these bindings. Prefer the top-level functions or ``d2p.nn`` for normal
model code.
"""

from ._base import VJPFields
from . import cky as cky
from . import damerau as damerau
from . import dtw as dtw
from . import eisner as eisner
from . import lcs as lcs
from . import lev as lev
from . import mas as mas
from . import nw as nw
from . import nw_affine as nw_affine
from . import osa as osa
from . import sw as sw
from . import sw_affine as sw_affine

__all__ = [
    "VJPFields",
    "sw",
    "sw_affine",
    "nw",
    "nw_affine",
    "dtw",
    "lcs",
    "lev",
    "osa",
    "damerau",
    "mas",
    "cky",
    "eisner",
]
