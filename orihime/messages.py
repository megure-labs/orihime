# SPDX-License-Identifier: Apache-2.0
"""Native forward DP tables for external grid and tree decoders.

The tables and value are detached. Decoders and arbitrary table VJPs belong to
the consuming package; ordinary Orihime map/value/entropy calls remain fully
differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from . import _ops

_forward = _ops._wrap("grid_forward_messages")
_tree_forward = _ops._wrap("tree_forward_messages")

GRID_ALGORITHMS = ("sw", "sw_affine", "sv", "sv_affine", "nw", "nw_affine",
                   "dtw", "lcs", "lev", "osa", "damerau", "mas")
TREE_ALGORITHMS = ("cky", "eisner")
_COST = frozenset(("dtw", "lev", "osa", "damerau"))
_THREE = frozenset(("sw_affine", "sv", "sv_affine", "nw_affine"))


@dataclass(frozen=True)
class ForwardMessages:
    """Versioned native tables with the data needed to interpret each edge.

    ``alpha`` is [B,states,N+1,M+1], except MAS [B,1,N,M]. State order is
    (M,I,D) for three-state grid algorithms and (H,) for one-state grids. Trees
    use [B,states,N,N], with CKY (I,) and Eisner (C_R,C_L,I_R,I_L).
    CKY additionally retains leaf_scores. Tables and ``value``
    use the algorithm's original score/cost orientation. ``value`` is the
    soft Bellman value, not an expected score. ``parameters`` contains the
    algorithm's scalar tensors in their public orientation. Metadata tensors
    are private snapshots; do not modify a result in place before decoding it.
    """
    alpha: Tensor
    value: Tensor
    inputs: Tensor
    parameters: tuple[Tensor, ...]
    temperature: Tensor
    lengths: Tensor
    topology: Tensor
    algorithm: str
    bandwidth: int | None
    layout_version: int = 1
    leaf_scores: Tensor | None = None

    @property
    def states(self) -> tuple[str, ...]:
        if self.algorithm == "cky":
            return ("I",)
        if self.algorithm == "eisner":
            return ("C_R", "C_L", "I_R", "I_L")
        return ("M", "I", "D") if self.algorithm in _THREE else ("H",)

    @property
    def orientation(self) -> str:
        return "cost" if self.algorithm in _COST else "score"

    @property
    def padded(self) -> bool:
        return self.algorithm not in ("mas", *TREE_ALGORITHMS)

    @property
    def is_tree(self) -> bool:
        return self.algorithm in TREE_ALGORITHMS


def _scalar(x: float | Tensor, ref: Tensor, name: str) -> Tensor:
    if isinstance(x, Tensor):
        if x.numel() != 1 or x.ndim > 1 or not x.is_floating_point():
            raise ValueError(f"{name} must be a floating scalar tensor")
        out = x.to(device=ref.device, dtype=torch.float32).reshape(()).clone()
    else:
        out = ref.new_tensor(float(x))
    if not bool(torch.isfinite(out)):
        raise ValueError(f"{name} must be finite in float32")
    return out


def forward_messages(
    inputs: Tensor, *, algorithm: str = "sv",
    leaf_scores: Tensor | None = None,
    gap_score: float | Tensor = 0.0,
    gap_open_score: float | Tensor = 0.0,
    gap_extend_score: float | Tensor = 0.0,
    insertion_cost: float | Tensor = 1.0,
    deletion_cost: float | Tensor = 1.0,
    transposition_cost: float | Tensor = 1.0,
    temperature: float | Tensor = 1.0,
    lengths: Tensor | None = None, mask: Tensor | None = None,
    bandwidth: int | None = None,
    allowed_transpositions: Tensor | None = None,
    transposition_sources: Tensor | None = None,
) -> ForwardMessages:
    """Run only the native forward DP on CPU, NVIDIA CUDA or AMD ROCm.

    Inputs must be contiguous float32 [B,N,M]. ``mask=True`` excludes a
    primary-input edge. Local algorithms count one empty alignment. DTW bands
    constrain occupancy; edit topology follows the corresponding public
    operator. The returned tables have no autograd rule: an external consumer
    may attach its own backward to these exact messages.
    """
    if algorithm in TREE_ALGORITHMS:
        if bandwidth is not None or allowed_transpositions is not None or transposition_sources is not None:
            raise ValueError("grid topology arguments are not supported for trees")
        return _tree_messages(inputs,leaf_scores,algorithm,temperature,lengths,mask)
    if leaf_scores is not None:
        raise ValueError("leaf_scores is only supported for CKY")
    if algorithm not in GRID_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {GRID_ALGORITHMS+TREE_ALGORITHMS}")
    if not isinstance(inputs, Tensor) or inputs.dtype != torch.float32:
        raise TypeError("inputs must be a float32 tensor")
    if inputs.ndim != 3 or not inputs.is_contiguous():
        raise ValueError("inputs must be contiguous with shape [B,N,M]")
    if inputs.device.type not in ("cpu", "cuda"):
        raise ValueError("forward messages support CPU, CUDA and ROCm devices")
    if bandwidth is not None and (algorithm != "dtw" or isinstance(bandwidth, bool)
                                  or not isinstance(bandwidth, int) or bandwidth < 0):
        raise ValueError("bandwidth is a non-negative integer for DTW only")
    if allowed_transpositions is not None and algorithm != "osa":
        raise ValueError("allowed_transpositions is only supported for OSA")
    if transposition_sources is not None and algorithm != "damerau":
        raise ValueError("transposition_sources is only supported for Damerau")
    b, n, m = inputs.shape
    if lengths is None:
        lengths = torch.tensor((n,m),device=inputs.device,dtype=torch.int32).expand(b,2).contiguous()
    if not isinstance(lengths, Tensor):
        raise TypeError("lengths must be a tensor")
    lengths = lengths.clone()
    if algorithm == "mas" and lengths.shape == (b,2) and bool((lengths[:,1] <= 0).any()):
        raise ValueError("MAS requires frame length >= token length > 0")
    primary = inputs.clone()
    if mask is not None:
        if (not isinstance(mask, Tensor) or mask.dtype != torch.bool or
            mask.shape != inputs.shape or mask.device != inputs.device):
            raise ValueError("mask must be boolean and match inputs")
        primary = primary.masked_fill(mask, math.inf if algorithm in _COST else -math.inf)
    t = _scalar(temperature, inputs, "temperature")
    if not bool(t > 0):
        raise ValueError("temperature must be positive")
    if algorithm.endswith("affine"):
        parameters = (_scalar(gap_open_score, inputs, "gap_open_score"),
                      _scalar(gap_extend_score, inputs, "gap_extend_score"))
    elif algorithm in ("sw", "sv", "nw"):
        parameters = (_scalar(gap_score, inputs, "gap_score"),)
    elif algorithm in ("lev", "osa", "damerau"):
        parameters = (_scalar(insertion_cost, inputs, "insertion_cost"),
                      _scalar(deletion_cost, inputs, "deletion_cost"))
        if algorithm != "lev":
            parameters += (_scalar(transposition_cost, inputs, "transposition_cost"),)
    else:
        parameters = ()
    oriented = -primary if algorithm in _COST else primary
    if bool((torch.isnan(oriented) | torch.isposinf(oriented)).any()):
        raise ValueError("inputs contain NaN or an incorrectly oriented infinity")
    if bool((torch.isfinite(oriented) & (oriented.abs()/t > 80)).any()) or any(
        bool(p.abs()/t > 80) for p in parameters
    ):
        raise ValueError("finite inputs and parameters must satisfy abs(value)/temperature <= 80")
    topology = inputs.new_empty((0,))
    if algorithm == "osa":
        if allowed_transpositions is None:
            topology = torch.zeros_like(inputs)
        else:
            if (not isinstance(allowed_transpositions, Tensor) or allowed_transpositions.dtype != torch.bool or
                allowed_transpositions.shape != inputs.shape or allowed_transpositions.device != inputs.device):
                raise ValueError("allowed_transpositions must be boolean and match inputs")
            topology = allowed_transpositions.to(torch.float32).contiguous()
    elif algorithm == "damerau":
        topology = (torch.full((*inputs.shape,2),-1,device=inputs.device,dtype=torch.int32)
                    if transposition_sources is None else transposition_sources.clone())
    dispatch = "CPU" if inputs.device.type == "cpu" else "CUDA"
    if not torch._C._dispatch_has_kernel_for_dispatch_key("orihime::grid_forward_messages", dispatch):
        raise RuntimeError(f"this Orihime build has no {dispatch} forward-message lane")
    values = [float(p.detach()) for p in parameters] + [0.0]*3
    with torch.no_grad():
        # Match native unreachable-state sentinels. The metadata retains the
        # structural infinities so external decoders can exclude these edges.
        native_inputs = torch.nan_to_num(primary,neginf=-1e30,posinf=1e30)
        alpha, value = _forward(native_inputs,lengths,topology,
            GRID_ALGORITHMS.index(algorithm),*values[:3],float(t.detach()),
            -1 if bandwidth is None else bandwidth)
    shape = (b,1,n,m) if algorithm == "mas" else (b,3 if algorithm in _THREE else 1,n+1,m+1)
    return ForwardMessages(alpha.reshape(shape),value,primary,parameters,t,lengths,
                           topology,algorithm,bandwidth)


def _tree_messages(inputs, leaves, algorithm, temperature, lengths, mask):
    rank = 4 if algorithm == "cky" else 3
    if not isinstance(inputs, Tensor) or inputs.dtype != torch.float32:
        raise TypeError("tree scores must be a float32 tensor")
    if inputs.ndim != rank or not inputs.is_contiguous() or len(set(inputs.shape[1:])) != 1:
        raise ValueError("CKY requires contiguous [B,N,N,N]; Eisner requires contiguous [B,N,N]")
    if inputs.device.type not in ("cpu","cuda"):
        raise ValueError("tree messages support CPU, CUDA and ROCm")
    b,n = inputs.shape[:2]
    if n < 1:
        raise ValueError("trees require N >= 1")
    primary = inputs.clone()
    if mask is not None:
        if not isinstance(mask,Tensor) or mask.dtype != torch.bool or mask.shape != inputs.shape or mask.device != inputs.device:
            raise ValueError("mask must be boolean and match inputs")
        primary = primary.masked_fill(mask,-math.inf)
    if algorithm == "cky":
        if (not isinstance(leaves,Tensor) or leaves.dtype != torch.float32 or
            leaves.shape != (b,n) or leaves.device != inputs.device or not leaves.is_contiguous()):
            raise ValueError("CKY requires contiguous float32 leaf_scores [B,N] on the input device")
        leaves = leaves.clone()
    elif leaves is not None:
        raise ValueError("Eisner has no leaf scores")
    else:
        leaves = inputs.new_empty(0)
    if algorithm == "cky" and isinstance(temperature,Tensor) and temperature.numel() != 1:
        if temperature.shape != (b,n,n) or not temperature.is_floating_point():
            raise ValueError("CKY temperature must be scalar or [B,N,N]")
        t = temperature.to(inputs).contiguous().clone()
    else:
        t = _scalar(temperature,inputs,"temperature")
    if not bool(torch.isfinite(t).all()) or not bool((t > 0).all()):
        raise ValueError("temperature must be finite and positive")
    lower_t = t.min() if t.numel() else inputs.new_tensor(1)
    for scores in (primary,leaves):
        if bool((torch.isnan(scores)|torch.isposinf(scores)).any()):
            raise ValueError("tree scores contain NaN or positive infinity")
        if bool((torch.isfinite(scores) & (scores.abs()/lower_t > 80)).any()):
            raise ValueError("finite scores must satisfy abs(value)/temperature <= 80")
    if lengths is None:
        lengths = torch.full((b,),n,dtype=torch.int32,device=inputs.device)
    if not isinstance(lengths,Tensor):
        raise TypeError("lengths must be a tensor")
    lengths = lengths.clone()
    with torch.no_grad():
        alpha,value = _tree_forward(torch.nan_to_num(primary,neginf=-1e30),
            torch.nan_to_num(leaves,neginf=-1e30),t,lengths,TREE_ALGORITHMS.index(algorithm))
    return ForwardMessages(alpha,value,primary,(),t,lengths,inputs.new_empty(0),
                           algorithm,None,leaf_scores=leaves if algorithm == "cky" else None)


__all__ = ["ForwardMessages", "forward_messages", "GRID_ALGORITHMS", "TREE_ALGORITHMS"]
