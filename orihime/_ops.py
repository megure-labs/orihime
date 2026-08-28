# SPDX-License-Identifier: Apache-2.0
"""
Low-level operator access for orihime.

This module provides direct access to the underlying C++/CUDA operators
registered with PyTorch's dispatcher. These are the raw functions without
any Python-side validation or wrapping.

For most users, the high-level ``orihime.<op>`` Operator instances are
recommended. Public low-level access is available through ``orihime.raw.<op>``;
this internal module supplies the named primitive wrappers used there.
Use the low-level surface only when you need:
- Direct control over forward/backward passes
- Integration with custom autograd functions
- Performance-critical code that can skip Python validation

Usage:
    from orihime.raw import sw

    score, marginals = sw.forward(scores, -1.0, 1.0, lengths)
    tangent = sw.marginals_hvp(scores, vector, -1.0, 1.0, lengths)
"""

import os
import glob
import torch

from . import _build_info
from ._build_contract import validate_build_compatibility


validate_build_compatibility(
    torch,
    expected_minor=_build_info.EXPECTED_TORCH_MINOR,
    expected_lane=_build_info.EXPECTED_LANE,
)

# Extension loading state
_extension_loaded = False
_cuda_dispatch_available = False


def _has_cuda_dispatch():
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key("orihime::sw_forward", "CUDA")
    except (AttributeError, RuntimeError):
        return False


def _load_extension():
    """Load the C++/CUDA extension library."""
    global _extension_loaded, _cuda_dispatch_available

    if _extension_loaded:
        return

    lib_dir = os.path.dirname(__file__)
    lib_pattern = os.path.join(lib_dir, '_C*.so')
    libs = sorted(glob.glob(lib_pattern))

    # For editable installs (meson-python), check the build directory
    if not libs:
        project_root = os.path.dirname(lib_dir)
        build_pattern = os.path.join(project_root, 'build', '*', '_C*.so')
        libs = sorted(glob.glob(build_pattern))

    if libs:
        try:
            torch.ops.load_library(libs[0])
        except OSError as exc:
            # The extension is ABI-locked to the exact torch it was compiled
            # against: libc10/libtorch symbol versions move every torch minor.
            # The common cause is `pip install orihime` building from source under
            # PEP 517 isolation, which installs its own (latest) torch into the
            # build environment and compiles against that instead of yours --
            # pip then reports success and this is the first sign anything is
            # wrong. Say so, rather than surfacing a bare undefined symbol.
            if "undefined symbol" in str(exc):
                raise ImportError(
                    f"orihime's compiled extension is not compatible with the "
                    f"installed torch {torch.__version__}.\n\n"
                    f"This usually means it was built against a different "
                    f"torch version. If you installed from source, PEP 517 "
                    f"build isolation pulls in its own torch, so build "
                    f"against yours instead:\n\n"
                    f"    pip install torch=={torch.__version__.split('+')[0]}\n"
                    f"    pip install meson-python meson ninja\n"
                    f"    pip install --no-build-isolation --no-cache-dir orihime\n\n"
                    f"(--no-cache-dir matters: pip keys its wheel cache on the "
                    f"sdist, not on the torch it was built against, so a "
                    f"previously built incompatible wheel is otherwise reused.)"
                    f"\n\nOriginal error: {exc}"
                ) from exc
            raise
        _extension_loaded = True
        _cuda_dispatch_available = _has_cuda_dispatch()
    else:
        raise ImportError(
            f"Could not find _C extension library in {lib_dir}. "
            "Run a source build with either "
            "`python -m pip install . --no-build-isolation` or "
            "`python -m pip install . --no-build-isolation "
            "--config-settings=setup-args=-Dcuda=disabled`."
        )


def _ensure_loaded():
    """Ensure extension is loaded before accessing ops."""
    if not _extension_loaded:
        _load_extension()


def _contains_cuda_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, (list, tuple)):
        return any(_contains_cuda_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(item) for item in value.values())
    return False


def _contains_fake_tensor(value):
    if isinstance(value, torch.Tensor):
        try:
            from torch._subclasses.fake_tensor import is_fake
        except ImportError:
            return False
        return is_fake(value)
    if isinstance(value, (list, tuple)):
        return any(_contains_fake_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_fake_tensor(item) for item in value.values())
    return False


def _is_pt2_dispatch(args, kwargs):
    compiler = getattr(torch, "compiler", None)
    if (
        compiler is not None
        and getattr(compiler, "is_compiling", None) is not None
        and compiler.is_compiling()
    ):
        return True
    return _contains_fake_tensor(args) or _contains_fake_tensor(kwargs)


def _make_pt2_call(raw):
    def call(*args, **kwargs):
        with torch._C._AutoDispatchBelowAutograd():
            return raw(*args, **kwargs)

    compiler = getattr(torch, "compiler", None)
    if (
        compiler is not None
        and getattr(compiler, "allow_in_graph", None) is not None
    ):
        return compiler.allow_in_graph(call)
    return call


def _ensure_cuda_supported(args, kwargs):
    _ensure_loaded()
    if _cuda_dispatch_available:
        return
    if _contains_cuda_tensor(args) or _contains_cuda_tensor(kwargs):
        raise RuntimeError(
            "This orihime installation was built without CUDA support. "
            "Install a CUDA-enabled artifact or rebuild from source with "
            "`--config-settings=setup-args=-Dcuda=enabled`."
        )


# Load on module import
_load_extension()


_VMAP_BATCH_ARGUMENTS = frozenset(
    {
        "arc_scores",
        "costs",
        "grad_marginals",
        "leaf_scores",
        "lengths",
        "merge_scores",
        "scores",
        "sub_costs",
        "trans_mask",
        "trans_src",
        "v",
        "v_leaf",
        "v_merge",
    }
)

_VMAP_FAMILY_PARAM_FIELDS = {
    "sw": ("gap", "temp"),
    "sw_affine": ("gap_open", "gap_ext", "temp"),
    "sv_linear": ("gap", "temp"),
    "sv_affine": ("gap_open", "gap_ext", "temp"),
    "nw": ("gap", "temp"),
    "nw_affine": ("gap_open", "gap_ext", "temp"),
    "dtw": ("temp",),
    "mas": ("temp",),
    "cky": ("temp",),
    "eisner": ("temp",),
    "lev": ("ins_cost", "del_cost", "temp"),
    "lcs": ("temp",),
    "osa": ("ins_cost", "del_cost", "trans_cost", "temp"),
    "damerau": ("ins_cost", "del_cost", "trans_cost", "temp"),
}


def _vmap_schema_argument_names(raw):
    return tuple(argument.name for argument in raw._schema.arguments)


def _vmap_restore_batch(tensor, vmap_batch, native_batch, op_name):
    expected_batch = vmap_batch * native_batch
    if tensor.ndim == 0 or tensor.shape[0] != expected_batch:
        raise RuntimeError(
            f"orihime::{op_name} returned shape {tuple(tensor.shape)} from a "
            f"folded batch of size {expected_batch}"
        )
    return tensor.reshape(vmap_batch, native_batch, *tensor.shape[1:])


def _vmap_call_by_schema(raw, arguments_by_name):
    return raw(
        *(
            arguments_by_name[argument.name]
            for argument in raw._schema.arguments
        )
    )


def _make_batch_folding_vmap_rule(
    op_name,
    reduced_output_sensitivities=(),
):
    raw = getattr(torch.ops.orihime, op_name).default
    argument_names = _vmap_schema_argument_names(raw)
    batch_indices = tuple(
        index
        for index, name in enumerate(argument_names)
        if name in _VMAP_BATCH_ARGUMENTS
    )
    sensitivity_ops = tuple(
        getattr(torch.ops.orihime, name).default
        for name in reduced_output_sensitivities
    )

    def vmap_rule(info, in_dims, *args):
        for index, in_dim in enumerate(in_dims):
            if in_dim is not None and index not in batch_indices:
                raise RuntimeError(
                    f"orihime::{op_name} does not support mapping scalar "
                    f"argument {argument_names[index]!r}; keep scalar "
                    "parameters unbatched"
                )

        native_batch = None
        exposed_arguments = list(args)
        for index in batch_indices:
            argument = args[index]
            if not isinstance(argument, torch.Tensor):
                continue
            in_dim = in_dims[index]
            if in_dim is None:
                exposed = argument.unsqueeze(0).expand(
                    info.batch_size, *argument.shape
                )
            else:
                exposed = argument.movedim(in_dim, 0)
            if exposed.ndim < 2:
                raise RuntimeError(
                    f"orihime::{op_name} batch argument "
                    f"{argument_names[index]!r} has no native batch dimension"
                )
            current_batch = exposed.shape[1]
            if native_batch is None:
                native_batch = current_batch
            elif current_batch != native_batch:
                raise RuntimeError(
                    f"orihime::{op_name} received inconsistent native batch "
                    f"sizes {native_batch} and {current_batch}"
                )
            exposed_arguments[index] = exposed

        if native_batch is None:
            raise RuntimeError(
                f"orihime::{op_name} requires a mapped batch-aligned tensor"
            )

        folded_arguments = list(args)
        for index in batch_indices:
            argument = exposed_arguments[index]
            if not isinstance(argument, torch.Tensor):
                continue
            folded_arguments[index] = argument.reshape(
                info.batch_size * native_batch,
                *argument.shape[2:],
            ).contiguous()

        result = raw(*folded_arguments)
        if isinstance(result, tuple):
            outputs = list(result)
            output_kind = tuple
        elif isinstance(result, list):
            outputs = result
            output_kind = list
        else:
            outputs = [result]
            output_kind = None

        batch_output_count = len(outputs) - len(sensitivity_ops)
        if batch_output_count < 1:
            raise RuntimeError(
                f"orihime::{op_name} vmap metadata does not match its outputs"
            )
        restored = [
            _vmap_restore_batch(
                output,
                info.batch_size,
                native_batch,
                op_name,
            )
            for output in outputs[:batch_output_count]
        ]

        if sensitivity_ops:
            arguments_by_name = dict(
                zip(argument_names, folded_arguments, strict=True)
            )
            grad_marginals = exposed_arguments[
                argument_names.index("grad_marginals")
            ]
            reduction_dims = tuple(range(1, grad_marginals.ndim))
            for output, sensitivity_op in zip(
                outputs[batch_output_count:],
                sensitivity_ops,
                strict=True,
            ):
                sensitivity = _vmap_call_by_schema(
                    sensitivity_op,
                    arguments_by_name,
                )
                sensitivity = _vmap_restore_batch(
                    sensitivity,
                    info.batch_size,
                    native_batch,
                    op_name,
                )
                contracted = (grad_marginals * sensitivity).sum(
                    dim=reduction_dims
                )
                restored.append(
                    contracted.reshape(info.batch_size, *output.shape)
                )

        out_dims = [0] * len(restored)
        if output_kind is tuple:
            return tuple(restored), tuple(out_dims)
        if output_kind is list:
            return restored, out_dims
        return restored[0], 0

    return vmap_rule


def _register_named_primitive_vmap_rules():
    for family, param_fields in _VMAP_FAMILY_PARAM_FIELDS.items():
        primitive_names = [
            f"{family}_forward",
            f"{family}_forward_t",
            f"{family}_value_grad_params",
            f"{family}_marginals_hvp",
            *(
                f"{family}_marginals_grad_{field}"
                for field in param_fields
            ),
        ]
        if family == "cky":
            primitive_names.append("cky_marginals_grad_leaf")

        for op_name in primitive_names:
            torch.library.register_vmap(
                getattr(torch.ops.orihime, op_name).default,
                _make_batch_folding_vmap_rule(op_name),
            )

        backward_name = f"{family}_marginals_backward"
        sensitivity_names = tuple(
            f"{family}_marginals_grad_{field}"
            for field in param_fields
        )
        torch.library.register_vmap(
            getattr(torch.ops.orihime, backward_name).default,
            _make_batch_folding_vmap_rule(
                backward_name,
                sensitivity_names,
            ),
        )


_register_named_primitive_vmap_rules()


# Expose the dispatcher namespace for internal adapters.
orihime = torch.ops.orihime


def _wrap(name: str):
    raw = getattr(torch.ops.orihime, name)
    pt2_call = _make_pt2_call(raw)

    def wrapped(*args, **kwargs):
        _ensure_cuda_supported(args, kwargs)
        if _is_pt2_dispatch(args, kwargs):
            # The shipped forward primitives have explicit AutogradCPU/CUDA
            # kernels. Fake tensors retain their logical device, so those
            # higher-priority registrations otherwise run against meta
            # storage before the registered Meta kernel can be selected.
            return pt2_call(*args, **kwargs)
        return raw(*args, **kwargs)

    return wrapped


# =============================================================================
# Smith-Waterman
# =============================================================================

sw_forward = _wrap("sw_forward")
sw_forward_t = _wrap("sw_forward_t")
sw_value_grad_params = _wrap("sw_value_grad_params")
sw_marginals_backward = _wrap("sw_marginals_backward")
sw_marginals_hvp = _wrap("sw_marginals_hvp")
sw_marginals_grad_gap = _wrap("sw_marginals_grad_gap")
sw_marginals_grad_temp = _wrap("sw_marginals_grad_temp")

sw_affine_forward = _wrap("sw_affine_forward")
sw_affine_forward_t = _wrap("sw_affine_forward_t")
sw_affine_value_grad_params = _wrap("sw_affine_value_grad_params")
sw_affine_marginals_backward = _wrap("sw_affine_marginals_backward")
sw_affine_marginals_hvp = _wrap("sw_affine_marginals_hvp")
sw_affine_marginals_grad_gap_open = _wrap("sw_affine_marginals_grad_gap_open")
sw_affine_marginals_grad_gap_ext = _wrap("sw_affine_marginals_grad_gap_ext")
sw_affine_marginals_grad_temp = _wrap("sw_affine_marginals_grad_temp")

# NEW API: canonical Saigo-Vert local alignment (affine gap)
sv_affine_forward = _wrap("sv_affine_forward")
sv_affine_forward_t = _wrap("sv_affine_forward_t")
sv_affine_value_grad_params = _wrap("sv_affine_value_grad_params")
sv_affine_marginals_backward = _wrap("sv_affine_marginals_backward")
sv_affine_marginals_hvp = _wrap("sv_affine_marginals_hvp")
sv_affine_marginals_grad_gap_open = _wrap("sv_affine_marginals_grad_gap_open")
sv_affine_marginals_grad_gap_ext = _wrap("sv_affine_marginals_grad_gap_ext")
sv_affine_marginals_grad_temp = _wrap("sv_affine_marginals_grad_temp")

# Canonical Saigo-Vert local alignment (linear gap)
sv_linear_forward = _wrap("sv_linear_forward")
sv_linear_forward_t = _wrap("sv_linear_forward_t")
sv_linear_value_grad_params = _wrap("sv_linear_value_grad_params")
sv_linear_marginals_backward = _wrap("sv_linear_marginals_backward")
sv_linear_marginals_hvp = _wrap("sv_linear_marginals_hvp")
sv_linear_marginals_grad_gap = _wrap("sv_linear_marginals_grad_gap")
sv_linear_marginals_grad_temp = _wrap("sv_linear_marginals_grad_temp")


# =============================================================================
# Dynamic Time Warping
# =============================================================================

dtw_forward = _wrap("dtw_forward")
dtw_forward_t = _wrap("dtw_forward_t")
dtw_value_grad_params = _wrap("dtw_value_grad_params")
dtw_marginals_backward = _wrap("dtw_marginals_backward")
dtw_marginals_hvp = _wrap("dtw_marginals_hvp")
dtw_marginals_grad_temp = _wrap("dtw_marginals_grad_temp")


# =============================================================================
# CKY Parsing
# =============================================================================

cky_forward = _wrap("cky_forward")
cky_forward_t = _wrap("cky_forward_t")
cky_value_grad_params = _wrap("cky_value_grad_params")
cky_marginals_backward = _wrap("cky_marginals_backward")
cky_marginals_hvp = _wrap("cky_marginals_hvp")
cky_marginals_grad_leaf = _wrap("cky_marginals_grad_leaf")
cky_marginals_grad_temp = _wrap("cky_marginals_grad_temp")


# =============================================================================
# Needleman-Wunsch (linear gap)
# =============================================================================

nw_forward = _wrap("nw_forward")
nw_forward_t = _wrap("nw_forward_t")
nw_value_grad_params = _wrap("nw_value_grad_params")
nw_marginals_backward = _wrap("nw_marginals_backward")
nw_marginals_hvp = _wrap("nw_marginals_hvp")
nw_marginals_grad_gap = _wrap("nw_marginals_grad_gap")
nw_marginals_grad_temp = _wrap("nw_marginals_grad_temp")

nw_affine_forward = _wrap("nw_affine_forward")
nw_affine_forward_t = _wrap("nw_affine_forward_t")
nw_affine_value_grad_params = _wrap("nw_affine_value_grad_params")
nw_affine_marginals_backward = _wrap("nw_affine_marginals_backward")
nw_affine_marginals_hvp = _wrap("nw_affine_marginals_hvp")
nw_affine_marginals_grad_gap_open = _wrap("nw_affine_marginals_grad_gap_open")
nw_affine_marginals_grad_gap_ext = _wrap("nw_affine_marginals_grad_gap_ext")
nw_affine_marginals_grad_temp = _wrap("nw_affine_marginals_grad_temp")


# =============================================================================
# Monotonic Alignment Search
# =============================================================================

mas_forward = _wrap("mas_forward")
mas_forward_t = _wrap("mas_forward_t")
mas_value_grad_params = _wrap("mas_value_grad_params")
mas_marginals_backward = _wrap("mas_marginals_backward")
mas_marginals_hvp = _wrap("mas_marginals_hvp")
mas_marginals_grad_temp = _wrap("mas_marginals_grad_temp")


# =============================================================================
# Eisner (Projective Dependency Parsing)
# =============================================================================

eisner_forward = _wrap("eisner_forward")
eisner_forward_t = _wrap("eisner_forward_t")
eisner_value_grad_params = _wrap("eisner_value_grad_params")
eisner_marginals_backward = _wrap("eisner_marginals_backward")
eisner_marginals_hvp = _wrap("eisner_marginals_hvp")
eisner_marginals_grad_temp = _wrap("eisner_marginals_grad_temp")


# =============================================================================
# Edit Distance Family
# =============================================================================

# Levenshtein
lev_forward = _wrap("lev_forward")
lev_forward_t = _wrap("lev_forward_t")
lev_value_grad_params = _wrap("lev_value_grad_params")
lev_marginals_backward = _wrap("lev_marginals_backward")
lev_marginals_hvp = _wrap("lev_marginals_hvp")
lev_marginals_grad_ins_cost = _wrap("lev_marginals_grad_ins_cost")
lev_marginals_grad_del_cost = _wrap("lev_marginals_grad_del_cost")
lev_marginals_grad_temp = _wrap("lev_marginals_grad_temp")

# Longest Common Subsequence
lcs_forward = _wrap("lcs_forward")
lcs_forward_t = _wrap("lcs_forward_t")
lcs_value_grad_params = _wrap("lcs_value_grad_params")
lcs_marginals_backward = _wrap("lcs_marginals_backward")
lcs_marginals_hvp = _wrap("lcs_marginals_hvp")
lcs_marginals_grad_temp = _wrap("lcs_marginals_grad_temp")

# OSA (Optimal String Alignment / Restricted Damerau-Levenshtein)
osa_forward = _wrap("osa_forward")
osa_forward_t = _wrap("osa_forward_t")
osa_value_grad_params = _wrap("osa_value_grad_params")
osa_marginals_backward = _wrap("osa_marginals_backward")
osa_marginals_hvp = _wrap("osa_marginals_hvp")
osa_marginals_grad_ins_cost = _wrap("osa_marginals_grad_ins_cost")
osa_marginals_grad_del_cost = _wrap("osa_marginals_grad_del_cost")
osa_marginals_grad_trans_cost = _wrap("osa_marginals_grad_trans_cost")
osa_marginals_grad_temp = _wrap("osa_marginals_grad_temp")

# True Damerau-Levenshtein
damerau_forward = _wrap("damerau_forward")
damerau_forward_t = _wrap("damerau_forward_t")
damerau_value_grad_params = _wrap("damerau_value_grad_params")
damerau_marginals_backward = _wrap("damerau_marginals_backward")
damerau_marginals_hvp = _wrap("damerau_marginals_hvp")
damerau_marginals_grad_ins_cost = _wrap("damerau_marginals_grad_ins_cost")
damerau_marginals_grad_del_cost = _wrap("damerau_marginals_grad_del_cost")
damerau_marginals_grad_trans_cost = _wrap("damerau_marginals_grad_trans_cost")
damerau_marginals_grad_temp = _wrap("damerau_marginals_grad_temp")
