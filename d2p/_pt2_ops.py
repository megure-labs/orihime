# SPDX-License-Identifier: Apache-2.0
"""Internal transform-routing helpers for the Python autograd boundary."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


_PT2_NAMESPACE = "d2p_pt2"
_PT2_LIBRARY: torch.library.Library | None = None
_PT2_INSTALLED_OPS: tuple[str, ...] = ()


def _interpreter_stack() -> tuple[object, ...]:
    """Return the active functorch interpreter stack, if one exists."""

    stack = torch._C._functorch.get_interpreter_stack()
    return () if stack is None else tuple(stack)


def _transform_active(transform: object) -> bool:
    return any(
        interpreter.key() == transform
        for interpreter in _interpreter_stack()
    )


def functorch_transform_active() -> bool:
    """Return whether execution is inside any torch.func transform."""

    return bool(_interpreter_stack())


def functorch_batching_active() -> bool:
    """Return whether any active torch.func layer is a batching transform."""

    return _transform_active(torch._C._functorch.TransformType.Vmap)


def functorch_tensor_batched(value: object) -> bool:
    """Return whether ``value`` itself carries a functorch batch dimension."""

    if not isinstance(value, Tensor) or pt2_compiling():
        return False
    functorch = torch._C._functorch
    tensor = value
    while functorch.is_gradtrackingtensor(tensor):
        tensor = functorch.get_unwrapped(tensor)
    return functorch.is_batchedtensor(tensor)


def functorch_jvp_active() -> bool:
    """Return whether any active torch.func layer is forward-mode AD."""

    return _transform_active(torch._C._functorch.TransformType.Jvp)


def pt2_compiling() -> bool:
    """Return whether Dynamo/PT2 is currently tracing this Python call."""

    compiler = getattr(torch, "compiler", None)
    if (
        compiler is not None
        and getattr(compiler, "is_compiling", None) is not None
    ):
        if compiler.is_compiling():
            return True
    dynamo = getattr(torch, "_dynamo", None)
    if (
        dynamo is not None
        and getattr(dynamo, "is_compiling", None) is not None
    ):
        if dynamo.is_compiling():
            return True
    return False


def _contains_fake_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
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


def _pt2_dispatch_active(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    return (
        pt2_compiling()
        or _contains_fake_tensor(args)
        or _contains_fake_tensor(kwargs)
    )


def _opaque_schema(name: str, raw: torch._ops.OpOverload) -> str:
    source = f"d2p::{name}"
    schema = str(raw._schema)
    if not schema.startswith(source):
        raise RuntimeError(
            f"unexpected schema name for {source}: {schema}"
        )
    schema = schema.replace(source, name, 1)
    # Keep scalar parameters as tensors until the opaque runtime boundary.
    # This avoids data-dependent SymFloat guards when a public parameter is a
    # differentiable scalar tensor. The runtime implementation converts these
    # mirror-only arguments back to the native schema's concrete floats.
    return schema.replace("float ", "Tensor ")


def _make_runtime_impl(
    original: Any,
    raw: torch._ops.OpOverload,
):
    def runtime(*args: Any, **kwargs: Any):
        # These wrappers are called inside the custom autograd Functions,
        # whose forward/backward bodies own differentiation. Keep the native
        # primitive below its explicit AutogradCPU/AutogradCUDA registration.
        normalized_args = list(args)
        normalized_kwargs = dict(kwargs)
        for index, argument in enumerate(raw._schema.arguments):
            if str(argument.type) != "float":
                continue
            if index < len(normalized_args):
                normalized_args[index] = float(
                    normalized_args[index].detach().item()
                )
            elif argument.name in normalized_kwargs:
                normalized_kwargs[argument.name] = float(
                    normalized_kwargs[argument.name].detach().item()
                )
        with torch._C._AutoDispatchBelowAutograd():
            return original(*normalized_args, **normalized_kwargs)

    return runtime


def _make_fake_impl(
    original: Any,
    raw: torch._ops.OpOverload,
):
    def fake(*args: Any, **kwargs: Any):
        # The existing internal wrapper selects the registered native Meta
        # implementation for FakeTensor inputs while bypassing the explicit
        # native autograd registration. Native Meta kernels use only tensor
        # metadata, so unbacked scalar symbols can use inert placeholders
        # instead of forcing data-dependent guards back into the raw schema.
        normalized_args = list(args)
        normalized_kwargs = dict(kwargs)
        for index, argument in enumerate(raw._schema.arguments):
            if str(argument.type) != "float":
                continue
            if index < len(normalized_args):
                normalized_args[index] = 0.0
            elif argument.name in normalized_kwargs:
                normalized_kwargs[argument.name] = 0.0
        return original(*normalized_args, **normalized_kwargs)

    return fake


def _make_pt2_wrapper(
    original: Any,
    opaque: Any,
    raw: torch._ops.OpOverload,
):
    float_arguments = tuple(
        (index, argument.name)
        for index, argument in enumerate(raw._schema.arguments)
        if str(argument.type) == "float"
    )

    def wrapped(*args: Any, **kwargs: Any):
        if _pt2_dispatch_active(args, kwargs):
            normalized_args = list(args)
            normalized_kwargs = dict(kwargs)
            reference = args[0] if args else None
            for index, argument_name in float_arguments:
                if index < len(normalized_args):
                    value = normalized_args[index]
                    if not isinstance(value, Tensor):
                        if reference is None:
                            raise RuntimeError(
                                "PT2 scalar wrapping requires a tensor input"
                            )
                        normalized_args[index] = reference.new_tensor(value)
                elif argument_name in normalized_kwargs:
                    value = normalized_kwargs[argument_name]
                    if not isinstance(value, Tensor):
                        if reference is None:
                            raise RuntimeError(
                                "PT2 scalar wrapping requires a tensor input"
                            )
                        normalized_kwargs[argument_name] = (
                            reference.new_tensor(value)
                        )
            return opaque(*normalized_args, **normalized_kwargs)
        return original(*args, **kwargs)

    return wrapped


def install_pt2_dispatch(
    native_ops: object,
    op_names: Iterable[str],
) -> None:
    """Install opaque PT2 mirrors for the shipped native primitives.

    The native forward operators have explicit autograd registrations. Dynamo
    used to record those raw targets after a temporary below-autograd guard,
    so Inductor fake propagation later re-entered the autograd kernel with
    storage-less tensors. Each mirror is an opaque custom operator with the
    native container contract and tensorized scalar parameters: FakeTensor
    execution delegates to the already-registered native Meta kernel, while
    CPU/CUDA runtime delegates to the existing native implementation.

    Only compile/FakeTensor calls are rerouted. Eager execution and functorch
    transforms retain the original dispatcher and batching behavior.
    """

    global _PT2_LIBRARY, _PT2_INSTALLED_OPS

    names = tuple(op_names)
    if _PT2_LIBRARY is not None:
        if names != _PT2_INSTALLED_OPS:
            raise RuntimeError(
                "PT2 dispatch was already installed for a different op set"
            )
        return

    library = torch.library.Library(_PT2_NAMESPACE, "DEF")
    originals: dict[str, Any] = {}
    raw_ops: dict[str, torch._ops.OpOverload] = {}
    for name in names:
        raw = getattr(torch.ops.d2p, name).default
        original = getattr(native_ops, name)
        raw_ops[name] = raw
        originals[name] = original
        library.define(_opaque_schema(name, raw))
        runtime = _make_runtime_impl(original, raw)
        library.impl(name, runtime, "CPU")
        library.impl(name, runtime, "CUDA")

    # Keep the Library alive before registering Fake implementations: PyTorch
    # removes registrations when their owning Library is destroyed.
    _PT2_LIBRARY = library
    _PT2_INSTALLED_OPS = names

    for name in names:
        original = originals[name]
        torch.library.register_fake(f"{_PT2_NAMESPACE}::{name}")(
            _make_fake_impl(original, raw_ops[name])
        )
        opaque = getattr(getattr(torch.ops, _PT2_NAMESPACE), name)
        setattr(
            native_ops,
            name,
            _make_pt2_wrapper(original, opaque, raw_ops[name]),
        )


def unwrap_grad_tracking_tensor(tensor: Tensor) -> Tensor:
    """Expose primal storage at a native-kernel boundary.

    Custom ``autograd.Function`` forward methods receive unwrapped primals,
    but their ``jvp`` methods and transform-driven backward methods receive
    GradTrackingTensor wrappers. The shipped native operators cannot inspect
    those wrappers' storage. Removing only grad-tracking layers preserves any
    enclosing BatchedTensor so the registered native batching rules still
    fold that dimension correctly.
    """

    # AOTAutograd already supplies ordinary FakeTensors while tracing a
    # compile-specific Function. Dynamo cannot trace functorch's private
    # ``is_gradtrackingtensor`` builtin, and no unwrapping is needed here.
    if pt2_compiling():
        return tensor

    functorch = torch._C._functorch
    while functorch.is_gradtrackingtensor(tensor):
        tensor = functorch.get_unwrapped(tensor)
    return tensor


__all__ = [
    "functorch_batching_active",
    "functorch_jvp_active",
    "functorch_tensor_batched",
    "functorch_transform_active",
    "install_pt2_dispatch",
    "pt2_compiling",
    "unwrap_grad_tracking_tensor",
]
