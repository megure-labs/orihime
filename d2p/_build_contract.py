# SPDX-License-Identifier: Apache-2.0
"""Fail early when a binary artifact and the active PyTorch do not match."""

from __future__ import annotations

from types import ModuleType


def _torch_minor(torch_module: ModuleType) -> str:
    public = str(torch_module.__version__).split("+", 1)[0]
    parts = public.split(".")
    return ".".join(parts[:2])


def _torch_lane(torch_module: ModuleType) -> str:
    cuda_version = torch_module.version.cuda
    if cuda_version is None:
        return "cpu"
    return "cu" + str(cuda_version).replace(".", "")


def validate_build_compatibility(
    torch_module: ModuleType,
    *,
    expected_minor: str | None,
    expected_lane: str | None,
) -> None:
    """Raise before native loading if a packaged ABI tuple is mismatched."""

    actual_minor = _torch_minor(torch_module)
    actual_lane = _torch_lane(torch_module)
    mismatches: list[str] = []
    if expected_minor is not None and actual_minor != expected_minor:
        mismatches.append(
            f"PyTorch minor {actual_minor} (expected {expected_minor})"
        )
    if expected_lane is not None and actual_lane != expected_lane:
        mismatches.append(
            f"PyTorch lane {actual_lane} (expected {expected_lane})"
        )
    if not mismatches:
        return

    expected = "/".join(
        value or "source"
        for value in (expected_minor, expected_lane)
    )
    actual = f"{actual_minor}/{actual_lane}"
    details = "; ".join(mismatches)
    raise ImportError(
        "this d2p binary does not match the active PyTorch: "
        f"{details}. Artifact ABI is {expected}; runtime ABI is {actual}. "
        "Install the exact wheel local version or conda build string documented "
        "for your PyTorch minor and CPU/CUDA lane."
    )
