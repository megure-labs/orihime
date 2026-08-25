# SPDX-License-Identifier: Apache-2.0
"""Artifact ABI identity must fail before native symbol loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from d2p._build_contract import validate_build_compatibility


def _fake_torch(version: str, cuda: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
    )


def test_source_build_has_no_packaged_abi_restriction() -> None:
    validate_build_compatibility(
        _fake_torch("9.7.1+cu999", "99.9"),
        expected_minor=None,
        expected_lane=None,
    )


def test_exact_packaged_abi_is_accepted() -> None:
    validate_build_compatibility(
        _fake_torch("2.13.1+cu130", "13.0"),
        expected_minor="2.13",
        expected_lane="cu130",
    )


@pytest.mark.parametrize(
    ("version", "cuda", "message"),
    (
        ("2.12.0+cu130", "13.0", r"minor 2.12 \(expected 2.13\)"),
        ("2.13.0+cpu", None, r"lane cpu \(expected cu130\)"),
    ),
)
def test_packaged_abi_mismatch_fails_closed(
    version: str,
    cuda: str | None,
    message: str,
) -> None:
    with pytest.raises(ImportError, match=message):
        validate_build_compatibility(
            _fake_torch(version, cuda),
            expected_minor="2.13",
            expected_lane="cu130",
        )
