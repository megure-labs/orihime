# SPDX-License-Identifier: Apache-2.0
"""Build-free contracts for native wave32/wave64 HIP dispatch."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIP_OPERATORS = (
    "sw",
    "sw_affine",
    "sv_linear",
    "sv_affine",
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
)


def test_all_hip_operators_use_the_native_wavefront_contract() -> None:
    for operator in HIP_OPERATORS:
        source = (ROOT / "src" / operator / "kernels_gpu.hip").read_text()
        assert '#include "common/wavefront.hiph"' in source, operator
        assert not re.search(
            r"(?:WARP|WAVEFRONT)_SIZE\s*=\s*32|#define\s+\w*WARP_SIZE\s+32",
            source,
        ), operator
        assert "0xffffffffULL" not in source, operator


def test_shared_wavefront_contract_selects_the_backend_width() -> None:
    wavefront = (ROOT / "src" / "common" / "wavefront.hiph").read_text()
    reductions = (ROOT / "src" / "common" / "reduce.hiph").read_text()
    hip_utils = (ROOT / "src" / "common" / "hip_utils.h").read_text()

    assert "static_cast<int>(warpSize)" in wavefront
    assert "FULL_WAVE_MASK = ~0ULL" in wavefront
    assert "MAX_WAVEFRONTS_PER_BLOCK = 32" in wavefront
    assert "wavefront_size()" in reductions
    assert "get_current_device_wavefront_size" in hip_utils
    assert "properties.warpSize == 32 || properties.warpSize == 64" in hip_utils


def test_sw_warp_launch_matches_the_active_device() -> None:
    source = (ROOT / "src" / "sw" / "kernels_gpu.hip").read_text()
    assert source.count("get_current_device_wavefront_size()") == 2
    assert "SW_WARPS_PER_BLOCK * SW_WARP_SIZE" not in source
