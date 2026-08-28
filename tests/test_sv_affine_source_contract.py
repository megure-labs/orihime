# SPDX-License-Identifier: Apache-2.0
"""Build-free structural checks for the complete sv_affine GPU family."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "src" / "sv_affine" / "kernels_gpu.cu"
HIP = ROOT / "src" / "sv_affine" / "kernels_gpu.hip"
CUDA_HEADER = ROOT / "src" / "sv_affine" / "kernels_gpu.cuh"
HIP_HEADER = ROOT / "src" / "sv_affine" / "kernels_gpu.hiph"

EXPECTED_HOSTS = [
    "sv_affine_forward",
    "sv_affine_backward",
    "sv_affine_hvp",
    "sv_affine_param_grad",
]


def _normalize_hip_to_cuda(source: str) -> str:
    replacements = (
        ("<hip/hip_runtime.h>", "<cuda_runtime.h>"),
        ("<c10/hip/HIPException.h>", "<c10/cuda/CUDAException.h>"),
        ("common/hip_utils.h", "common/cuda_utils.h"),
        ("common/numerics.hiph", "common/numerics.h"),
        ("hipStream_t", "cudaStream_t"),
        ("hipMemsetAsync", "cudaMemsetAsync"),
        ("hipStreamSynchronize", "cudaStreamSynchronize"),
        ("hip_stream", "cuda_stream"),
        ("0xffffffffULL", "0xffffffff"),
        ("C10_HIP", "C10_CUDA"),
        (".hip", ".cu"),
        ("HIP", "CUDA"),
    )
    for hip_spelling, cuda_spelling in replacements:
        source = source.replace(hip_spelling, cuda_spelling)
    return source


def _host_symbols(path: Path) -> list[str]:
    return re.findall(r"void\s+(sv_affine_\w+)\s*\(", path.read_text())


def _device_symbols(path: Path) -> list[str]:
    return re.findall(r"__global__\s+void\s+(sv_affine_\w+)\s*\(", path.read_text())


def test_cuda_and_hip_expose_the_same_complete_launch_surface():
    cuda = CUDA.read_text()
    hip = HIP.read_text()
    assert _host_symbols(CUDA_HEADER) == EXPECTED_HOSTS
    assert _host_symbols(HIP_HEADER) == EXPECTED_HOSTS
    assert _device_symbols(CUDA) == _device_symbols(HIP)
    assert all(cuda.count(f"{symbol}<<<") >= 1 for symbol in _device_symbols(CUDA))
    assert all(hip.count(f"{symbol}<<<") >= 1 for symbol in _device_symbols(HIP))
    assert "cudaMemsetAsync(d_U" in cuda
    assert "hipMemsetAsync(d_U" in hip
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in cuda
    assert "C10_HIP_KERNEL_LAUNCH_CHECK" in hip


def test_hip_kernel_is_a_runtime_only_port_of_the_cuda_recurrence():
    assert _normalize_hip_to_cuda(HIP.read_text()) == CUDA.read_text()
