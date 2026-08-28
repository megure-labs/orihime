// SPDX-License-Identifier: Apache-2.0
// hip_utils.h - HIP utility functions for orihime
//
// Device guards, grid sizing, stream helpers, launch validation.
// Include this in PyTorch binding files that dispatch to HIP kernels.
//
// Usage:
//   #include "common/hip_utils.h"
//   ORIHIME_CUDA_GUARD(scores);
//   int grid = orihime::common::compute_grid_size(n, block_size);

#pragma once
#include <algorithm>              // std::min
#include <hip/hip_runtime.h>
#include <c10/util/Exception.h>   // TORCH_CHECK
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>  // C10_HIP_CHECK, C10_HIP_KERNEL_LAUNCH_CHECK
#include <ATen/hip/HIPContext.h>

// PyTorch 2.12's HIP headers expose the hipified launch check only under the
// CUDA-spelled name. Keep Orihime's HIP-spelled call sites source-compatible
// with both header generations without changing their runtime semantics.
#if !defined(C10_HIP_KERNEL_LAUNCH_CHECK) && \
    defined(C10_CUDA_KERNEL_LAUNCH_CHECK)
#define C10_HIP_KERNEL_LAUNCH_CHECK C10_CUDA_KERNEL_LAUNCH_CHECK
#endif

// __syncwarp() portability shim.
//
// ROCm only introduced __syncwarp in 7.x. src/sw/kernels_gpu.hip is its sole user
// (the warp-parallel fast path), so without this the whole HIP backend requires
// ROCm >= 7 for one operator's optimization. The three builtins below exist in
// 6.x and are exactly what ROCm 7's own amd_warp_sync_functions.h expands to,
// so this is a re-declaration rather than a reimplementation.
// The two guards are deliberately different. hipcc parses every __global__ body
// in BOTH its host and device passes, so the DECLARATION must exist in both or
// the host pass fails on an unresolved call. The amdgcn builtins, however, only
// exist in the device pass. Hence: declare under __HIPCC__, emit the body under
// __HIP_DEVICE_COMPILE__. The host pass gets a valid empty stub it never calls.
#if defined(__HIPCC__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR < 7
__device__ inline void __syncwarp() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#endif
}
#endif

// Host-only headers for the recordStream helpers below. Kept out of device
// (.hip) translation units so hipcc never parses the ATen tensor/allocator API.
#if !defined(__HIPCC__)
#include <initializer_list>
#include <ATen/core/Tensor.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#endif

namespace orihime {
namespace common {

// ============================================================================
// Grid/block sizing utilities
// ============================================================================

inline constexpr int DEFAULT_BLOCK_SIZE = 256;
inline constexpr int MAX_GRID_SIZE = 65535;

// Ceiling division (common pattern for grid sizing)
__host__ __device__ __forceinline__
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Compute grid size, clamped to max
inline int compute_grid_size(int num_elements, int block_size) {
    int grid = ceil_div(num_elements, block_size);
    return std::min(grid, MAX_GRID_SIZE);
}

// ============================================================================
// Device guard helper
// ============================================================================

// RAII device guard - use at start of every HIP op to ensure
// we're on the correct device for multi-GPU scenarios
// Usage: ORIHIME_CUDA_GUARD(scores);
#define ORIHIME_CUDA_GUARD(tensor) \
    c10::hip::HIPGuardMasqueradingAsCUDA device_guard((tensor).device())

// ============================================================================
// Stream helpers
// ============================================================================

inline hipStream_t get_cuda_stream() {
    return at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
}

// ----------------------------------------------------------------------------
// Caching-allocator stream tracking (host translation units only)
// ----------------------------------------------------------------------------
//
// recordStream tells the PyTorch caching allocator that a tensor's storage is
// used on the current HIP stream, so it will not free or reuse that memory
// while orihime kernels (now launched on the current stream, not the default
// stream) are still reading/writing it. This is a correctness requirement once
// launches move off stream 0 and a precondition for HIP-graph capture.
//
// Guarded out of device (.hip) translation units: kernels_gpu.hip includes this
// header only for get_cuda_stream()/the C10 check macros and never calls the
// helpers below, so hipcc never has to parse the ATen tensor/allocator surface.
#if !defined(__HIPCC__)

// Record a single tensor's storage against the current HIP stream.
inline void record_stream_current(const at::Tensor& t) {
    if (t.defined() && t.is_cuda()) {
        const auto current_stream =
            at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        c10::hip::HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA(
            t.storage().data_ptr(),
            c10::hip::HIPStreamMasqueradingAsCUDA(current_stream.unwrap()));
    }
}

// Record several tensors at once: record_streams_current({&a, &b, &c});
inline void record_streams_current(std::initializer_list<const at::Tensor*> tensors) {
    for (const at::Tensor* t : tensors) {
        record_stream_current(*t);
    }
}

#endif  // !defined(__HIPCC__)

// ============================================================================
// Launch validation
// ============================================================================

// Validate launch config before kernel dispatch
inline void check_launch_config(dim3 grid, dim3 block, const char* kernel_name) {
    TORCH_CHECK(grid.x > 0 && grid.y > 0 && grid.z > 0,
        kernel_name, ": grid dimensions must be positive");
    TORCH_CHECK(block.x > 0 && block.y > 0 && block.z > 0,
        kernel_name, ": block dimensions must be positive");
    TORCH_CHECK(block.x * block.y * block.z <= 1024,
        kernel_name, ": block size exceeds 1024 threads");
}

// Simplified 1D launch validation
inline void check_launch_config_1d(int grid, int block, const char* kernel_name) {
    check_launch_config(dim3(grid), dim3(block), kernel_name);
}

}  // namespace common
}  // namespace orihime
