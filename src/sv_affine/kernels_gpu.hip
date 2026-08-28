// SPDX-License-Identifier: Apache-2.0
// soft_sv_affine.hip
//
// Direct Affine Gap Soft Smith-Waterman in HIP
// Three-state DP (M, I, D) with affine gap penalties for local alignment
//
// Operations:
//   - Forward:  Compute partition function S = logsumexp over all alignments
//   - Backward: Compute all gradients (dS/dscores, dS/dgap_open, dS/dgap_ext, dS/dT)
//   - HVP:      Hessian-vector product d^2S/dscores^2 * V (forward-mode)
//
// States:
//   M (Match):     Aligned positions (diagonal transition + score)
//   I (Insertion): Gap in sequence 1 (vertical transition)
//   D (Deletion):  Gap in sequence 2 (horizontal transition)
//
// Shapes:
//   scores:      [B, L1, L2]           - similarity scores
//   alpha:       [B, 3*(L1+1)*(L2+1)]  - DP tables for M,I,D (stacked)
//   partition:   [B]                   - partition function values
//   posteriors:  [B, L1, L2]           - alignment marginals (dS/dscores)
//   grad_open:   [B]                   - expected number of gap-open events
//   grad_ext:    [B]                   - expected number of gap-extend events
//   grad_T:      [B]                   - temperature gradient

#include <hip/hip_runtime.h>
#include <math.h>

// PyTorch HIP exception/check macros: C10_HIP_CHECK and
// C10_HIP_KERNEL_LAUNCH_CHECK localize runtime/launch failures to this op
// boundary instead of letting them surface later as poisoned-context errors.
#include <c10/hip/HIPException.h>

// Shared utilities
#include "common/hip_utils.h"
#include "common/numerics.hiph"

using namespace orihime::common;

namespace {

constexpr int SV_AFFINE_WARP_SIZE = 32;

__device__ __forceinline__ void sv_affine_warp_reduce_sum(KahanSum& acc) {
    int lane = threadIdx.x % SV_AFFINE_WARP_SIZE;

    #pragma unroll
    for (int offset = SV_AFFINE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float peer_sum = __shfl_down_sync(0xffffffffULL, acc.sum, offset);
        float peer_c = __shfl_down_sync(0xffffffffULL, acc.c, offset);
        if (lane + offset < SV_AFFINE_WARP_SIZE) {
            acc.add(peer_sum);
            acc.add(-peer_c);
        }
    }
}

__device__ __forceinline__ float sv_affine_block_reduce_sum(
    const KahanSum& local_acc
) {
    __shared__ float shared_sum[32];
    __shared__ float shared_c[32];

    int lane = threadIdx.x % SV_AFFINE_WARP_SIZE;
    int warp_id = threadIdx.x / SV_AFFINE_WARP_SIZE;

    KahanSum warp_acc;
    warp_acc.sum = local_acc.sum;
    warp_acc.c = local_acc.c;
    sv_affine_warp_reduce_sum(warp_acc);

    if (lane == 0) {
        shared_sum[warp_id] = warp_acc.sum;
        shared_c[warp_id] = warp_acc.c;
    }
    __syncthreads();

    int num_warps = blockDim.x / SV_AFFINE_WARP_SIZE;
    KahanSum block_acc;
    if (threadIdx.x < num_warps) {
        block_acc.sum = shared_sum[lane];
        block_acc.c = shared_c[lane];
    }
    if (warp_id == 0) {
        sv_affine_warp_reduce_sum(block_acc);
    }

    // Prevent a back-to-back reduction from overwriting the shared partials
    // before every thread has finished reading them.
    __syncthreads();
    return block_acc.result();
}

__device__ __forceinline__ float sv_affine_warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = SV_AFFINE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffULL, value, offset)
        );
    }
    return value;
}

__device__ __forceinline__ float sv_affine_block_reduce_max(float value) {
    __shared__ float shared[32];

    int lane = threadIdx.x % SV_AFFINE_WARP_SIZE;
    int warp_id = threadIdx.x / SV_AFFINE_WARP_SIZE;

    value = sv_affine_warp_reduce_max(value);
    if (lane == 0) {
        shared[warp_id] = value;
    }
    __syncthreads();

    int num_warps = blockDim.x / SV_AFFINE_WARP_SIZE;
    value = (threadIdx.x < num_warps) ? shared[lane] : NINF;
    if (warp_id == 0) {
        value = sv_affine_warp_reduce_max(value);
    }
    return value;
}

__device__ __forceinline__ float kahan_sum2(float v1, float v2) {
    KahanSum sum;
    sum.add(v1);
    sum.add(v2);
    return sum.result();
}

__device__ __forceinline__ float kahan_sum3(
    float v1, float v2, float v3
) {
    KahanSum sum;
    sum.add(v1);
    sum.add(v2);
    sum.add(v3);
    return sum.result();
}

__device__ __forceinline__ float kahan_sum4(
    float v1, float v2, float v3, float v4
) {
    KahanSum sum;
    sum.add(v1);
    sum.add(v2);
    sum.add(v3);
    sum.add(v4);
    return sum.result();
}

}  // namespace

// Parameter types for param gradient kernels
enum ParamType {
    PARAM_GAP_OPEN = 0,
    PARAM_GAP_EXT = 1,
    PARAM_TEMPERATURE = 2
};

// ============================================================================
// Soft-max Helpers (for code reuse across forward/backward/HVP)
// These return logsumexp value + weights, different from common/softmax.cuh
// ============================================================================

// Compute soft-max weights for 2 logits, return logsumexp value
__device__ __forceinline__ float softmax2_with_lse(
    float v1, float v2,
    float T,
    float& w1, float& w2
) {
    float max_v = fmaxf(v1, v2);
    if (max_v <= NINF) {
        w1 = w2 = 0.0f;
        return NINF;
    }
    w1 = safe_exp((v1 - max_v) / T);
    w2 = safe_exp((v2 - max_v) / T);
    float sum_w = kahan_sum2(w1, w2);
    float inv = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;
    w1 *= inv;
    w2 *= inv;
    return max_v + T * logf(fmaxf(sum_w, 1e-30f));
}

// Compute soft-max weights for 3 logits, return logsumexp value
__device__ __forceinline__ float softmax3_with_lse(
    float v1, float v2, float v3,
    float T,
    float& w1, float& w2, float& w3
) {
    float max_v = fmaxf(v1, fmaxf(v2, v3));
    if (max_v <= NINF) {
        w1 = w2 = w3 = 0.0f;
        return NINF;
    }
    w1 = safe_exp((v1 - max_v) / T);
    w2 = safe_exp((v2 - max_v) / T);
    w3 = safe_exp((v3 - max_v) / T);
    float sum_w = kahan_sum3(w1, w2, w3);
    float inv = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;
    w1 *= inv;
    w2 *= inv;
    w3 *= inv;
    return max_v + T * logf(fmaxf(sum_w, 1e-30f));
}

// Compute soft-max weights for 4 logits, return logsumexp value
__device__ __forceinline__ float softmax4_with_lse(
    float v1, float v2, float v3, float v4,
    float T,
    float& w1, float& w2, float& w3, float& w4
) {
    float max_v = fmaxf(fmaxf(v1, v2), fmaxf(v3, v4));
    if (max_v <= NINF) {
        w1 = w2 = w3 = w4 = 0.0f;
        return NINF;
    }
    w1 = safe_exp((v1 - max_v) / T);
    w2 = safe_exp((v2 - max_v) / T);
    w3 = safe_exp((v3 - max_v) / T);
    w4 = safe_exp((v4 - max_v) / T);
    float sum_w = kahan_sum4(w1, w2, w3, w4);
    float inv = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;
    w1 *= inv;
    w2 *= inv;
    w3 *= inv;
    w4 *= inv;
    return max_v + T * logf(fmaxf(sum_w, 1e-30f));
}

// Compute tangent of 2-way soft-max weights (softmax Jacobian)
// Returns tangent of logsumexp: d_out = sum_k w_k * dv_k
__device__ __forceinline__ float softmax2_tangent(
    float w1, float w2,
    float dv1, float dv2,
    float T,
    float& dw1, float& dw2
) {
    float sum_w_dv = kahan_sum2(w1 * dv1, w2 * dv2);
    dw1 = w1 * (dv1 - sum_w_dv) / T;
    dw2 = w2 * (dv2 - sum_w_dv) / T;
    return sum_w_dv;  // tangent of logsumexp output
}

// Compute tangent of 3-way soft-max weights (softmax Jacobian)
__device__ __forceinline__ float softmax3_tangent(
    float w1, float w2, float w3,
    float dv1, float dv2, float dv3,
    float T,
    float& dw1, float& dw2, float& dw3
) {
    float sum_w_dv = kahan_sum3(w1 * dv1, w2 * dv2, w3 * dv3);
    dw1 = w1 * (dv1 - sum_w_dv) / T;
    dw2 = w2 * (dv2 - sum_w_dv) / T;
    dw3 = w3 * (dv3 - sum_w_dv) / T;
    return sum_w_dv;
}

// Compute tangent of 4-way soft-max weights (softmax Jacobian)
// Returns tangent of logsumexp: d_out = sum_k w_k * dv_k
__device__ __forceinline__ float softmax4_tangent(
    float w1, float w2, float w3, float w4,
    float dv1, float dv2, float dv3, float dv4,
    float T,
    float& dw1, float& dw2, float& dw3, float& dw4
) {
    float sum_w_dv = kahan_sum4(
        w1 * dv1,
        w2 * dv2,
        w3 * dv3,
        w4 * dv4
    );
    dw1 = w1 * (dv1 - sum_w_dv) / T;
    dw2 = w2 * (dv2 - sum_w_dv) / T;
    dw3 = w3 * (dv3 - sum_w_dv) / T;
    dw4 = w4 * (dv4 - sum_w_dv) / T;
    return sum_w_dv;  // tangent of logsumexp output
}

__device__ __forceinline__ bool has_finite_nonzero_signal(float value) {
    return isfinite(value) && value != 0.0f;
}

// ============================================================================
// FORWARD PASS
// ============================================================================

// Initialize alpha: M[0,0] = 0, all others = -inf
__global__ void sv_affine_init_alpha_kernel(
    float* __restrict__ alpha,
    int B, int L1, int L2
) {
    size_t stride_state = (size_t)(L1 + 1) * (L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride_all;
    if (idx >= total) return;

    size_t b     = idx / stride_all;
    size_t rem   = idx - b * stride_all;
    int state = (int)(rem / stride_state);
    size_t off = rem - (size_t)state * stride_state;
    size_t i = off / (size_t)(L2 + 1);
    size_t j = off % (size_t)(L2 + 1);

    // M[0,0] = 0 for local alignment start
    if (state == 0 && i == 0 && j == 0) {
        alpha[idx] = 0.0f;
    } else {
        alpha[idx] = NINF;
    }
}

// Forward DP for one anti-diagonal
// M state: soft-max over {M[i-1,j-1]+s, I[i-1,j-1]+s, D[i-1,j-1]+s, s (sky)}
// I state: soft-max over {M[i-1,j]+gap_open, I[i-1,j]+gap_ext}
// D state: soft-max over {M[i,j-1]+gap_open, I[i,j-1]+gap_open, D[i,j-1]+gap_ext}
__global__ void sv_affine_forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];      // actual length for this batch
    int L2 = lengths[b * 2 + 1];

    size_t stride_state  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all    = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t stride_row    = (size_t)max_L2 + 1;
    size_t score_row     = (size_t)max_L2;

    float* M = alpha + (size_t)b * stride_all;
    float* I = M + stride_state;
    float* D = I + stride_state;
    const float* s = scores + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride_row + (size_t)j;
            M[idx] = NINF;
            I[idx] = NINF;
            D[idx] = NINF;
            continue;
        }

        size_t idx       = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left  = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float score = s[score_idx];

        // ========== M state ==========
        // From M/I/D at diagonal + score, or sky (fresh start with score)
        float m1 = (i > 1 && j > 1) ? M[idx_diag] + score : NINF;
        float m2 = (i > 1 && j > 1) ? I[idx_diag] + score : NINF;
        float m3 = (i > 1 && j > 1) ? D[idx_diag] + score : NINF;
        float m4 = score;  // sky: local alignment can start fresh
        float wm1, wm2, wm3, wm4;
        M[idx] = softmax4_with_lse(m1, m2, m3, m4, T, wm1, wm2, wm3, wm4);

        // ========== I state (gap in seq1, vertical) ==========
        // From M[i-1,j] + gap_open or I[i-1,j] + gap_ext
        float i1 = (i > 1) ? M[idx_up] + gap_open : NINF;
        float i2 = (i > 1) ? I[idx_up] + gap_ext  : NINF;
        float wi1, wi2;
        I[idx] = softmax2_with_lse(i1, i2, T, wi1, wi2);

        // ========== D state (gap in seq2, horizontal) ==========
        // Canonical one-way I->D transition enumerates both-sided skips once.
        float d1 = (j > 1) ? M[idx_left] + gap_open : NINF;
        float d2 = (j > 1) ? I[idx_left] + gap_open : NINF;
        float d3 = (j > 1) ? D[idx_left] + gap_ext  : NINF;
        float wd1, wd2, wd3;
        D[idx] = softmax3_with_lse(d1, d2, d3, T, wd1, wd2, wd3);
    }
}

// Canonical partition: LSE_T(0, all true M cells). The 0 is the empty alignment.
__global__ void sv_affine_partition_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ partition,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2, float T
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t stride_row   = (size_t)max_L2 + 1;

    const float* M = alpha + (size_t)b * stride_all;
    // Find max over true match cells and the single empty-alignment term.
    float local_max = 0.0f;
    size_t valid_cells = (size_t)L1 * (size_t)L2;
    for (size_t t = threadIdx.x; t < valid_cells; t += blockDim.x) {
        size_t i = t / (size_t)L2 + 1;
        size_t j = t % (size_t)L2 + 1;
        size_t idx = i * stride_row + j;
        local_max = fmaxf(local_max, M[idx]);
    }
    float max_val = sv_affine_block_reduce_max(local_max);

    __shared__ float sh_max;
    if (threadIdx.x == 0) sh_max = max_val;
    __syncthreads();
    max_val = sh_max;

    // Sum exp over true M cells plus exactly one empty-alignment term.
    KahanSum local_sum;
    if (threadIdx.x == 0) {
        local_sum.add(safe_exp(-max_val / T));
    }
    for (size_t t = threadIdx.x; t < valid_cells; t += blockDim.x) {
        size_t i = t / (size_t)L2 + 1;
        size_t j = t % (size_t)L2 + 1;
        size_t idx = i * stride_row + j;
        local_sum.add(safe_exp((M[idx] - max_val) / T));
    }
    float sum_exp = sv_affine_block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        partition[b] = max_val + T * logf(sum_exp);
    }
}

// ============================================================================
// BACKWARD PASS - All Gradients
// ============================================================================

// Initialize beta = dS/dalpha for valid cells only
__global__ void sv_affine_init_beta_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ partition,
    float* __restrict__ beta,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2, float T
) {
    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t stride_row   = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride_all;
    if (idx >= total) return;

    size_t b     = idx / stride_all;
    size_t rem   = idx - b * stride_all;
    int state = (int)(rem / stride_state);
    size_t off = rem - (size_t)state * stride_state;
    size_t i = off / stride_row;
    size_t j = off % stride_row;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Only true M cells terminate; I/D and boundary sentinels have no terminal seed.
    if (state != 0 || i == 0 || j == 0 || i > (size_t)L1 || j > (size_t)L2) {
        beta[idx] = 0.0f;
        return;
    }

    float a = alpha[idx];
    float S = partition[b];

    // Fix #10: Posterior clamping - clamp log-posterior to <= 0 to ensure beta <= 1
    // Numerical errors can cause alpha > S slightly, resulting in posteriors > 1
    if (a > NINF) {
        float log_post = (a - S) / T;
        log_post = fminf(log_post, 0.0f);  // Clamp to <= 0
        beta[idx] = safe_exp(log_post);
    } else {
        beta[idx] = 0.0f;
    }
}

// Backward DP: propagate beta through all states, accumulate gradients
__global__ void sv_affine_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_open,
    float* __restrict__ grad_ext,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all    = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t stride_row    = (size_t)max_L2 + 1;
    size_t score_row     = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;
    const float* s   = scores + (size_t)b * stride_scores;

    float* B_M = beta + (size_t)b * stride_all;
    float* B_I = B_M + stride_state;
    float* B_D = B_I + stride_state;

    float* g_scores = posteriors + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    KahanSum local_open;
    KahanSum local_ext;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left  = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float score = s[score_idx];

        float betaM = B_M[idx];
        float betaI = B_I[idx];
        float betaD = B_D[idx];

        // ========== M state backprop ==========
        if (has_finite_nonzero_signal(betaM)) {
            float m1 = (i > 1 && j > 1) ? A_M[idx_diag] + score : NINF;
            float m2 = (i > 1 && j > 1) ? A_I[idx_diag] + score : NINF;
            float m3 = (i > 1 && j > 1) ? A_D[idx_diag] + score : NINF;
            float m4 = score;
            float w1, w2, w3, w4;
            if (softmax4_with_lse(m1, m2, m3, m4, T, w1, w2, w3, w4) > NINF) {
                // Propagate to predecessor states
                if (m1 > NINF) atomicAdd(&B_M[idx_diag], betaM * w1);
                if (m2 > NINF) atomicAdd(&B_I[idx_diag], betaM * w2);
                if (m3 > NINF) atomicAdd(&B_D[idx_diag], betaM * w3);
                // Score gradient: all transitions into M use score
                // betaM * (w1 + w2 + w3 + w4) = betaM * 1 = betaM
                atomicAdd(&g_scores[score_idx], betaM);
            }
        }

        // ========== I state backprop ==========
        if (has_finite_nonzero_signal(betaI) && i > 1) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float w1, w2;
            if (softmax2_with_lse(i1, i2, T, w1, w2) > NINF) {
                atomicAdd(&B_M[idx_up], betaI * w1);
                atomicAdd(&B_I[idx_up], betaI * w2);
                // Gap gradients
                local_open.add(betaI * w1);  // M->I is gap open
                local_ext.add(betaI * w2);   // I->I is gap extend
            }
        }

        // ========== D state backprop ==========
        if (has_finite_nonzero_signal(betaD) && j > 1) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;
            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                atomicAdd(&B_M[idx_left], betaD * w1);
                atomicAdd(&B_I[idx_left], betaD * w2);
                atomicAdd(&B_D[idx_left], betaD * w3);
                // Gap gradients
                local_open.add(betaD * w1);  // M->D is gap open
                local_open.add(betaD * w2);  // I->D is gap open
                local_ext.add(betaD * w3);   // D->D is gap extend
            }
        }
    }

    // Block-reduce gap gradients
    float block_open = sv_affine_block_reduce_sum(local_open);
    float block_ext  = sv_affine_block_reduce_sum(local_ext);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_open[b], block_open);
        atomicAdd(&grad_ext[b],  block_ext);
    }
}

// Compute dS/dT (posteriors are already masked from backward pass)
__global__ void sv_affine_grad_T_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ posteriors,
    const float* __restrict__ grad_open,
    const float* __restrict__ grad_ext,
    const float* __restrict__ partition,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t score_row = (size_t)max_L2;

    const float* p = posteriors + (size_t)b * stride_scores;
    const float* s = scores + (size_t)b * stride_scores;

    // E[match_score] - only sum over valid cells
    KahanSum local_sum;
    size_t valid_size = (size_t)L1 * (size_t)L2;
    for (size_t t = threadIdx.x; t < valid_size; t += blockDim.x) {
        size_t i = t / (size_t)L2;
        size_t j = t % (size_t)L2;
        size_t idx = i * score_row + j;
        local_sum.add(p[idx] * s[idx]);
    }
    float match_sum = sv_affine_block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        float expected_gap_cost = kahan_sum2(
            grad_open[b] * gap_open,
            grad_ext[b] * gap_ext
        );
        float expected_total = kahan_sum2(match_sum, expected_gap_cost);
        grad_T[b] = (partition[b] - expected_total) / T;
    }
}

// ============================================================================
// HVP (Hessian-Vector Product) - Forward Mode
// ============================================================================

// Forward-mode: propagate tangents through affine forward DP
__global__ void sv_affine_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all    = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t stride_row    = (size_t)max_L2 + 1;
    size_t score_row     = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;
    const float* s   = scores + (size_t)b * stride_scores;
    const float* v   = V + (size_t)b * stride_scores;

    float* dA_M = d_alpha + (size_t)b * stride_all;
    float* dA_I = dA_M + stride_state;
    float* dA_D = dA_I + stride_state;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride_row + (size_t)j;
            dA_M[idx] = 0.0f;
            dA_I[idx] = 0.0f;
            dA_D[idx] = 0.0f;
            continue;
        }

        size_t idx       = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left  = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float score = s[score_idx];
        float V_s   = v[score_idx];

        // ========== M state tangent ==========
        {
            float m1 = (i > 1 && j > 1) ? A_M[idx_diag] + score : NINF;
            float m2 = (i > 1 && j > 1) ? A_I[idx_diag] + score : NINF;
            float m3 = (i > 1 && j > 1) ? A_D[idx_diag] + score : NINF;
            float m4 = score;
            float w1, w2, w3, w4;
            if (softmax4_with_lse(m1, m2, m3, m4, T, w1, w2, w3, w4) > NINF) {
                // Tangent of logits
                float dm1 = (i > 1 && j > 1) ? dA_M[idx_diag] + V_s : 0.0f;
                float dm2 = (i > 1 && j > 1) ? dA_I[idx_diag] + V_s : 0.0f;
                float dm3 = (i > 1 && j > 1) ? dA_D[idx_diag] + V_s : 0.0f;
                float dm4 = V_s;
                // Forward-mode: d_out = sum_k w_k * d_logit_k
                dA_M[idx] = kahan_sum4(
                    w1 * dm1,
                    w2 * dm2,
                    w3 * dm3,
                    w4 * dm4
                );
            } else {
                dA_M[idx] = 0.0f;
            }
        }

        // ========== I state tangent ==========
        if (i > 1) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float w1, w2;
            if (softmax2_with_lse(i1, i2, T, w1, w2) > NINF) {
                float di1 = dA_M[idx_up];
                float di2 = dA_I[idx_up];
                dA_I[idx] = kahan_sum2(w1 * di1, w2 * di2);
            } else {
                dA_I[idx] = 0.0f;
            }
        } else {
            dA_I[idx] = 0.0f;
        }

        // ========== D state tangent ==========
        if (j > 1) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;
            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                float dd1 = dA_M[idx_left];
                float dd2 = dA_I[idx_left];
                float dd3 = dA_D[idx_left];
                dA_D[idx] = kahan_sum3(
                    w1 * dd1,
                    w2 * dd2,
                    w3 * dd3
                );
            } else {
                dA_D[idx] = 0.0f;
            }
        } else {
            dA_D[idx] = 0.0f;
        }
    }
}

// Compute tangent of partition function (only over valid cells)
__global__ void sv_affine_hvp_partition_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ d_alpha,
    const float* __restrict__ partition,
    float* __restrict__ d_partition,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2, float T
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t stride_row   = (size_t)max_L2 + 1;

    const float* A_base  = alpha + (size_t)b * stride_all;
    const float* dA_base = d_alpha + (size_t)b * stride_all;
    float S = partition[b];

    // Sum over canonical M-only terminal cells. The empty term has zero tangent.
    KahanSum local_sum;
    size_t valid_cells = (size_t)L1 * (size_t)L2;
    for (size_t t = threadIdx.x; t < valid_cells; t += blockDim.x) {
        size_t i = t / (size_t)L2 + 1;
        size_t j = t % (size_t)L2 + 1;
        size_t idx = i * stride_row + j;
        if (A_base[idx] > NINF) {
            float log_post = fminf((A_base[idx] - S) / T, 0.0f);
            local_sum.add(safe_exp(log_post) * dA_base[idx]);
        }
    }
    float sum = sv_affine_block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        d_partition[b] = sum;
    }
}

// Backward pass of forward-mode HVP
__global__ void sv_affine_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ partition,
    const float* __restrict__ d_alpha,
    const float* __restrict__ d_partition,
    const float* __restrict__ V,
    float* __restrict__ beta,      // accumulated beta (like regular backward)
    float* __restrict__ d_beta,    // tangent of beta
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all    = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t stride_row    = (size_t)max_L2 + 1;
    size_t score_row     = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;
    const float* s   = scores + (size_t)b * stride_scores;
    const float* v   = V + (size_t)b * stride_scores;

    const float* dA_M = d_alpha + (size_t)b * stride_all;
    const float* dA_I = dA_M + stride_state;
    const float* dA_D = dA_I + stride_state;

    // Accumulated beta buffers (like regular backward)
    float* B_M = beta + (size_t)b * stride_all;
    float* B_I = B_M + stride_state;
    float* B_D = B_I + stride_state;

    // Tangent of beta buffers
    float* dB_M = d_beta + (size_t)b * stride_all;
    float* dB_I = dB_M + stride_state;
    float* dB_D = dB_I + stride_state;

    float* H = H_scores + (size_t)b * stride_scores;

    float S  = partition[b];
    float dS = d_partition[b];

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left  = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float score = s[score_idx];
        float V_s   = v[score_idx];

        // ========== M state ==========
        // Read ACCUMULATED beta (like regular backward)
        float betaM = B_M[idx];
        // Tangent of beta initialization: db_init = beta_init * (dA - dS) / T
        // Fix #10: Apply posterior clamping
        float beta_initM = 0.0f;
        if (A_M[idx] > NINF) {
            float log_post = (A_M[idx] - S) / T;
            log_post = fminf(log_post, 0.0f);  // Clamp to <= 0
            beta_initM = safe_exp(log_post);
        }
        float db_initM = beta_initM * (dA_M[idx] - dS) / T;
        // Total d_beta = accumulated from successors + initial tangent
        float dbM = kahan_sum2(dB_M[idx], db_initM);

        // Skip only when both signals are exactly zero or non-finite.
        if (has_finite_nonzero_signal(betaM) || has_finite_nonzero_signal(dbM)) {
            float m1 = (i > 1 && j > 1) ? A_M[idx_diag] + score : NINF;
            float m2 = (i > 1 && j > 1) ? A_I[idx_diag] + score : NINF;
            float m3 = (i > 1 && j > 1) ? A_D[idx_diag] + score : NINF;
            float m4 = score;
            float w1, w2, w3, w4;

            if (softmax4_with_lse(m1, m2, m3, m4, T, w1, w2, w3, w4) > NINF) {
                // Tangent of logits (all include V_s since all use score)
                float dm1 = (i > 1 && j > 1) ? dA_M[idx_diag] + V_s : 0.0f;
                float dm2 = (i > 1 && j > 1) ? dA_I[idx_diag] + V_s : 0.0f;
                float dm3 = (i > 1 && j > 1) ? dA_D[idx_diag] + V_s : 0.0f;
                float dm4 = V_s;
                float dw1, dw2, dw3, dw4;
                softmax4_tangent(w1, w2, w3, w4, dm1, dm2, dm3, dm4, T, dw1, dw2, dw3, dw4);

                // Propagate BOTH beta and d_beta to predecessors
                // d_beta[pred] += db_curr * w + beta_curr * dw
                if (m1 > NINF) {
                    atomicAdd(&B_M[idx_diag], betaM * w1);
                    atomicAdd(
                        &dB_M[idx_diag],
                        kahan_sum2(dbM * w1, betaM * dw1)
                    );
                }
                if (m2 > NINF) {
                    atomicAdd(&B_I[idx_diag], betaM * w2);
                    atomicAdd(
                        &dB_I[idx_diag],
                        kahan_sum2(dbM * w2, betaM * dw2)
                    );
                }
                if (m3 > NINF) {
                    atomicAdd(&B_D[idx_diag], betaM * w3);
                    atomicAdd(
                        &dB_D[idx_diag],
                        kahan_sum2(dbM * w3, betaM * dw3)
                    );
                }

                // H_scores contribution: d(posterior) = d(betaM * sum_k w_k * uses_score_k)
                // For M state: all 4 transitions use score, so:
                //   posterior = betaM * (w1 + w2 + w3 + w4) = betaM * 1 = betaM
                //   d(posterior) = dbM * 1 + betaM * (dw1 + dw2 + dw3 + dw4)
                //                = dbM + betaM * 0  (since sum of dw = 0 for softmax)
                //                = dbM
                atomicAdd(&H[score_idx], dbM);
            }
        }

        // ========== I state ==========
        // Read ACCUMULATED beta
        float betaI = B_I[idx];
        float dbI = dB_I[idx];

        if ((has_finite_nonzero_signal(betaI) || has_finite_nonzero_signal(dbI)) && i > 1) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float w1, w2;

            if (softmax2_with_lse(i1, i2, T, w1, w2) > NINF) {
                // Tangent of logits (no V contribution - gap params are constant)
                float di1 = dA_M[idx_up];
                float di2 = dA_I[idx_up];
                float dw1, dw2;
                softmax2_tangent(w1, w2, di1, di2, T, dw1, dw2);

                // Propagate BOTH beta and d_beta
                atomicAdd(&B_M[idx_up], betaI * w1);
                atomicAdd(
                    &dB_M[idx_up],
                    kahan_sum2(dbI * w1, betaI * dw1)
                );
                atomicAdd(&B_I[idx_up], betaI * w2);
                atomicAdd(
                    &dB_I[idx_up],
                    kahan_sum2(dbI * w2, betaI * dw2)
                );
                // No H_scores contribution - I state doesn't use score
            }
        }

        // ========== D state ==========
        // Read ACCUMULATED beta
        float betaD = B_D[idx];
        float dbD = dB_D[idx];

        if ((has_finite_nonzero_signal(betaD) || has_finite_nonzero_signal(dbD)) && j > 1) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;

            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                // Tangent of logits (no V contribution - gap params are constant)
                float dd1 = dA_M[idx_left];
                float dd2 = dA_I[idx_left];
                float dd3 = dA_D[idx_left];
                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, dd1, dd2, dd3, T, dw1, dw2, dw3);

                // Propagate BOTH beta and d_beta
                atomicAdd(&B_M[idx_left], betaD * w1);
                atomicAdd(
                    &dB_M[idx_left],
                    kahan_sum2(dbD * w1, betaD * dw1)
                );
                atomicAdd(&B_I[idx_left], betaD * w2);
                atomicAdd(
                    &dB_I[idx_left],
                    kahan_sum2(dbD * w2, betaD * dw2)
                );
                atomicAdd(&B_D[idx_left], betaD * w3);
                atomicAdd(
                    &dB_D[idx_left],
                    kahan_sum2(dbD * w3, betaD * dw3)
                );
                // No H_scores contribution - D state doesn't use score
            }
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT KERNELS (d^2S/dscores/dtheta)
// ============================================================================

// Forward U pass: compute U = dalpha/dtheta for parameter theta
// Same wavefront pattern as forward DP, but propagates tangent instead of alpha
__global__ void sv_affine_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag,
    int param_type  // 0=gap_open, 1=gap_ext, 2=temperature
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all    = 3 * stride_state;
    size_t stride_row    = (size_t)max_L2 + 1;
    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t score_row     = (size_t)max_L2;

    // Alpha arrays (for computing weights)
    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    // Scores
    const float* s = scores + (size_t)b * stride_scores;

    // U arrays (tangent of alpha)
    float* U_M = U + (size_t)b * stride_all;
    float* U_I = U_M + stride_state;
    float* U_D = U_I + stride_state;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride_row + (size_t)j;
            U_M[idx] = 0.0f;
            U_I[idx] = 0.0f;
            U_D[idx] = 0.0f;
            continue;
        }

        size_t idx       = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left  = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);
        float score = s[score_idx];

        // ========== M state U ==========
        // M transitions: soft-max over {M/I/D diagonal + score, sky=score}
        {
            float m1 = (i > 1 && j > 1) ? A_M[idx_diag] + score : NINF;
            float m2 = (i > 1 && j > 1) ? A_I[idx_diag] + score : NINF;
            float m3 = (i > 1 && j > 1) ? A_D[idx_diag] + score : NINF;
            float m4 = score;  // sky: local alignment start
            float w1, w2, w3, w4;
            float lse = softmax4_with_lse(
                m1, m2, m3, m4, T,
                w1, w2, w3, w4
            );

            if (lse <= NINF) {
                U_M[idx] = 0.0f;
            } else {
                // Tangent: U_M = sum_k w_k * U_pred[k]
                // (no direct param contribution to M state)
                float u1 = (i > 1 && j > 1) ? U_M[idx_diag] : 0.0f;
                float u2 = (i > 1 && j > 1) ? U_I[idx_diag] : 0.0f;
                float u3 = (i > 1 && j > 1) ? U_D[idx_diag] : 0.0f;
                float u4 = 0.0f;  // sky has no predecessor

                float dU_M = kahan_sum4(
                    w1 * u1,
                    w2 * u2,
                    w3 * u3,
                    w4 * u4
                );

                // For temperature: add (alpha - E[val]) / T term
                if (param_type == PARAM_TEMPERATURE) {
                    float alpha_M = A_M[idx];
                    float E_val = kahan_sum4(
                        w1 * m1,
                        w2 * m2,
                        w3 * m3,
                        w4 * m4
                    );
                    dU_M = kahan_sum2(dU_M, (alpha_M - E_val) / T);
                }

                U_M[idx] = dU_M;
            }
        }

        // ========== I state U ==========
        // I state: softmax2_with_lse(M[i-1,j] + gap_open, I[i-1,j] + gap_ext)
        if (i > 1) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float w1, w2;
            float lse = softmax2_with_lse(i1, i2, T, w1, w2);

            if (lse > NINF) {
                float u1 = U_M[idx_up];
                float u2 = U_I[idx_up];

                // Direct param contributions
                float direct1 = (param_type == PARAM_GAP_OPEN) ? 1.0f : 0.0f;
                float direct2 = (param_type == PARAM_GAP_EXT) ? 1.0f : 0.0f;

                float dU_I = kahan_sum2(
                    w1 * (u1 + direct1),
                    w2 * (u2 + direct2)
                );

                // For temperature: add (alpha - E[val]) / T term
                if (param_type == PARAM_TEMPERATURE) {
                    float alpha_I = A_I[idx];
                    float E_val = kahan_sum2(w1 * i1, w2 * i2);
                    dU_I = kahan_sum2(dU_I, (alpha_I - E_val) / T);
                }

                U_I[idx] = dU_I;
            } else {
                U_I[idx] = 0.0f;
            }
        } else {
            U_I[idx] = 0.0f;
        }

        // ========== D state U ==========
        // D state: canonical one-way I->D cross transition is also a gap open.
        if (j > 1) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;
            float lse = softmax3_with_lse(d1, d2, d3, T, w1, w2, w3);

            if (lse > NINF) {
                float u1 = U_M[idx_left];
                float u2 = U_I[idx_left];
                float u3 = U_D[idx_left];

                // Direct param contributions
                float direct1 = (param_type == PARAM_GAP_OPEN) ? 1.0f : 0.0f;
                float direct2 = (param_type == PARAM_GAP_OPEN) ? 1.0f : 0.0f;
                float direct3 = (param_type == PARAM_GAP_EXT) ? 1.0f : 0.0f;

                float dU_D = kahan_sum3(
                    w1 * (u1 + direct1),
                    w2 * (u2 + direct2),
                    w3 * (u3 + direct3)
                );

                // For temperature: add (alpha - E[val]) / T term
                if (param_type == PARAM_TEMPERATURE) {
                    float alpha_D = A_D[idx];
                    float E_val = kahan_sum3(
                        w1 * d1,
                        w2 * d2,
                        w3 * d3
                    );
                    dU_D = kahan_sum2(dU_D, (alpha_D - E_val) / T);
                }

                U_D[idx] = dU_D;
            } else {
                U_D[idx] = 0.0f;
            }
        } else {
            U_D[idx] = 0.0f;
        }
    }
}

// Backward W pass: compute W = dbeta/dtheta and accumulate dP/dtheta
// Same structure as HVP backward but with U values as tangent inputs
__global__ void sv_affine_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ partition,
    const float* __restrict__ U,
    const float* __restrict__ dS_dtheta,  // [B] pre-computed dS/dtheta
    float* __restrict__ beta,              // workspace (accumulated)
    float* __restrict__ W,                 // workspace (tangent of beta)
    float* __restrict__ dP_dtheta,         // output [B, L1, L2]
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag,
    int param_type
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all    = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * max_L2;
    size_t stride_row    = (size_t)max_L2 + 1;
    size_t score_row     = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;
    const float* s   = scores + (size_t)b * stride_scores;

    const float* U_M = U + (size_t)b * stride_all;
    const float* U_I = U_M + stride_state;
    const float* U_D = U_I + stride_state;

    float* B_M = beta + (size_t)b * stride_all;
    float* B_I = B_M + stride_state;
    float* B_D = B_I + stride_state;

    float* W_M = W + (size_t)b * stride_all;
    float* W_I = W_M + stride_state;
    float* W_D = W_I + stride_state;

    float* dP = dP_dtheta + (size_t)b * stride_scores;

    float S = partition[b];
    float dS = dS_dtheta[b];

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left  = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float score = s[score_idx];

        // ========== M state ==========
        float betaM = B_M[idx];
        // Fix #10: Apply posterior clamping
        float beta_initM = 0.0f;
        if (A_M[idx] > NINF) {
            float log_post = (A_M[idx] - S) / T;
            log_post = fminf(log_post, 0.0f);  // Clamp to <= 0
            beta_initM = safe_exp(log_post);
        }
        float dW_initM = beta_initM * (U_M[idx] - dS) / T;
        // For temperature: add extra term from dbeta_init/dT = ... - beta_init*(alpha-S)/T^2
        if (param_type == PARAM_TEMPERATURE && A_M[idx] > NINF) {
            dW_initM = kahan_sum2(
                dW_initM,
                -beta_initM * (A_M[idx] - S) / (T * T)
            );
        }
        float wM = kahan_sum2(W_M[idx], dW_initM);

        if (has_finite_nonzero_signal(betaM) || has_finite_nonzero_signal(wM)) {
            float m1 = (i > 1 && j > 1) ? A_M[idx_diag] + score : NINF;
            float m2 = (i > 1 && j > 1) ? A_I[idx_diag] + score : NINF;
            float m3 = (i > 1 && j > 1) ? A_D[idx_diag] + score : NINF;
            float m4 = score;
            float w1, w2, w3, w4;

            if (softmax4_with_lse(m1, m2, m3, m4, T, w1, w2, w3, w4) > NINF) {
                // Tangent of logits for M state (all use score, no direct param)
                float du1 = (i > 1 && j > 1) ? U_M[idx_diag] : 0.0f;
                float du2 = (i > 1 && j > 1) ? U_I[idx_diag] : 0.0f;
                float du3 = (i > 1 && j > 1) ? U_D[idx_diag] : 0.0f;
                float du4 = 0.0f;

                float dw1, dw2, dw3, dw4;
                softmax4_tangent(w1, w2, w3, w4, du1, du2, du3, du4, T, dw1, dw2, dw3, dw4);

                // For temperature: add direct dw/dT = w_k * (E[v] - v_k) / T^2
                if (param_type == PARAM_TEMPERATURE) {
                    float E_v = kahan_sum4(
                        w1 * m1,
                        w2 * m2,
                        w3 * m3,
                        w4 * m4
                    );
                    float inv_T2 = 1.0f / (T * T);
                    dw1 = kahan_sum2(dw1, w1 * (E_v - m1) * inv_T2);
                    dw2 = kahan_sum2(dw2, w2 * (E_v - m2) * inv_T2);
                    dw3 = kahan_sum2(dw3, w3 * (E_v - m3) * inv_T2);
                    dw4 = kahan_sum2(dw4, w4 * (E_v - m4) * inv_T2);
                }

                // Propagate beta and W to predecessors
                if (m1 > NINF) {
                    atomicAdd(&B_M[idx_diag], betaM * w1);
                    atomicAdd(
                        &W_M[idx_diag],
                        kahan_sum2(wM * w1, betaM * dw1)
                    );
                }
                if (m2 > NINF) {
                    atomicAdd(&B_I[idx_diag], betaM * w2);
                    atomicAdd(
                        &W_I[idx_diag],
                        kahan_sum2(wM * w2, betaM * dw2)
                    );
                }
                if (m3 > NINF) {
                    atomicAdd(&B_D[idx_diag], betaM * w3);
                    atomicAdd(
                        &W_D[idx_diag],
                        kahan_sum2(wM * w3, betaM * dw3)
                    );
                }

                // dP/dtheta contribution: d(posterior) = dW (since sum of dw = 0)
                atomicAdd(&dP[score_idx], wM);
            }
        }

        // ========== I state ==========
        float betaI = B_I[idx];
        float wI = W_I[idx];

        if ((has_finite_nonzero_signal(betaI) || has_finite_nonzero_signal(wI)) && i > 1) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float w1, w2;

            if (softmax2_with_lse(i1, i2, T, w1, w2) > NINF) {
                // Tangent of logits for I state
                float du1 = U_M[idx_up];
                float du2 = U_I[idx_up];
                // Add direct param contribution to tangent
                if (param_type == PARAM_GAP_OPEN) du1 += 1.0f;
                if (param_type == PARAM_GAP_EXT) du2 += 1.0f;

                float dw1, dw2;
                softmax2_tangent(w1, w2, du1, du2, T, dw1, dw2);

                // For temperature: add direct dw/dT = w_k * (E[v] - v_k) / T^2
                if (param_type == PARAM_TEMPERATURE) {
                    float E_v = kahan_sum2(w1 * i1, w2 * i2);
                    float inv_T2 = 1.0f / (T * T);
                    dw1 = kahan_sum2(dw1, w1 * (E_v - i1) * inv_T2);
                    dw2 = kahan_sum2(dw2, w2 * (E_v - i2) * inv_T2);
                }

                // Propagate beta and W
                atomicAdd(&B_M[idx_up], betaI * w1);
                atomicAdd(
                    &W_M[idx_up],
                    kahan_sum2(wI * w1, betaI * dw1)
                );
                atomicAdd(&B_I[idx_up], betaI * w2);
                atomicAdd(
                    &W_I[idx_up],
                    kahan_sum2(wI * w2, betaI * dw2)
                );
                // I state doesn't contribute to score gradient
            }
        }

        // ========== D state ==========
        float betaD = B_D[idx];
        float wD = W_D[idx];

        if ((has_finite_nonzero_signal(betaD) || has_finite_nonzero_signal(wD)) && j > 1) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;

            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                // Tangent of logits for D state
                float du1 = U_M[idx_left];
                float du2 = U_I[idx_left];
                float du3 = U_D[idx_left];
                // Add direct param contribution to tangent
                if (param_type == PARAM_GAP_OPEN) {
                    du1 += 1.0f;
                    du2 += 1.0f;
                }
                if (param_type == PARAM_GAP_EXT) du3 += 1.0f;

                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, du1, du2, du3, T, dw1, dw2, dw3);

                // For temperature: add direct dw/dT = w_k * (E[v] - v_k) / T^2
                if (param_type == PARAM_TEMPERATURE) {
                    float E_v = kahan_sum3(
                        w1 * d1,
                        w2 * d2,
                        w3 * d3
                    );
                    float inv_T2 = 1.0f / (T * T);
                    dw1 = kahan_sum2(dw1, w1 * (E_v - d1) * inv_T2);
                    dw2 = kahan_sum2(dw2, w2 * (E_v - d2) * inv_T2);
                    dw3 = kahan_sum2(dw3, w3 * (E_v - d3) * inv_T2);
                }

                // Propagate beta and W
                atomicAdd(&B_M[idx_left], betaD * w1);
                atomicAdd(
                    &W_M[idx_left],
                    kahan_sum2(wD * w1, betaD * dw1)
                );
                atomicAdd(&B_I[idx_left], betaD * w2);
                atomicAdd(
                    &W_I[idx_left],
                    kahan_sum2(wD * w2, betaD * dw2)
                );
                atomicAdd(&B_D[idx_left], betaD * w3);
                atomicAdd(
                    &W_D[idx_left],
                    kahan_sum2(wD * w3, betaD * dw3)
                );
                // D state doesn't contribute to score gradient
            }
        }
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

extern "C" void sv_affine_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,     // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int threads = 256;
    hipStream_t stream = orihime::common::get_cuda_stream();
    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t total_alpha  = (size_t)B * stride_all;

    // Initialize alpha
    size_t blocks_init = (total_alpha + threads - 1) / threads;
    sv_affine_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(d_alpha, B, max_L1, max_L2);
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Wavefront DP
    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        sv_affine_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }

    // Partition function
    sv_affine_partition_kernel<<<B, threads, 0, stream>>>(d_alpha, d_partition, d_lengths, B, max_L1, max_L2, T);
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Synchronize to ensure all kernels complete before returning
    C10_HIP_CHECK(hipStreamSynchronize(stream));
}

extern "C" void sv_affine_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    float* d_beta,          // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_posteriors,    // output [B,max_L1,max_L2]
    float* d_grad_open,     // output [B]
    float* d_grad_ext,      // output [B]
    float* d_grad_T,        // output [B]
    const int* d_lengths,   // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int threads = 256;
    hipStream_t stream = orihime::common::get_cuda_stream();
    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t total_alpha  = (size_t)B * stride_all;
    size_t score_elems  = (size_t)B * max_L1 * max_L2;

    // Zero outputs
    C10_HIP_CHECK(hipMemsetAsync(d_beta,       0, sizeof(float) * total_alpha, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_grad_open,  0, sizeof(float) * B, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_grad_ext,   0, sizeof(float) * B, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_grad_T,     0, sizeof(float) * B, stream));

    // Initialize beta
    size_t blocks_init = (total_alpha + threads - 1) / threads;
    sv_affine_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Backward DP
    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        sv_affine_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_grad_open, d_grad_ext,
            d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }

    // Temperature gradient
    sv_affine_grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_grad_open, d_grad_ext, d_partition, d_grad_T,
        d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T
    );
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Synchronize to ensure all kernels complete before returning
    C10_HIP_CHECK(hipStreamSynchronize(stream));
}

extern "C" void sv_affine_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    const float* d_V,           // input [B,max_L1,max_L2]
    float* d_d_alpha,           // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_d_partition,       // workspace [B]
    float* d_beta,              // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_d_beta,            // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_H_scores,          // output [B,max_L1,max_L2]
    const int* d_lengths,       // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int threads = 256;
    hipStream_t stream = orihime::common::get_cuda_stream();
    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t total_alpha  = (size_t)B * stride_all;
    size_t score_elems  = (size_t)B * max_L1 * max_L2;

    // d_beta is a caller-provided workspace from the PyTorch caching allocator
    // (torch::zeros), matching every sibling operator. It is fully overwritten
    // by sv_affine_init_beta_kernel below, so there is no raw driver allocation
    // here and nothing to leak on a thrown HIP check.

    // Zero workspaces and output
    C10_HIP_CHECK(hipMemsetAsync(d_d_alpha,     0, sizeof(float) * total_alpha, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_d_partition, 0, sizeof(float) * B, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_d_beta,      0, sizeof(float) * total_alpha, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_H_scores,    0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward pass:
    // compute d_alpha tangents
    for (int k = 2; k <= max_diag; ++k) {
        sv_affine_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }

    // Compute d_partition
    sv_affine_hvp_partition_kernel<<<B, threads, 0, stream>>>(
        d_alpha, d_d_alpha, d_partition, d_d_partition, d_lengths, B, max_L1, max_L2, T
    );
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Initialize beta = exp((alpha - S) / T) before backward pass
    size_t blocks_init = (total_alpha + threads - 1) / threads;
    sv_affine_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Backward pass: compute H_scores
    // Now passes d_beta (accumulated beta) and d_d_beta (tangent)
    for (int k = max_diag; k >= 2; --k) {
        sv_affine_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_partition, d_d_alpha, d_d_partition, d_V,
            d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }

    // Synchronize to ensure all kernels complete before returning
    C10_HIP_CHECK(hipStreamSynchronize(stream));
}

extern "C" void sv_affine_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    const float* d_dS_dtheta,       // pre-computed [B] - dS/dtheta from backward
    float* d_U,                     // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_beta,                  // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_W,                     // workspace [B,3*(max_L1+1)*(max_L2+1)]
    float* d_dP_dtheta,             // output [B,max_L1,max_L2]
    const int* d_lengths,           // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int param_type                  // 0=gap_open, 1=gap_ext, 2=temperature
) {
    int threads = 256;
    hipStream_t stream = orihime::common::get_cuda_stream();
    size_t stride_state = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_all   = 3 * stride_state;
    size_t total_alpha  = (size_t)B * stride_all;
    size_t score_elems  = (size_t)B * max_L1 * max_L2;

    // Zero workspaces and output
    C10_HIP_CHECK(hipMemsetAsync(d_U,         0, sizeof(float) * total_alpha, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_beta,      0, sizeof(float) * total_alpha, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_W,         0, sizeof(float) * total_alpha, stream));
    C10_HIP_CHECK(hipMemsetAsync(d_dP_dtheta, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward U pass: compute U = dalpha/dtheta
    for (int k = 2; k <= max_diag; ++k) {
        sv_affine_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_lengths, B, max_L1, max_L2,
            gap_open, gap_ext, T, k, param_type
        );
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }

    // Initialize beta = exp((alpha - S) / T) before backward pass
    size_t blocks_init = (total_alpha + threads - 1) / threads;
    sv_affine_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_HIP_KERNEL_LAUNCH_CHECK();

    // Backward W pass: compute W = dbeta/dtheta and accumulate dP/dtheta
    for (int k = max_diag; k >= 2; --k) {
        sv_affine_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_partition, d_U, d_dS_dtheta,
            d_beta, d_W, d_dP_dtheta, d_lengths,
            B, max_L1, max_L2, gap_open, gap_ext, T, k, param_type
        );
        C10_HIP_KERNEL_LAUNCH_CHECK();
    }

    // Synchronize to ensure all kernels complete before returning
    C10_HIP_CHECK(hipStreamSynchronize(stream));
}
