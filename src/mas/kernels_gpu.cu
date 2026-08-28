// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_gpu.cu
 * @brief Soft Monotonic Alignment Search (MAS) CUDA Kernel Implementations
 *
 * MAS aligns frames (T) to text positions (S) with monotonic constraint.
 * Uses anti-diagonal wavefront parallelization for the 2D DP.
 */

#include "kernels_gpu.cuh"
#include "common/cuda_utils.h"
#include "common/numerics.h"
#include <cuda_runtime.h>
#include <cmath>

namespace orihime {
namespace mas {

using orihime::common::KahanSum;

// =============================================================================
// Device Helpers
// =============================================================================

#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T safe_exp(T x) {
    if (x < (T)-88.0f) return (T)0.0f;
    if (x > (T)88.0f) x = (T)88.0f;
    return exp(x);
}

__device__ __forceinline__ size_t cell_offset(int t, int s, int max_S) {
    return (size_t)t * (size_t)max_S + (size_t)s;
}

__device__ __forceinline__ KahanSum warp_reduce_kahan(KahanSum sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_sum = __shfl_down_sync(0xffffffff, sum.sum, offset);
        float other_c = __shfl_down_sync(0xffffffff, sum.c, offset);
        sum.add(other_sum);
        sum.add(-other_c);
    }
    return sum;
}

__device__ __forceinline__ float block_reduce_kahan(KahanSum sum) {
    __shared__ float shared_sum[32];
    __shared__ float shared_c[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    sum = warp_reduce_kahan(sum);
    if (lane == 0) {
        shared_sum[wid] = sum.sum;
        shared_c[wid] = sum.c;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    KahanSum block_sum;
    if (threadIdx.x < num_warps) {
        block_sum.sum = shared_sum[lane];
        block_sum.c = shared_c[lane];
    }
    if (wid == 0) block_sum = warp_reduce_kahan(block_sum);
    return block_sum.result();
}

__device__ __forceinline__ float softmax2(float a, float b, float T) {
    float m = fmaxf(a, b);
    if (m <= NINF) return NINF;

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;

    KahanSum sum;
    sum.add(ea);
    sum.add(eb);
    if (sum.result() <= 0.0f) return NINF;
    return m + T * logf(sum.result());
}

__device__ __forceinline__ void softmax2_weights(
    float a, float b, float T,
    float& wa, float& wb
) {
    float m = fmaxf(a, b);
    if (m <= NINF) {
        wa = wb = 0.0f;
        return;
    }

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;

    KahanSum sum;
    sum.add(ea);
    sum.add(eb);
    float total = sum.result();
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
    } else {
        wa = wb = 0.0f;
    }
}

__device__ __forceinline__ void softmax2_tangent(
    float w1, float w2,
    float dv1, float dv2,
    float T,
    float& dw1, float& dw2
) {
    KahanSum weighted_tangent;
    weighted_tangent.add(w1 * dv1);
    weighted_tangent.add(w2 * dv2);
    dw1 = w1 * (dv1 - weighted_tangent.result()) / T;
    dw2 = w2 * (dv2 - weighted_tangent.result()) / T;
}

// =============================================================================
// Forward Kernels
// =============================================================================

__global__ void init_alpha_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    size_t stride = (size_t)max_T * max_S;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t off = idx - b * stride;
    int t = (int)(off / max_S);
    int s = (int)(off % max_S);

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    const float* sc = scores + b * stride;
    float* a = alpha + b * stride;

    if (t >= T || s >= S) {
        a[cell_offset(t, s, max_S)] = NINF;
        return;
    }

    if (t == 0 && s == 0) {
        a[0] = sc[0];
    } else if (s == 0 && t > 0) {
        a[cell_offset(t, 0, max_S)] = NINF;
    } else if (t == 0 && s > 0) {
        a[s] = NINF;
    } else {
        a[cell_offset(t, s, max_S)] = NINF;
    }
}

__global__ void init_first_col_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    const float* sc = scores + b * stride;
    float* a = alpha + b * stride;

    KahanSum first_col_sum;
    first_col_sum.add(sc[0]);
    for (int t = 1; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        first_col_sum.add(sc[idx]);
        a[idx] = first_col_sum.result();
    }
}

__global__ void forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* sc = scores + b * stride;
    float* a = alpha + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag - 1);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 1 || s >= S) continue;

        size_t idx = cell_offset(t, s, max_S);
        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = a[idx_diag];

        KahanSum cell_sum;
        cell_sum.add(sc[idx]);
        cell_sum.add(softmax2(stay, diag, T_temp));
        a[idx] = cell_sum.result();
    }
}

__global__ void score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ score,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;
    size_t final_idx = cell_offset(T - 1, S - 1, max_S);

    score[b] = alpha[b * stride + final_idx];
}

// =============================================================================
// Backward Kernels
// =============================================================================

__global__ void init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    size_t stride = (size_t)max_T * max_S;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t off = idx - b * stride;
    int t = (int)(off / max_S);
    int s = (int)(off % max_S);

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    if (t == T - 1 && s == S - 1) {
        beta[idx] = 1.0f;
    } else {
        beta[idx] = 0.0f;
    }
}

__global__ void backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    float* be = beta + b * stride;
    float* P = posteriors + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 0 || s >= S) continue;
        if (t < 1) continue;

        size_t idx = cell_offset(t, s, max_S);
        float beta_ts = be[idx];
        if (beta_ts < 1e-30f) continue;

        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = (s >= 1) ? a[idx_diag] : NINF;

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        if (stay > NINF) {
            atomicAdd(&be[idx_stay], beta_ts * w_stay);
        }
        if (s >= 1 && diag > NINF) {
            atomicAdd(&be[idx_diag], beta_ts * w_diag);
        }

        P[idx] = beta_ts;
    }
}

__global__ void backward_first_col_kernel(
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    float* be = beta + b * stride;
    float* P = posteriors + b * stride;

    for (int t = T - 1; t >= 0; t--) {
        size_t idx = cell_offset(t, 0, max_S);
        P[idx] = be[idx];
    }
}

__global__ void grad_T_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ posteriors,
    const float* __restrict__ partition,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S, float T_temp
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;
    const float* sc = scores + b * stride;
    const float* P = posteriors + b * stride;

    KahanSum expected_score;

    size_t total_valid = (size_t)T * (size_t)S;
    for (size_t idx = threadIdx.x; idx < total_valid; idx += blockDim.x) {
        int t = (int)(idx / (size_t)S);
        int s = (int)(idx % (size_t)S);
        if (t < T && s < S) {
            size_t flat_idx = cell_offset(t, s, max_S);
            expected_score.add(P[flat_idx] * sc[flat_idx]);
        }
    }

    float reduced_expected_score = block_reduce_kahan(expected_score);

    if (threadIdx.x == 0) {
        grad_T[b] = (partition[b] - reduced_expected_score) / T_temp;
    }
}

// =============================================================================
// HVP Kernels
// =============================================================================

__global__ void hvp_init_first_col_kernel(
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    const float* v = V + b * stride;
    float* da = d_alpha + b * stride;

    KahanSum first_col_tangent;
    first_col_tangent.add(v[0]);
    da[0] = first_col_tangent.result();
    for (int t = 1; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        first_col_tangent.add(v[idx]);
        da[idx] = first_col_tangent.result();
    }
}

__global__ void hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    const float* v = V + b * stride;
    float* da = d_alpha + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag - 1);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 1 || s >= S) continue;

        size_t idx = cell_offset(t, s, max_S);
        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = a[idx_diag];

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        KahanSum tangent_sum;
        tangent_sum.add(v[idx]);
        tangent_sum.add(w_stay * da[idx_stay]);
        tangent_sum.add(w_diag * da[idx_diag]);
        da[idx] = tangent_sum.result();
    }
}

__global__ void hvp_score_kernel(
    const float* __restrict__ d_alpha,
    float* __restrict__ d_score,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;
    size_t final_idx = cell_offset(T - 1, S - 1, max_S);

    d_score[b] = d_alpha[b * stride + final_idx];
}

__global__ void hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    const float* da = d_alpha + b * stride;
    float* be = beta + b * stride;
    float* dbe = d_beta + b * stride;
    float* H = H_scores + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 0 || s >= S) continue;
        if (t < 1) continue;

        size_t idx = cell_offset(t, s, max_S);
        float beta_ts = be[idx];
        float dbeta_ts = dbe[idx];

        if (beta_ts < 1e-30f && fabsf(dbeta_ts) < 1e-30f) continue;

        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = (s >= 1) ? a[idx_diag] : NINF;

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        float da_stay = da[idx_stay];
        float da_diag = (s >= 1) ? da[idx_diag] : 0.0f;

        float dw_stay, dw_diag;
        softmax2_tangent(w_stay, w_diag, da_stay, da_diag, T_temp, dw_stay, dw_diag);

        if (stay > NINF) {
            atomicAdd(&be[idx_stay], beta_ts * w_stay);
            KahanSum dbe_stay;
            dbe_stay.add(dbeta_ts * w_stay);
            dbe_stay.add(beta_ts * dw_stay);
            atomicAdd(&dbe[idx_stay], dbe_stay.result());
        }
        if (s >= 1 && diag > NINF) {
            atomicAdd(&be[idx_diag], beta_ts * w_diag);
            KahanSum dbe_diag;
            dbe_diag.add(dbeta_ts * w_diag);
            dbe_diag.add(beta_ts * dw_diag);
            atomicAdd(&dbe[idx_diag], dbe_diag.result());
        }

        H[idx] = dbeta_ts;
    }
}

__global__ void hvp_backward_first_col_kernel(
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    float* dbe = d_beta + b * stride;
    float* H = H_scores + b * stride;

    for (int t = 0; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        H[idx] = dbe[idx];
    }
}

// =============================================================================
// Parameter Gradient Kernels
// =============================================================================

__global__ void param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    float* u = U + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag - 1);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 1 || s >= S) continue;

        size_t idx = cell_offset(t, s, max_S);
        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = a[idx_diag];

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        float u_stay = u[idx_stay];
        float u_diag = u[idx_diag];

        if (a[idx] <= NINF) {
            u[idx] = 0.0f;
            continue;
        }

        KahanSum expected_value;
        expected_value.add(w_stay * stay);
        expected_value.add(w_diag * diag);
        float E_v = expected_value.result();
        float soft_value = softmax2(stay, diag, T_temp);

        // d/dT [T * logsumexp(v / T)] = sum_i w_i * dv_i/dT + (f - E[v]) / T.
        KahanSum u_sum;
        u_sum.add(w_stay * u_stay);
        u_sum.add(w_diag * u_diag);
        u_sum.add((soft_value - E_v) / T_temp);
        u[idx] = u_sum.result();
    }
}

__global__ void param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    const float* u = U + b * stride;
    float* be = beta + b * stride;
    float* w = W + b * stride;
    float* dP = dP_dT + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 0 || s >= S) continue;
        if (t < 1) continue;

        size_t idx = cell_offset(t, s, max_S);
        float beta_ts = be[idx];
        float w_ts = w[idx];

        if (beta_ts < 1e-30f && fabsf(w_ts) < 1e-30f) continue;

        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = (s >= 1) ? a[idx_diag] : NINF;

        float wt_stay, wt_diag;
        softmax2_weights(stay, diag, T_temp, wt_stay, wt_diag);

        float u_stay = u[idx_stay];
        float u_diag = (s >= 1) ? u[idx_diag] : 0.0f;

        float dw_stay, dw_diag;
        softmax2_tangent(wt_stay, wt_diag, u_stay, u_diag, T_temp, dw_stay, dw_diag);

        KahanSum expected_value;
        expected_value.add(wt_stay * stay);
        expected_value.add(wt_diag * diag);
        float E_v = expected_value.result();
        float inv_T2 = 1.0f / (T_temp * T_temp);
        KahanSum dw_stay_sum;
        dw_stay_sum.add(dw_stay);
        dw_stay_sum.add(wt_stay * (E_v - stay) * inv_T2);
        dw_stay = dw_stay_sum.result();
        KahanSum dw_diag_sum;
        dw_diag_sum.add(dw_diag);
        dw_diag_sum.add(wt_diag * (E_v - diag) * inv_T2);
        dw_diag = dw_diag_sum.result();

        if (stay > NINF) {
            atomicAdd(&be[idx_stay], beta_ts * wt_stay);
            KahanSum w_stay_sum;
            w_stay_sum.add(w_ts * wt_stay);
            w_stay_sum.add(beta_ts * dw_stay);
            atomicAdd(&w[idx_stay], w_stay_sum.result());
        }
        if (s >= 1 && diag > NINF) {
            atomicAdd(&be[idx_diag], beta_ts * wt_diag);
            KahanSum w_diag_sum;
            w_diag_sum.add(w_ts * wt_diag);
            w_diag_sum.add(beta_ts * dw_diag);
            atomicAdd(&w[idx_diag], w_diag_sum.result());
        }

        dP[idx] = w_ts;
    }
}

__global__ void param_grad_backward_first_col_kernel(
    const float* __restrict__ W,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    const float* w = W + b * stride;
    float* dP = dP_dT + b * stride;

    for (int t = 0; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        dP[idx] = w[idx];
    }
}

// =============================================================================
// Host Functions
// =============================================================================

void forward(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = orihime::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    size_t blocks_init = (total + threads - 1) / threads;
    init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_scores, d_alpha, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    init_first_col_kernel<<<B, 1, 0, stream>>>(
        d_scores, d_alpha, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_T + max_S - 2;
    for (int k = 2; k <= max_diag; ++k) {
        forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_partition, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = orihime::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    size_t blocks_init = (total + threads - 1) / threads;
    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_T + max_S - 2;
    for (int k = max_diag; k >= 1; --k) {
        backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    backward_first_col_kernel<<<B, 1, 0, stream>>>(
        d_beta, d_posteriors, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_partition, d_grad_T, d_lengths,
        B, max_T, max_S, temperature
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = orihime::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * total, stream));

    hvp_init_first_col_kernel<<<B, 1, 0, stream>>>(
        d_V, d_d_alpha, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_T + max_S - 2;

    for (int k = 2; k <= max_diag; ++k) {
        hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_d_alpha, d_d_score, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    size_t blocks_init = (total + threads - 1) / threads;
    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 1; --k) {
        hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_V, d_d_alpha, d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    hvp_backward_first_col_kernel<<<B, 1, 0, stream>>>(
        d_d_beta, d_H_scores, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void param_grad(
    const float* d_alpha,
    const float* d_scores,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = orihime::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    C10_CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dT, 0, sizeof(float) * total, stream));

    int max_diag = max_T + max_S - 2;

    for (int k = 2; k <= max_diag; ++k) {
        param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_U, d_lengths, B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    size_t blocks_init = (total + threads - 1) / threads;
    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 1; --k) {
        param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_U, d_beta, d_W, d_dP_dT, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    param_grad_backward_first_col_kernel<<<B, 1, 0, stream>>>(
        d_W, d_dP_dT, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mas
} // namespace orihime
