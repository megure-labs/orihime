// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_gpu.cuh
 * @brief Canonical Saigo-Vert linear-gap CUDA launch interface
 *
 * The three-state recurrence charges one `gap` for every consumed unmatched
 * symbol. It contains exactly one I->D cross, no D->I cross, and terminates
 * only in M plus one explicit empty-alignment term:
 *
 *   M[i,j] = score[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
 *   I[i,j] = LSE_T(M[i-1,j] + gap, I[i-1,j] + gap)
 *   D[i,j] = LSE_T(M[i,j-1] + gap, I[i,j-1] + gap, D[i,j-1] + gap)
 *   S = LSE_T(0, {M[i,j] : i >= 1, j >= 1})
 *
 * Alpha and derivative workspaces use [B, 3*(L1+1)*(L2+1)] storage.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void sv_linear_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T
);

void sv_linear_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_gap,
    float* d_grad_T,
    const int* d_lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T
);

void sv_linear_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_partition,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T
);

void sv_linear_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    const float* d_dS_dtheta,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dtheta,
    const int* d_lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T,
    int param_type
);

#ifdef __cplusplus
}
#endif
