// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Canonical Saigo-Vert linear-gap CPU launch interface
 *
 * Mirrors the three-state CUDA recurrence with one per-gap-symbol penalty,
 * exactly one I->D cross, no D->I cross, M-only termination, and one explicit
 * empty alignment. CPU reductions use Kahan compensated summation.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void sv_linear_forward_cpu(
    const float* scores,
    float* alpha,
    float* partition,
    const int* lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T
);

void sv_linear_backward_cpu(
    const float* alpha,
    const float* scores,
    const float* partition,
    float* beta,
    float* posteriors,
    float* grad_gap,
    float* grad_T,
    const int* lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T
);

void sv_linear_hvp_cpu(
    const float* alpha,
    const float* scores,
    const float* partition,
    const float* V,
    float* d_alpha,
    float* d_partition,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float T
);

void sv_linear_param_grad_cpu(
    const float* alpha,
    const float* scores,
    const float* partition,
    const float* dS_dtheta,
    float* U,
    float* beta,
    float* W,
    float* dP_dtheta,
    const int* lengths,
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
