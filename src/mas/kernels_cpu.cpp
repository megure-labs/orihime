// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft Monotonic Alignment Search (MAS) CPU Kernel Implementations
 *
 * CPU implementation of MAS for TTS/ASR alignment.
 */

#include "kernels_cpu.h"
#include "common/numerics.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace orihime {
namespace mas {
namespace cpu {

using orihime::common::KahanSum;

// =============================================================================
// Helper Functions
// =============================================================================

static inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

static inline float softmax2(float a, float b, float T) {
    float m = std::max(a, b);
    if (m <= NINF) return NINF;

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;

    KahanSum sum;
    sum.add(ea);
    sum.add(eb);
    if (sum.result() <= 0.0f) return NINF;
    return m + T * std::log(sum.result());
}

static inline void softmax2_weights(
    float a, float b, float T,
    float& wa, float& wb
) {
    float m = std::max(a, b);
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

static inline void softmax2_tangent(
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

static inline size_t cell_index(int t, int s, int max_S) {
    return static_cast<size_t>(t) * static_cast<size_t>(max_S) + static_cast<size_t>(s);
}

// =============================================================================
// Forward Pass
// =============================================================================

void forward(
    const float* scores,
    float* alpha,
    float* partition,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    const size_t stride = (size_t)max_T * max_S;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* sc = scores + b * stride;
            float* a = alpha + b * stride;
            int T = lengths[b * 2];
            int S = lengths[b * 2 + 1];

            // Initialize to NINF
            for (size_t i = 0; i < stride; i++) {
                a[i] = NINF;
            }

            // Base case: alpha(0, 0) = score(0, 0)
            a[0] = sc[0];

            // Base case: alpha(t, 0) = alpha(t-1, 0) + score(t, 0)
            KahanSum first_col_sum;
            first_col_sum.add(sc[0]);
            for (int t = 1; t < T; t++) {
                const size_t idx = cell_index(t, 0, max_S);
                first_col_sum.add(sc[idx]);
                a[idx] = first_col_sum.result();
            }

            // Fill DP table
            for (int t = 1; t < T; t++) {
                for (int s = 1; s < S; s++) {
                    const size_t idx = cell_index(t, s, max_S);
                    const size_t idx_stay = cell_index(t - 1, s, max_S);
                    const size_t idx_diag = cell_index(t - 1, s - 1, max_S);

                    float stay = a[idx_stay];
                    float diag = a[idx_diag];

                    KahanSum cell_sum;
                    cell_sum.add(sc[idx]);
                    cell_sum.add(softmax2(stay, diag, temperature));
                    a[idx] = cell_sum.result();
                }
            }

            // Score
            const size_t final_idx = cell_index(T - 1, S - 1, max_S);
            partition[b] = a[final_idx];
        }
    });
}

// =============================================================================
// Backward Pass
// =============================================================================

void backward(
    const float* alpha,
    const float* scores,
    const float* partition,
    float* beta,
    float* posteriors,
    float* grad_T,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    const size_t stride = (size_t)max_T * max_S;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * stride;
            const float* sc = scores + b * stride;
            float* be = beta + b * stride;
            float* P = posteriors + b * stride;
            int T = lengths[b * 2];
            int S = lengths[b * 2 + 1];

            // Initialize
            for (size_t i = 0; i < stride; i++) {
                be[i] = 0.0f;
                P[i] = 0.0f;
            }

            // beta(T-1, S-1) = 1
            const size_t final_idx = cell_index(T - 1, S - 1, max_S);
            be[final_idx] = 1.0f;

            // Backward pass
            for (int t = T - 1; t >= 1; t--) {
                for (int s = S - 1; s >= 0; s--) {
                    const size_t idx = cell_index(t, s, max_S);
                    float beta_ts = be[idx];

                    if (beta_ts < 1e-30f) continue;

                    // Posteriors
                    P[idx] = beta_ts;

                    // Recompute weights
                    const size_t idx_stay = cell_index(t - 1, s, max_S);

                    float stay = a[idx_stay];
                    float diag = NINF;
                    size_t idx_diag = 0;
                    if (s >= 1) {
                        idx_diag = cell_index(t - 1, s - 1, max_S);
                        diag = a[idx_diag];
                    }

                    float w_stay, w_diag;
                    softmax2_weights(stay, diag, temperature, w_stay, w_diag);

                    // Propagate beta
                    be[idx_stay] += beta_ts * w_stay;
                    if (s >= 1) {
                        be[idx_diag] += beta_ts * w_diag;
                    }
                }
            }

            // First row posteriors
            for (int t = 0; t < T; t++) {
                const size_t idx = cell_index(t, 0, max_S);
                P[idx] = be[idx];
            }

            // Temperature gradient
            KahanSum expected_score;
            for (int t = 0; t < T; t++) {
                for (int s = 0; s < S; s++) {
                    const size_t idx = cell_index(t, s, max_S);
                    expected_score.add(P[idx] * sc[idx]);
                }
            }
            grad_T[b] = (partition[b] - expected_score.result()) / temperature;
        }
    });
}

// =============================================================================
// HVP (Hessian-Vector Product)
// =============================================================================

void hvp(
    const float* alpha,
    const float* scores,
    const float* V,
    float* d_alpha,
    float* d_score,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    const size_t stride = (size_t)max_T * max_S;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * stride;
            const float* v = V + b * stride;
            float* da = d_alpha + b * stride;
            float* be = beta + b * stride;
            float* dbe = d_beta + b * stride;
            float* H = H_scores + b * stride;
            int T = lengths[b * 2];
            int S = lengths[b * 2 + 1];

            // Initialize
            for (size_t i = 0; i < stride; i++) {
                da[i] = 0.0f;
                be[i] = 0.0f;
                dbe[i] = 0.0f;
                H[i] = 0.0f;
            }

            // Forward tangent: first column
            KahanSum first_col_tangent;
            first_col_tangent.add(v[0]);
            da[0] = first_col_tangent.result();
            for (int t = 1; t < T; t++) {
                const size_t idx = cell_index(t, 0, max_S);
                first_col_tangent.add(v[idx]);
                da[idx] = first_col_tangent.result();
            }

            // Forward tangent: main DP
            for (int t = 1; t < T; t++) {
                for (int s = 1; s < S; s++) {
                    const size_t idx = cell_index(t, s, max_S);
                    const size_t idx_stay = cell_index(t - 1, s, max_S);
                    const size_t idx_diag = cell_index(t - 1, s - 1, max_S);

                    float stay = a[idx_stay];
                    float diag = a[idx_diag];

                    float w_stay, w_diag;
                    softmax2_weights(stay, diag, temperature, w_stay, w_diag);

                    KahanSum tangent_sum;
                    tangent_sum.add(v[idx]);
                    tangent_sum.add(w_stay * da[idx_stay]);
                    tangent_sum.add(w_diag * da[idx_diag]);
                    da[idx] = tangent_sum.result();
                }
            }

            // d_score
            const size_t final_idx = cell_index(T - 1, S - 1, max_S);
            d_score[b] = da[final_idx];

            // Initialize beta at terminal
            be[final_idx] = 1.0f;

            // Backward tangent pass
            for (int t = T - 1; t >= 1; t--) {
                for (int s = S - 1; s >= 0; s--) {
                    const size_t idx = cell_index(t, s, max_S);
                    float beta_ts = be[idx];
                    float dbeta_ts = dbe[idx];

                    if (beta_ts < 1e-30f && std::abs(dbeta_ts) < 1e-30f) continue;

                    H[idx] = dbeta_ts;

                    const size_t idx_stay = cell_index(t - 1, s, max_S);

                    float stay = a[idx_stay];
                    float diag = NINF;
                    size_t idx_diag = 0;
                    if (s >= 1) {
                        idx_diag = cell_index(t - 1, s - 1, max_S);
                        diag = a[idx_diag];
                    }

                    float w_stay, w_diag;
                    softmax2_weights(stay, diag, temperature, w_stay, w_diag);

                    float da_stay = da[idx_stay];
                    float da_diag = (s >= 1) ? da[idx_diag] : 0.0f;

                    float dw_stay, dw_diag;
                    softmax2_tangent(w_stay, w_diag, da_stay, da_diag, temperature, dw_stay, dw_diag);

                    be[idx_stay] += beta_ts * w_stay;
                    KahanSum dbe_stay;
                    dbe_stay.add(dbeta_ts * w_stay);
                    dbe_stay.add(beta_ts * dw_stay);
                    dbe[idx_stay] += dbe_stay.result();

                    if (s >= 1) {
                        be[idx_diag] += beta_ts * w_diag;
                        KahanSum dbe_diag;
                        dbe_diag.add(dbeta_ts * w_diag);
                        dbe_diag.add(beta_ts * dw_diag);
                        dbe[idx_diag] += dbe_diag.result();
                    }
                }
            }

            // First column
            for (int t = 0; t < T; t++) {
                const size_t idx = cell_index(t, 0, max_S);
                H[idx] = dbe[idx];
            }
        }
    });
}

// =============================================================================
// Parameter Gradient (dP/dT)
// =============================================================================

void param_grad(
    const float* alpha,
    const float* scores,
    float* U,
    float* beta,
    float* W,
    float* dP_dT,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    const size_t stride = (size_t)max_T * max_S;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * stride;
            const float* sc = scores + b * stride;
            float* u = U + b * stride;
            float* be = beta + b * stride;
            float* w = W + b * stride;
            float* dP = dP_dT + b * stride;
            int T = lengths[b * 2];
            int S = lengths[b * 2 + 1];

            // Initialize
            for (size_t i = 0; i < stride; i++) {
                u[i] = 0.0f;
                be[i] = 0.0f;
                w[i] = 0.0f;
                dP[i] = 0.0f;
            }

            // Forward U pass
            for (int t = 1; t < T; t++) {
                for (int s = 1; s < S; s++) {
                    const size_t idx = cell_index(t, s, max_S);
                    const size_t idx_stay = cell_index(t - 1, s, max_S);
                    const size_t idx_diag = cell_index(t - 1, s - 1, max_S);

                    float stay = a[idx_stay];
                    float diag = a[idx_diag];

                    float w_stay, w_diag;
                    softmax2_weights(stay, diag, temperature, w_stay, w_diag);

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
                    float soft_value = a[idx] - sc[idx];

                    // d/dT [T * logsumexp(v / T)] = sum_i w_i * dv_i/dT + (f - E[v]) / T.
                    KahanSum u_sum;
                    u_sum.add(w_stay * u_stay);
                    u_sum.add(w_diag * u_diag);
                    u_sum.add((soft_value - E_v) / temperature);
                    u[idx] = u_sum.result();
                }
            }

            // Initialize beta at terminal
            const size_t final_idx = cell_index(T - 1, S - 1, max_S);
            be[final_idx] = 1.0f;

            // Backward W pass
            for (int t = T - 1; t >= 1; t--) {
                for (int s = S - 1; s >= 0; s--) {
                    const size_t idx = cell_index(t, s, max_S);
                    float beta_ts = be[idx];
                    float w_ts = w[idx];

                    if (beta_ts < 1e-30f && std::abs(w_ts) < 1e-30f) continue;

                    dP[idx] = w_ts;

                    const size_t idx_stay = cell_index(t - 1, s, max_S);

                    float stay = a[idx_stay];
                    float diag = NINF;
                    size_t idx_diag = 0;
                    if (s >= 1) {
                        idx_diag = cell_index(t - 1, s - 1, max_S);
                        diag = a[idx_diag];
                    }

                    float wt_stay, wt_diag;
                    softmax2_weights(stay, diag, temperature, wt_stay, wt_diag);

                    float u_stay = u[idx_stay];
                    float u_diag = (s >= 1) ? u[idx_diag] : 0.0f;

                    float dw_stay, dw_diag;
                    softmax2_tangent(wt_stay, wt_diag, u_stay, u_diag, temperature, dw_stay, dw_diag);

                    KahanSum expected_value;
                    expected_value.add(wt_stay * stay);
                    expected_value.add(wt_diag * diag);
                    float E_v = expected_value.result();
                    float inv_T2 = 1.0f / (temperature * temperature);
                    KahanSum dw_stay_sum;
                    dw_stay_sum.add(dw_stay);
                    dw_stay_sum.add(wt_stay * (E_v - stay) * inv_T2);
                    dw_stay = dw_stay_sum.result();
                    KahanSum dw_diag_sum;
                    dw_diag_sum.add(dw_diag);
                    dw_diag_sum.add(wt_diag * (E_v - diag) * inv_T2);
                    dw_diag = dw_diag_sum.result();

                    be[idx_stay] += beta_ts * wt_stay;
                    KahanSum w_stay_sum;
                    w_stay_sum.add(w_ts * wt_stay);
                    w_stay_sum.add(beta_ts * dw_stay);
                    w[idx_stay] += w_stay_sum.result();

                    if (s >= 1) {
                        be[idx_diag] += beta_ts * wt_diag;
                        KahanSum w_diag_sum;
                        w_diag_sum.add(w_ts * wt_diag);
                        w_diag_sum.add(beta_ts * dw_diag);
                        w[idx_diag] += w_diag_sum.result();
                    }
                }
            }

            // First column posteriors depend on the propagated W values too.
            for (int t = 0; t < T; t++) {
                const size_t idx = cell_index(t, 0, max_S);
                dP[idx] = w[idx];
            }
        }
    });
}

} // namespace cpu
} // namespace mas
} // namespace orihime
