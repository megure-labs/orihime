// SPDX-License-Identifier: Apache-2.0
// numerics.h - CPU numerical primitives for orihime
//
// Shared across all orihime CPU operators. Float32 only.
// Values MUST match numerics.cuh exactly for CPU/CUDA parity.
//
// Usage:
//   #include "common/numerics.h"
//   float x = orihime::common::safe_exp(val);

#pragma once
#include <cmath>
#include <algorithm>

#if defined(__CUDACC__)
#define ORIHIME_NUMERICS_HD __host__ __device__
#else
#define ORIHIME_NUMERICS_HD
#endif

namespace orihime {
namespace common {

// ============================================================================
// IMPORTANT: These values MUST match numerics.cuh exactly for CPU/CUDA parity.
// All orihime operations are float32 only.
// ============================================================================

inline constexpr float NINF = -1e30f;
inline constexpr float PINF = 1e30f;
inline constexpr float EXP_CLAMP_MIN = -88.0f;
inline constexpr float EXP_CLAMP_MAX = 88.0f;

ORIHIME_NUMERICS_HD inline float safe_exp(float x) {
    if (x < EXP_CLAMP_MIN) return 0.0f;
    if (x > EXP_CLAMP_MAX) x = EXP_CLAMP_MAX;
#if defined(__CUDA_ARCH__)
    return expf(x);
#else
    return std::exp(x);
#endif
}

// ============================================================================
// Kahan compensated summation for numerical stability
// Use for accumulating many small values (common in CPU DP)
// ============================================================================

struct KahanSum {
    float sum = 0.0f;
    float c = 0.0f;  // Compensation for lost low-order bits

    ORIHIME_NUMERICS_HD KahanSum() {}

    ORIHIME_NUMERICS_HD inline void add(float val) {
        float y = val - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    ORIHIME_NUMERICS_HD inline float result() const { return sum; }

    ORIHIME_NUMERICS_HD inline void reset() {
        sum = 0.0f;
        c = 0.0f;
    }
};

// ============================================================================
// CPU logsumexp helpers (temperature-scaled)
// Using Kahan summation for numerical stability
// ============================================================================

ORIHIME_NUMERICS_HD inline float logsumexp2(float a, float b, float temp) {
#if defined(__CUDA_ARCH__)
    float max_v = fmaxf(a, b);
#else
    float max_v = std::max(a, b);
#endif
    if (max_v <= NINF) return NINF;
    KahanSum sum;
    sum.add(safe_exp((a - max_v) / temp));
    sum.add(safe_exp((b - max_v) / temp));
#if defined(__CUDA_ARCH__)
    return max_v + temp * logf(sum.result());
#else
    return max_v + temp * std::log(sum.result());
#endif
}

ORIHIME_NUMERICS_HD inline float logsumexp3(float a, float b, float c, float temp) {
#if defined(__CUDA_ARCH__)
    float max_v = fmaxf(fmaxf(a, b), c);
#else
    float max_v = std::max({a, b, c});
#endif
    if (max_v <= NINF) return NINF;
    KahanSum sum;
    sum.add(safe_exp((a - max_v) / temp));
    sum.add(safe_exp((b - max_v) / temp));
    sum.add(safe_exp((c - max_v) / temp));
#if defined(__CUDA_ARCH__)
    return max_v + temp * logf(sum.result());
#else
    return max_v + temp * std::log(sum.result());
#endif
}

ORIHIME_NUMERICS_HD inline float logsumexp4(float a, float b, float c, float d, float temp) {
#if defined(__CUDA_ARCH__)
    float max_v = fmaxf(fmaxf(a, b), fmaxf(c, d));
#else
    float max_v = std::max({a, b, c, d});
#endif
    if (max_v <= NINF) return NINF;
    KahanSum sum;
    sum.add(safe_exp((a - max_v) / temp));
    sum.add(safe_exp((b - max_v) / temp));
    sum.add(safe_exp((c - max_v) / temp));
    sum.add(safe_exp((d - max_v) / temp));
#if defined(__CUDA_ARCH__)
    return max_v + temp * logf(sum.result());
#else
    return max_v + temp * std::log(sum.result());
#endif
}

// ============================================================================
// CPU softmin (for minimization: DTW, Levenshtein)
// ============================================================================

ORIHIME_NUMERICS_HD inline float softmin3(float a, float b, float c, float temp) {
#if defined(__CUDA_ARCH__)
    float min_v = fminf(fminf(a, b), c);
#else
    float min_v = std::min({a, b, c});
#endif
    if (min_v >= PINF) return PINF;
    KahanSum sum;
    sum.add(safe_exp((min_v - a) / temp));
    sum.add(safe_exp((min_v - b) / temp));
    sum.add(safe_exp((min_v - c) / temp));
#if defined(__CUDA_ARCH__)
    return min_v - temp * logf(sum.result());
#else
    return min_v - temp * std::log(sum.result());
#endif
}

}  // namespace common
}  // namespace orihime

#undef ORIHIME_NUMERICS_HD
