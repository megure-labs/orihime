// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Schema definitions for canonical Saigo-Vert linear-gap operators
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // Canonical Saigo-Vert local alignment with one per-gap-symbol penalty:
    //   M[i,j] = scores[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
    //   I[i,j] = LSE_T(M[i-1,j] + gap, I[i-1,j] + gap)
    //   D[i,j] = LSE_T(M[i,j-1] + gap, I[i,j-1] + gap,
    //                  D[i,j-1] + gap)
    //   S = LSE_T(0, {M[i,j] : i >= 1, j >= 1})

    // Compatibility operators.
    m.def("soft_sv_linear(Tensor scores, Tensor gap, Tensor temperature, Tensor lengths) -> Tensor[]");
    m.def("soft_sv_linear_float(Tensor scores, float gap, float temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_sv_linear_with_grads(Tensor scores, float gap, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("soft_sv_linear_hvp(Tensor scores, Tensor tangent, float gap, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_sv_linear_param_jacobian(Tensor scores, int param_type, float gap, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_sv_linear_backward_full(Tensor scores, Tensor grad_alignment, float gap, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor)");

    // Namespaced API.
    m.def("sv_linear_forward(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor[]");
    m.def("sv_linear_forward_t(Tensor scores, Tensor gap, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("sv_linear_value_grad_params(Tensor scores, float gap, float temp, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("sv_linear_marginals_backward(Tensor scores, Tensor grad_marginals, float gap, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("sv_linear_marginals_hvp(Tensor scores, Tensor v, float gap, float temp, Tensor? lengths) -> Tensor");
    m.def("sv_linear_marginals_grad_gap(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor");
    m.def("sv_linear_marginals_grad_temp(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
