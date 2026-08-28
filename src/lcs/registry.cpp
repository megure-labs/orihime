// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft LCS Operator Schema Registration
 *
 * Registers operator schemas for the orihime::soft_lcs family of functions.
 * CPU and CUDA implementations are registered in their respective files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(orihime, m) {
    // Main autograd function with tensor parameters
    // Returns (lcs_score, posteriors) tuple
    m.def("soft_lcs(Tensor scores, Tensor temperature, Tensor lengths) -> Tensor[]");

    // Convenience function with float parameters (no param gradient)
    m.def("soft_lcs_float(Tensor scores, float temperature, Tensor? lengths) -> Tensor[]");

    // Forward + backward with explicit outputs for debugging
    // Returns (lcs_score, posteriors, grad_T)
    m.def("soft_lcs_with_grads(Tensor scores, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor)");

    // Hessian-vector product
    m.def("soft_lcs_hvp(Tensor scores, Tensor tangent, float temperature, Tensor? lengths) -> Tensor");

    // Parameter Jacobian: dP/dT
    m.def("soft_lcs_param_jacobian(Tensor scores, float temperature, Tensor? lengths) -> Tensor");

    // Complete backward given grad_posteriors
    // Returns (grad_scores, grad_temperature)
    m.def("soft_lcs_backward_full(Tensor scores, Tensor grad_output, float temperature, Tensor? lengths) -> (Tensor, Tensor)");

    // Namespaced API (lcs_*)
    // LCS has only one scalar parameter: temperature. Skips are free.
    m.def("lcs_forward(Tensor scores, float temp, Tensor? lengths) -> Tensor[]");
    m.def("lcs_forward_t(Tensor scores, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("lcs_value_grad_params(Tensor scores, float temp, Tensor? lengths) -> Tensor");
    m.def("lcs_marginals_backward(Tensor scores, Tensor grad_marginals, float temp, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("lcs_marginals_hvp(Tensor scores, Tensor v, float temp, Tensor? lengths) -> Tensor");
    m.def("lcs_marginals_grad_temp(Tensor scores, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
