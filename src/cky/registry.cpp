// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft CKY Operator Schema Registration
 *
 * Registers operator schemas for the d2p::soft_cky family of functions.
 * CPU and CUDA implementations are registered in their respective files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // Main autograd function with tensor temperature
    // Returns (logZ, posteriors) tuple
    m.def("soft_cky(Tensor merge_scores, Tensor leaf_scores, Tensor temperature) -> Tensor[]");

    // Convenience function with float temperature (no temp gradient)
    m.def("soft_cky_float(Tensor merge_scores, Tensor leaf_scores, float temperature) -> Tensor[]");

    // Forward + backward with explicit outputs for debugging
    // Returns (logZ, posteriors, grad_merge, grad_leaf)
    m.def("soft_cky_with_grads(Tensor merge_scores, Tensor leaf_scores, float temperature) -> (Tensor, Tensor, Tensor, Tensor)");

    // Hessian-vector product
    m.def("soft_cky_hvp(Tensor merge_scores, Tensor leaf_scores, Tensor v_merge, Tensor v_leaf, float temperature) -> Tensor");

    // Parameter Jacobian: dP/dT
    m.def("soft_cky_param_jacobian(Tensor merge_scores, Tensor leaf_scores, float temperature) -> Tensor");

    // Complete backward given grad_posteriors
    // Returns (grad_merge, grad_leaf, grad_temperature)
    m.def("soft_cky_backward_full(Tensor merge_scores, Tensor leaf_scores, Tensor grad_posteriors, float temperature) -> (Tensor, Tensor, Tensor)");

    // Namespaced API (cky_*)
    m.def("cky_forward(Tensor merge_scores, Tensor leaf_scores, float temp) -> Tensor[]");
    m.def("cky_forward_t(Tensor merge_scores, Tensor leaf_scores, Tensor temp) -> Tensor[]");
    m.def("cky_value_grad_params(Tensor merge_scores, Tensor leaf_scores, float temp) -> (Tensor, Tensor)");
    m.def("cky_marginals_backward(Tensor merge_scores, Tensor leaf_scores, Tensor grad_marginals, float temp) -> (Tensor, Tensor, Tensor)");
    m.def("cky_marginals_hvp(Tensor merge_scores, Tensor leaf_scores, Tensor v_merge, Tensor v_leaf, float temp) -> Tensor");
    m.def("cky_marginals_grad_leaf(Tensor merge_scores, Tensor leaf_scores, Tensor v_leaf, float temp) -> Tensor");
    m.def("cky_marginals_grad_temp(Tensor merge_scores, Tensor leaf_scores, float temp) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
