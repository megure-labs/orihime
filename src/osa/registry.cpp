// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft OSA Operator Schema Registration
 *
 * Registers operator schemas for the orihime::soft_osa family of functions.
 * CPU and CUDA implementations are registered in their respective files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(orihime, m) {
    // Main autograd function with tensor parameters
    // Returns (osa_score, posteriors) tuple
    m.def("soft_osa(Tensor sub_costs, Tensor trans_mask, Tensor ins_cost, Tensor del_cost, Tensor trans_cost, Tensor temperature, Tensor lengths) -> Tensor[]");

    // Convenience function with float parameters (no param gradient)
    m.def("soft_osa_float(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> Tensor[]");

    // Forward + backward with explicit outputs for debugging
    // Returns (osa_score, posteriors, grad_T, grad_ins, grad_del, grad_trans)
    m.def("soft_osa_with_grads(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    // Hessian-vector product
    m.def("soft_osa_hvp(Tensor sub_costs, Tensor trans_mask, Tensor tangent, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> Tensor");

    // Parameter Jacobian: dP/d{param_type}
    // param_type: 0=ins, 1=del, 2=trans, 3=temperature
    m.def("soft_osa_param_jacobian(Tensor sub_costs, Tensor trans_mask, int param_type, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> Tensor");

    // Complete backward given grad_posteriors
    // Returns (grad_sub_costs, grad_temperature, grad_ins, grad_del, grad_trans)
    m.def("soft_osa_backward_full(Tensor sub_costs, Tensor trans_mask, Tensor grad_output, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    // Namespaced API (osa_*)
    m.def("osa_forward(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> Tensor[]");
    m.def("osa_forward_t(Tensor sub_costs, Tensor trans_mask, Tensor ins_cost, Tensor del_cost, Tensor trans_cost, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("osa_value_grad_params(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("osa_marginals_backward(Tensor sub_costs, Tensor trans_mask, Tensor grad_marginals, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("osa_marginals_hvp(Tensor sub_costs, Tensor trans_mask, Tensor v, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> Tensor");
    m.def("osa_marginals_grad_ins_cost(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> Tensor");
    m.def("osa_marginals_grad_del_cost(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> Tensor");
    m.def("osa_marginals_grad_trans_cost(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> Tensor");
    m.def("osa_marginals_grad_temp(Tensor sub_costs, Tensor trans_mask, float ins_cost, float del_cost, float trans_cost, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
