// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft Eisner PyTorch Library Registration
 *
 * Defines the operator schemas for differentiable Eisner dependency parsing.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(orihime, m) {
    m.def("soft_eisner(Tensor arc_scores, Tensor temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_eisner_float(Tensor arc_scores, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_eisner_with_grads(Tensor arc_scores, float temperature, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("soft_eisner_hvp(Tensor arc_scores, Tensor V, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_eisner_backward_full(Tensor arc_scores, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("soft_eisner_param_jacobian(Tensor arc_scores, float temperature, Tensor? lengths) -> Tensor");

    // Namespaced API (eisner_*)
    m.def("eisner_forward(Tensor arc_scores, float temp, Tensor? lengths) -> Tensor[]");
    m.def("eisner_forward_t(Tensor arc_scores, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("eisner_value_grad_params(Tensor arc_scores, float temp, Tensor? lengths) -> Tensor");
    m.def("eisner_marginals_backward(Tensor arc_scores, Tensor grad_marginals, float temp, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("eisner_marginals_hvp(Tensor arc_scores, Tensor v, float temp, Tensor? lengths) -> Tensor");
    m.def("eisner_marginals_grad_temp(Tensor arc_scores, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
