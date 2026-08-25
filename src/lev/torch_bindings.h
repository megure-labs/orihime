// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace d2p {
namespace lev {

std::vector<torch::Tensor> lev_forward_cpu_wrapper(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

std::vector<torch::Tensor> lev_forward_t_cpu(
    torch::Tensor scores,
    torch::Tensor ins_cost,
    torch::Tensor del_cost,
    torch::Tensor temp,
    torch::Tensor lengths
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
lev_value_grad_params_cpu(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
lev_marginals_backward_cpu(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor v,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_grad_ins_cost_cpu(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_grad_del_cost_cpu(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_grad_temp_cpu(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

std::vector<torch::Tensor> lev_forward_cuda_wrapper(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

std::vector<torch::Tensor> lev_forward_t_cuda(
    torch::Tensor scores,
    torch::Tensor ins_cost,
    torch::Tensor del_cost,
    torch::Tensor temp,
    torch::Tensor lengths
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
lev_value_grad_params_cuda(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
lev_marginals_backward_cuda(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor v,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_grad_ins_cost_cuda(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_grad_del_cost_cuda(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

torch::Tensor lev_marginals_grad_temp_cuda(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temp,
    c10::optional<torch::Tensor> lengths
);

}  // namespace lev
}  // namespace d2p
