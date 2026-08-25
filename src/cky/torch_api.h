// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace d2p::cky {

std::vector<torch::Tensor> cky_forward_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temp
);

std::vector<torch::Tensor> cky_forward_t_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor temp
);

std::tuple<torch::Tensor, torch::Tensor> cky_value_grad_params_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temp
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cky_marginals_backward_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor grad_marginals,
    double temp
);

torch::Tensor cky_marginals_hvp_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor v_merge,
    torch::Tensor v_leaf,
    double temp
);

torch::Tensor cky_marginals_grad_leaf_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor v_leaf,
    double temp
);

torch::Tensor cky_marginals_grad_temp_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temp
);

std::vector<torch::Tensor> cky_forward_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temp
);

std::vector<torch::Tensor> cky_forward_t_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor temp
);

std::tuple<torch::Tensor, torch::Tensor> cky_value_grad_params_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temp
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cky_marginals_backward_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor grad_marginals,
    double temp
);

torch::Tensor cky_marginals_hvp_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor v_merge,
    torch::Tensor v_leaf,
    double temp
);

torch::Tensor cky_marginals_grad_leaf_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor v_leaf,
    double temp
);

torch::Tensor cky_marginals_grad_temp_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temp
);

}  // namespace d2p::cky
