// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft MAS CPU PyTorch Bindings
 *
 * CPU implementation of Monotonic Alignment Search for TTS/ASR.
 */

#include <torch/extension.h>
#include <limits>
#include <vector>

#include "common/torch_utils.h"
#include "kernels_cpu.h"

using namespace orihime::common;

namespace {

void validate_mas_scores_cpu(const torch::Tensor& scores) {
    ORIHIME_CHECK_CPU(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D [B, T, S]");
    ORIHIME_CHECK_FLOAT(scores);
    for (int dim = 0; dim < 3; ++dim) {
        TORCH_CHECK(
            scores.size(dim) <= std::numeric_limits<int>::max(),
            "scores dimensions must fit int32, got ", scores.sizes()
        );
    }
}

void validate_mas_tangent_cpu(
    const torch::Tensor& scores,
    const torch::Tensor& V
) {
    ORIHIME_CHECK_CPU(V);
    ORIHIME_CHECK_FLOAT(V);
    TORCH_CHECK(
        V.sizes() == scores.sizes(),
        "V must have same shape as scores"
    );
}

void validate_mas_temperature_tensor_cpu(
    const torch::Tensor& scores,
    const torch::Tensor& temperature
) {
    ORIHIME_CHECK_CPU(temperature);
    ORIHIME_CHECK_FLOAT(temperature);
    TORCH_CHECK(
        temperature.device() == scores.device(),
        "temperature must be on same device as scores, got ",
        temperature.device(), " vs ", scores.device()
    );
    TORCH_CHECK(
        temperature.numel() == 1,
        "temperature must contain exactly one value"
    );
}

void validate_mas_lengths_cpu(
    const torch::Tensor& lengths,
    int B,
    int max_T,
    int max_S
) {
    ORIHIME_CHECK_CPU(lengths);
    ORIHIME_CHECK_CONTIGUOUS(lengths);
    TORCH_CHECK(
        lengths.dim() == 2 && lengths.size(0) == B && lengths.size(1) == 2,
        "lengths must be [B, 2], got ", lengths.sizes()
    );
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");

    auto lengths_acc = lengths.accessor<int32_t, 2>();
    for (int b = 0; b < B; ++b) {
        int T = lengths_acc[b][0];
        int S = lengths_acc[b][1];
        TORCH_CHECK(
            T >= 1 && T <= max_T,
            "lengths[", b, ",0] must be between 1 and ", max_T, ", got ", T
        );
        TORCH_CHECK(
            S >= 1 && S <= max_S,
            "lengths[", b, ",1] must be between 1 and ", max_S, ", got ", S
        );
    }
}

torch::Tensor resolve_mas_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_T,
    int max_S,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_T, max_S, device);
    validate_mas_lengths_cpu(lengths, B, max_T, max_S);
    return lengths;
}

}  // namespace

// =============================================================================
// Autograd Function
// =============================================================================

class SoftMASCPUFunction : public torch::autograd::Function<SoftMASCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor scores,
        double temperature,
        c10::optional<torch::Tensor> lengths_opt
    ) {
        validate_mas_scores_cpu(scores);

        auto scores_contig = scores.contiguous();
        int B = scores_contig.size(0);
        int max_T = scores_contig.size(1);
        int max_S = scores_contig.size(2);

        torch::Tensor lengths = resolve_mas_lengths_cpu(
            lengths_opt, B, max_T, max_S, scores_contig.device()
        );

        auto alpha = torch::empty({B, max_T, max_S}, scores_contig.options());
        auto partition = torch::empty({B}, scores_contig.options());

        orihime::mas::cpu::forward(
            scores_contig.data_ptr<float>(),
            alpha.data_ptr<float>(),
            partition.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_T, max_S, (float)temperature
        );

        // Backward DP pass: compute posteriors and grad_T
        auto beta = torch::empty_like(alpha);
        auto posteriors = torch::empty({B, max_T, max_S}, scores_contig.options());
        auto grad_T = torch::empty({B}, scores_contig.options());

        orihime::mas::cpu::backward(
            alpha.data_ptr<float>(),
            scores_contig.data_ptr<float>(),
            partition.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_T, max_S, (float)temperature
        );

        ctx->save_for_backward({scores_contig, alpha, partition, lengths, grad_T});
        ctx->saved_data["temperature"] = temperature;

        return {partition, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto scores = saved[0];
        auto alpha = saved[1];
        auto partition = saved[2];
        auto lengths = saved[3];
        auto grad_T_fwd = saved[4];
        double temperature = ctx->saved_data["temperature"].toDouble();

        int B = scores.size(0);
        int max_T = scores.size(1);
        int max_S = scores.size(2);

        auto options = scores.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_T, max_S}, options);

        // ============ Gradient from score (partition) ============
        if (grad_score.defined() && grad_score.numel() > 0) {
            auto beta = torch::empty_like(alpha);
            auto posteriors = torch::empty({B, max_T, max_S}, options);
            auto tmp_T = torch::empty({B}, options);

            orihime::mas::cpu::backward(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_T, max_S, (float)temperature
            );

            grad_scores += posteriors * grad_score.contiguous().view({B, 1, 1});
        }

        // ============ Gradient from posteriors (alignment) ============
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            TORCH_CHECK(grad_posteriors.sizes() == scores.sizes(),
                        "grad_posteriors shape mismatch");

            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            grad_posteriors = grad_posteriors.contiguous();

            // HVP: d^2S/dscores^2 * grad_posteriors
            auto d_alpha = torch::empty_like(alpha);
            auto d_score = torch::empty({B}, options);
            auto beta = torch::empty_like(alpha);
            auto d_beta = torch::empty_like(alpha);
            auto hvp_result = torch::empty({B, max_T, max_S}, options);

            orihime::mas::cpu::hvp(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_result.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_T, max_S, (float)temperature
            );

            grad_scores += hvp_result;
        }

        return {grad_scores, torch::Tensor(), torch::Tensor()};
    }
};

class MASForwardTCPUFunction
    : public torch::autograd::Function<MASForwardTCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor scores,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        ctx->set_materialize_grads(false);

        validate_mas_scores_cpu(scores);
        validate_mas_temperature_tensor_cpu(scores, temperature);

        auto scores_contig = scores.contiguous();
        int B = scores_contig.size(0);
        int max_T = scores_contig.size(1);
        int max_S = scores_contig.size(2);
        validate_mas_lengths_cpu(lengths, B, max_T, max_S);

        float temp_val = temperature.item<float>();
        auto alpha = torch::empty_like(scores_contig);
        auto partition = torch::empty({B}, scores_contig.options());

        orihime::mas::cpu::forward(
            scores_contig.data_ptr<float>(),
            alpha.data_ptr<float>(),
            partition.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_T, max_S, temp_val
        );

        auto beta = torch::empty_like(alpha);
        auto posteriors = torch::empty_like(scores_contig);
        auto grad_T = torch::empty({B}, scores_contig.options());

        orihime::mas::cpu::backward(
            alpha.data_ptr<float>(),
            scores_contig.data_ptr<float>(),
            partition.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_T, max_S, temp_val
        );

        ctx->save_for_backward({
            scores_contig, alpha, partition, lengths, grad_T, temperature
        });
        ctx->saved_data["temperature"] = static_cast<double>(temp_val);

        return {partition, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto scores = saved[0];
        auto alpha = saved[1];
        auto partition = saved[2];
        auto lengths = saved[3];
        auto grad_T_fwd = saved[4];
        auto temperature = saved[5];
        float temp_val = static_cast<float>(
            ctx->saved_data["temperature"].toDouble()
        );

        int B = scores.size(0);
        int max_T = scores.size(1);
        int max_S = scores.size(2);
        auto options = scores.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];
        torch::Tensor grad_scores = torch::zeros_like(scores);
        torch::Tensor total_grad_T = torch::zeros_like(temperature);

        if (grad_score.defined() && grad_score.numel() > 0) {
            auto beta = torch::empty_like(alpha);
            auto posteriors = torch::empty_like(scores);
            auto tmp_T = torch::empty({B}, options);

            orihime::mas::cpu::backward(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_T, max_S, temp_val
            );

            grad_score = grad_score.contiguous();
            grad_scores += posteriors * grad_score.view({B, 1, 1});
            total_grad_T += (
                grad_score * grad_T_fwd
            ).sum().reshape(temperature.sizes());
        }

        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            TORCH_CHECK(
                grad_posteriors.sizes() == scores.sizes(),
                "grad_posteriors shape mismatch"
            );
            ORIHIME_CHECK_CPU(grad_posteriors);
            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            grad_posteriors = grad_posteriors.contiguous();

            auto d_alpha = torch::empty_like(alpha);
            auto d_score = torch::empty({B}, options);
            auto beta = torch::empty_like(alpha);
            auto d_beta = torch::empty_like(alpha);
            auto hvp_result = torch::empty_like(scores);

            orihime::mas::cpu::hvp(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_result.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_T, max_S, temp_val
            );
            grad_scores += hvp_result;

            auto U = torch::empty_like(alpha);
            auto param_beta = torch::empty_like(alpha);
            auto W = torch::empty_like(alpha);
            auto dP_dT = torch::empty_like(scores);

            orihime::mas::cpu::param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                U.data_ptr<float>(),
                param_beta.data_ptr<float>(),
                W.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_T, max_S, temp_val
            );
            total_grad_T += (
                grad_posteriors * dP_dT
            ).sum().reshape(temperature.sizes());
        }

        return {grad_scores, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Wrapper Functions
// =============================================================================

torch::Tensor soft_mas_cpu(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_mas_scores_cpu(scores);

    auto scores_contig = scores.contiguous();
    int B = scores_contig.size(0);
    int max_T = scores_contig.size(1);
    int max_S = scores_contig.size(2);

    torch::Tensor lens = resolve_mas_lengths_cpu(
        lengths, B, max_T, max_S, scores_contig.device()
    );

    auto alpha = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto partition = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::forward(
        scores_contig.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    auto beta = torch::empty_like(alpha);
    auto posteriors = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto grad_T = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::backward(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    return posteriors;
}

std::vector<torch::Tensor> soft_mas_cpu_float(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    return SoftMASCPUFunction::apply(scores, temperature, lengths);
}

std::tuple<torch::Tensor, torch::Tensor> soft_mas_cpu_with_grads(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_mas_scores_cpu(scores);

    auto scores_contig = scores.contiguous();
    int B = scores_contig.size(0);
    int max_T = scores_contig.size(1);
    int max_S = scores_contig.size(2);

    torch::Tensor lens = resolve_mas_lengths_cpu(
        lengths, B, max_T, max_S, scores_contig.device()
    );

    auto alpha = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto partition = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::forward(
        scores_contig.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    auto beta = torch::empty_like(alpha);
    auto posteriors = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto grad_T = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::backward(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    return std::make_tuple(partition, posteriors);
}

torch::Tensor soft_mas_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor V,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_mas_scores_cpu(scores);
    validate_mas_tangent_cpu(scores, V);

    auto scores_contig = scores.contiguous();
    auto V_contig = V.contiguous();
    int B = scores_contig.size(0);
    int max_T = scores_contig.size(1);
    int max_S = scores_contig.size(2);

    torch::Tensor lens = resolve_mas_lengths_cpu(
        lengths, B, max_T, max_S, scores_contig.device()
    );

    auto alpha = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto partition = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::forward(
        scores_contig.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    auto d_alpha = torch::empty_like(alpha);
    auto d_score = torch::empty({B}, scores_contig.options());
    auto beta = torch::empty_like(alpha);
    auto d_beta = torch::empty_like(alpha);
    auto H_scores = torch::empty({B, max_T, max_S}, scores_contig.options());

    orihime::mas::cpu::hvp(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        V_contig.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    return H_scores;
}

torch::Tensor soft_mas_param_jacobian_cpu(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_mas_scores_cpu(scores);

    auto scores_contig = scores.contiguous();
    int B = scores_contig.size(0);
    int max_T = scores_contig.size(1);
    int max_S = scores_contig.size(2);

    torch::Tensor lens = resolve_mas_lengths_cpu(
        lengths, B, max_T, max_S, scores_contig.device()
    );

    auto alpha = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto partition = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::forward(
        scores_contig.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    auto U = torch::empty_like(alpha);
    auto beta = torch::empty_like(alpha);
    auto W = torch::empty_like(alpha);
    auto dP_dT = torch::empty({B, max_T, max_S}, scores_contig.options());

    orihime::mas::cpu::param_grad(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    return dP_dT;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> soft_mas_backward_full_cpu(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_mas_scores_cpu(scores);

    auto scores_contig = scores.contiguous();
    int B = scores_contig.size(0);
    int max_T = scores_contig.size(1);
    int max_S = scores_contig.size(2);

    torch::Tensor lens = resolve_mas_lengths_cpu(
        lengths, B, max_T, max_S, scores_contig.device()
    );

    auto alpha = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto partition = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::forward(
        scores_contig.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    auto beta = torch::empty_like(alpha);
    auto posteriors = torch::empty({B, max_T, max_S}, scores_contig.options());
    auto grad_T = torch::empty({B}, scores_contig.options());

    orihime::mas::cpu::backward(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, (float)temperature
    );

    return std::make_tuple(partition, posteriors, grad_T);
}

// =============================================================================
// Namespaced API Wrappers (mas_*)
// =============================================================================

std::vector<torch::Tensor> mas_forward_cpu(
    torch::Tensor scores,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_mas_cpu_float(scores, temp, lengths);
}

std::vector<torch::Tensor> mas_forward_t_cpu(
    torch::Tensor scores,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return MASForwardTCPUFunction::apply(scores, temp, lengths);
}

torch::Tensor mas_value_grad_params_cpu(
    torch::Tensor scores,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_mas_backward_full_cpu(scores, temp, lengths);
    return std::get<2>(result);
}

std::tuple<torch::Tensor, torch::Tensor> mas_marginals_backward_cpu(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    validate_mas_scores_cpu(scores);
    validate_mas_tangent_cpu(scores, grad_marginals);

    auto scores_contig = scores.contiguous();
    auto grad_marginals_contig = grad_marginals.contiguous();
    int B = scores_contig.size(0);
    int max_T = scores_contig.size(1);
    int max_S = scores_contig.size(2);
    auto options = scores_contig.options();

    torch::Tensor lens = resolve_mas_lengths_cpu(
        lengths, B, max_T, max_S, scores_contig.device()
    );

    auto alpha = torch::empty_like(scores_contig);
    auto partition = torch::empty({B}, options);
    orihime::mas::cpu::forward(
        scores_contig.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, static_cast<float>(temp)
    );

    auto d_alpha = torch::empty_like(alpha);
    auto d_score = torch::empty({B}, options);
    auto beta = torch::empty_like(alpha);
    auto d_beta = torch::empty_like(alpha);
    auto grad_scores = torch::empty_like(scores_contig);
    orihime::mas::cpu::hvp(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        grad_marginals_contig.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_scores.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, static_cast<float>(temp)
    );

    auto U = torch::empty_like(alpha);
    auto param_beta = torch::empty_like(alpha);
    auto W = torch::empty_like(alpha);
    auto dP_dT = torch::empty_like(scores_contig);
    orihime::mas::cpu::param_grad(
        alpha.data_ptr<float>(),
        scores_contig.data_ptr<float>(),
        U.data_ptr<float>(),
        param_beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lens.data_ptr<int>(),
        B, max_T, max_S, static_cast<float>(temp)
    );

    torch::Tensor grad_temp = (
        grad_marginals_contig * dP_dT
    ).sum().reshape({1});
    return std::make_tuple(grad_scores, grad_temp);
}

torch::Tensor mas_marginals_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor v,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_mas_hvp_cpu(scores, v, temp, lengths);
}

torch::Tensor mas_marginals_grad_temp_cpu(
    torch::Tensor scores,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_mas_param_jacobian_cpu(scores, temp, lengths);
}

// =============================================================================
// Library Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(orihime, CPU, m) {
    m.impl("soft_mas", soft_mas_cpu);
    m.impl("soft_mas_float", soft_mas_cpu_float);
    m.impl("soft_mas_with_grads", soft_mas_cpu_with_grads);
    m.impl("soft_mas_hvp", soft_mas_hvp_cpu);
    m.impl("soft_mas_param_jacobian", soft_mas_param_jacobian_cpu);
    m.impl("soft_mas_backward_full", soft_mas_backward_full_cpu);

    m.impl("mas_forward", mas_forward_cpu);
    m.impl("mas_forward_t", mas_forward_t_cpu);
    m.impl("mas_value_grad_params", mas_value_grad_params_cpu);
    m.impl("mas_marginals_backward", mas_marginals_backward_cpu);
    m.impl("mas_marginals_hvp", mas_marginals_hvp_cpu);
    m.impl("mas_marginals_grad_temp", mas_marginals_grad_temp_cpu);
}

TORCH_LIBRARY_IMPL(orihime, AutogradCPU, m) {
    m.impl("soft_mas_float", soft_mas_cpu_float);
    m.impl("mas_forward", mas_forward_cpu);
    m.impl("mas_forward_t", mas_forward_t_cpu);
}

#endif // USE_TORCH_LIBRARY
