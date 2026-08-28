// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Canonical Saigo-Vert linear-gap CPU Extension with PyTorch Autograd
 *
 * Canonical three-state DP with one linear gap penalty. Implementations register
 * via TORCH_LIBRARY_IMPL for automatic dispatch.
 *
 * The M/I/D recurrence has exactly one I->D cross, no D->I cross, M-only
 * termination, and one explicit empty-alignment term.
 */

#include <torch/extension.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"

// CPU kernel declarations
#include "sv_linear/kernels_cpu.h"

using namespace orihime::common;

namespace {

struct SvLinearCpuShape {
    int B;
    int max_L1;
    int max_L2;
    int64_t alpha_size;
};

SvLinearCpuShape checked_sv_linear_cpu_shape(const torch::Tensor& scores) {
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    constexpr int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    const int64_t B64 = scores.size(0);
    const int64_t max_L1_64 = scores.size(1);
    const int64_t max_L2_64 = scores.size(2);

    TORCH_CHECK(B64 <= int_max, "SV linear CPU batch size must be <= ", int_max, ", got ", B64);
    TORCH_CHECK(max_L1_64 <= int_max, "SV linear CPU L1 dimension must be <= ", int_max, ", got ", max_L1_64);
    TORCH_CHECK(max_L2_64 <= int_max, "SV linear CPU L2 dimension must be <= ", int_max, ", got ", max_L2_64);

    const int64_t alpha_rows = max_L1_64 + 1;
    const int64_t alpha_cols = max_L2_64 + 1;
    constexpr int64_t max_cells = int_max / 3;
    TORCH_CHECK(
        alpha_rows <= max_cells / alpha_cols,
        "SV linear CPU DP table size exceeds supported int32 range: 3 * (L1 + 1) * (L2 + 1) must be <= ",
        int_max,
        ", got ",
        alpha_rows,
        " * ",
        alpha_cols
    );

    return {
        static_cast<int>(B64),
        static_cast<int>(max_L1_64),
        static_cast<int>(max_L2_64),
        3 * alpha_rows * alpha_cols
    };
}

void validate_sv_linear_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
    ORIHIME_CHECK_CPU(lengths);
    ORIHIME_CHECK_CONTIGUOUS(lengths);
    TORCH_CHECK(lengths.dim() == 2 && lengths.size(0) == B && lengths.size(1) == 2);
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");

    auto lengths_acc = lengths.accessor<int32_t, 2>();
    for (int b = 0; b < B; ++b) {
        int l1 = lengths_acc[b][0];
        int l2 = lengths_acc[b][1];
        TORCH_CHECK(
            l1 >= 0 && l1 <= max_L1,
            "lengths[", b, ",0] must be between 0 and ", max_L1, ", got ", l1
        );
        TORCH_CHECK(
            l2 >= 0 && l2 <= max_L2,
            "lengths[", b, ",1] must be between 0 and ", max_L2, ", got ", l2
        );
    }
}

torch::Tensor resolve_sv_linear_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_sv_linear_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

}  // namespace

// =============================================================================
// Saigo-Vert linear CPU Autograd Function
// =============================================================================

class SoftSVLinearCPUFunction : public torch::autograd::Function<SoftSVLinearCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor gap,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        // r70: leave the unused posteriors output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP/param-grad path.
        ctx->set_materialize_grads(false);

        ORIHIME_CHECK_INPUT_CPU(scores);
        TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
        TORCH_CHECK(gap.numel() == 1, "gap must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;

        validate_sv_linear_lengths_cpu(lengths, B, max_L1, max_L2);

        float gap_val = gap.item<float>();
        float temp_val = temperature.item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor partition = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_gap = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        sv_linear_forward_cpu(
            scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
        );

        sv_linear_backward_cpu(
            alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
            beta.data_ptr<float>(), posteriors.data_ptr<float>(),
            grad_gap.data_ptr<float>(), grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
        );

        ctx->save_for_backward({scores.clone(), alpha.clone(), partition.clone(), lengths.clone(),
                                grad_gap.clone(), grad_T.clone()});
        ctx->saved_data["gap"] = gap_val;
        ctx->saved_data["temperature"] = temp_val;

        return {partition, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor scores = saved[0];
        torch::Tensor alpha = saved[1];
        torch::Tensor partition = saved[2];
        torch::Tensor lengths = saved[3];
        torch::Tensor grad_gap_fwd = saved[4];
        torch::Tensor grad_T_fwd = saved[5];

        float gap_val = static_cast<float>(ctx->saved_data["gap"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;

        auto options = scores.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_gap = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from score path
        if (grad_score.defined() && grad_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_gap = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            sv_linear_backward_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                beta.data_ptr<float>(), posteriors.data_ptr<float>(),
                tmp_gap.data_ptr<float>(), tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
            );

            grad_scores += grad_score.view({B, 1, 1}) * posteriors;
            total_grad_gap += (grad_score * grad_gap_fwd).sum().reshape({1});
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from alignment path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_partition = torch::zeros({B}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            sv_linear_hvp_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(), d_alpha.data_ptr<float>(),
                d_partition.data_ptr<float>(), d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param grads
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

            sv_linear_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_gap_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val, 0
            );
            total_grad_gap += (grad_posteriors * dP_dtheta).sum().reshape({1});

            sv_linear_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val, 1
            );
            total_grad_T += (grad_posteriors * dP_dtheta).sum().reshape({1});
        }

        return {grad_scores, total_grad_gap, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions (CPU)
// =============================================================================

std::vector<torch::Tensor> soft_sv_linear_cpu(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftSVLinearCPUFunction::apply(scores, gap, temperature, lengths);
}

std::vector<torch::Tensor> soft_sv_linear_cpu_float(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
    int B = shape.B;
    int L1 = shape.max_L1;
    int L2 = shape.max_L2;

    torch::Tensor gap_t = torch::tensor({static_cast<float>(gap)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = resolve_sv_linear_lengths_cpu(lengths_opt, B, L1, L2, scores.device());

    return SoftSVLinearCPUFunction::apply(scores, gap_t, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_sv_linear_cpu_with_grads(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    ORIHIME_CHECK_INPUT_CPU(scores);
    SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sv_linear_lengths_cpu(lengths_opt, B, max_L1, max_L2, scores.device());

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    sv_linear_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    sv_linear_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    return std::make_tuple(partition, posteriors, grad_gap, grad_T);
}

torch::Tensor soft_sv_linear_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    ORIHIME_CHECK_INPUT_CPU(scores);
    ORIHIME_CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");
    SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sv_linear_lengths_cpu(lengths_opt, B, max_L1, max_L2, scores.device());

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    sv_linear_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    sv_linear_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        tangent.data_ptr<float>(), d_alpha.data_ptr<float>(), d_partition.data_ptr<float>(),
        d_beta.data_ptr<float>(), H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    return H_scores;
}

torch::Tensor soft_sv_linear_param_jacobian_cpu(
    torch::Tensor scores, int64_t param_type, double gap, double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    ORIHIME_CHECK_INPUT_CPU(scores);
    TORCH_CHECK(param_type >= 0 && param_type <= 1, "param_type must be 0 or 1");
    SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sv_linear_lengths_cpu(lengths_opt, B, max_L1, max_L2, scores.device());

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    sv_linear_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    sv_linear_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    torch::Tensor dS_dtheta = (param_type == 0) ? grad_gap : grad_T;

    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    sv_linear_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        dS_dtheta.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), param_type
    );

    return dP_dtheta;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_sv_linear_backward_full_cpu(
    torch::Tensor scores, torch::Tensor grad_alignment,
    double gap, double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    ORIHIME_CHECK_INPUT_CPU(scores);
    TORCH_CHECK(!grad_alignment.is_cuda(), "grad_alignment must be a CPU tensor");
    TORCH_CHECK(grad_alignment.sizes() == scores.sizes(), "grad_alignment must have same shape as scores");
    SvLinearCpuShape shape = checked_sv_linear_cpu_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sv_linear_lengths_cpu(lengths_opt, B, max_L1, max_L2, scores.device());

    grad_alignment = grad_alignment.contiguous().to(torch::kFloat32);

    // Forward
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    sv_linear_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    sv_linear_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        beta_fwd.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap_fwd.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    // HVP
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    sv_linear_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_alignment.data_ptr<float>(), d_alpha.data_ptr<float>(), d_partition.data_ptr<float>(),
        d_beta.data_ptr<float>(), grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    // Param grads
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    sv_linear_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_gap_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_gap = (grad_alignment * dP_dtheta).sum().reshape({1});

    sv_linear_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dtheta).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_gap, total_grad_T);
}

// =============================================================================
// Namespaced API wrappers (sv_linear_*)
// =============================================================================

// sv_linear::forward - returns (value, marginals)
std::vector<torch::Tensor> orihime_sv_linear_forward_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sv_linear_cpu_float(scores, gap, temp, lengths);
}

// sv_linear::forward_t - tensor params version
std::vector<torch::Tensor> orihime_sv_linear_forward_t_cpu(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_sv_linear_cpu(scores, gap, temp, lengths);
}

// sv_linear::value_grad_params - returns (grad_gap, grad_temp) per batch
std::tuple<torch::Tensor, torch::Tensor> orihime_sv_linear_value_grad_params_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_sv_linear_cpu_with_grads(scores, gap, temp, lengths);
    return std::make_tuple(std::get<2>(result), std::get<3>(result));
}

// sv_linear::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> orihime_sv_linear_marginals_backward_cpu(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sv_linear_backward_full_cpu(scores, grad_marginals, gap, temp, lengths);
}

// sv_linear::marginals_hvp - Hessian-vector product
torch::Tensor orihime_sv_linear_marginals_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor v,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sv_linear_hvp_cpu(scores, v, gap, temp, lengths);
}

// sv_linear::marginals_grad_gap - d(marginals)/d(gap)
torch::Tensor orihime_sv_linear_marginals_grad_gap_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sv_linear_param_jacobian_cpu(scores, 0, gap, temp, lengths);
}

// sv_linear::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor orihime_sv_linear_marginals_grad_temp_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sv_linear_param_jacobian_cpu(scores, 1, gap, temp, lengths);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(orihime, CPU, m) {
    m.impl("soft_sv_linear", soft_sv_linear_cpu);
    m.impl("soft_sv_linear_float", soft_sv_linear_cpu_float);
    m.impl("soft_sv_linear_with_grads", soft_sv_linear_cpu_with_grads);
    m.impl("soft_sv_linear_hvp", soft_sv_linear_hvp_cpu);
    m.impl("soft_sv_linear_param_jacobian", soft_sv_linear_param_jacobian_cpu);
    m.impl("soft_sv_linear_backward_full", soft_sv_linear_backward_full_cpu);

    // Namespaced API
    m.impl("sv_linear_forward", orihime_sv_linear_forward_cpu);
    m.impl("sv_linear_forward_t", orihime_sv_linear_forward_t_cpu);
    m.impl("sv_linear_value_grad_params", orihime_sv_linear_value_grad_params_cpu);
    m.impl("sv_linear_marginals_backward", orihime_sv_linear_marginals_backward_cpu);
    m.impl("sv_linear_marginals_hvp", orihime_sv_linear_marginals_hvp_cpu);
    m.impl("sv_linear_marginals_grad_gap", orihime_sv_linear_marginals_grad_gap_cpu);
    m.impl("sv_linear_marginals_grad_temp", orihime_sv_linear_marginals_grad_temp_cpu);
}

TORCH_LIBRARY_IMPL(orihime, AutogradCPU, m) {
    m.impl("soft_sv_linear", soft_sv_linear_cpu);
    m.impl("soft_sv_linear_float", soft_sv_linear_cpu_float);

    // Namespaced API - autograd versions
    m.impl("sv_linear_forward", orihime_sv_linear_forward_cpu);
    m.impl("sv_linear_forward_t", orihime_sv_linear_forward_t_cpu);
}

#endif // USE_TORCH_LIBRARY
