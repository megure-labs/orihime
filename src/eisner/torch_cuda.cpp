// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Soft Eisner CUDA PyTorch Bindings
 *
 * GPU-accelerated projective dependency parsing with full autograd support.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#include "common/torch_utils.h"
#include "common/cuda_utils.h"
#include "kernels.cuh"

// =============================================================================
// Helper Macros
// =============================================================================

using namespace orihime::common;

namespace {

void validate_eisner_arc_scores_cuda(const torch::Tensor& arc_scores) {
    ORIHIME_CHECK_INPUT_CUDA(arc_scores);
    TORCH_CHECK(arc_scores.dim() == 3, "arc_scores must be 3D [B, n, n]");
    TORCH_CHECK(arc_scores.size(1) == arc_scores.size(2), "arc_scores must be [B, n, n]");
    TORCH_CHECK(arc_scores.size(0) > 0, "soft_eisner requires B > 0");
    TORCH_CHECK(arc_scores.size(1) > 0, "soft_eisner requires n > 0");
}

void validate_eisner_lengths_cuda(
    const torch::Tensor& lengths,
    int B,
    int n,
    torch::Device device
) {
    ORIHIME_CHECK_CONTIGUOUS(lengths);
    ORIHIME_CHECK_LENGTHS_1D(lengths, B, device);

    auto lengths_cpu = lengths.to(torch::kCPU);
    auto lengths_acc = lengths_cpu.accessor<int32_t, 1>();
    for (int b = 0; b < B; ++b) {
        int seq_len = lengths_acc[b];
        TORCH_CHECK(
            seq_len >= 1 && seq_len <= n,
            "lengths[", b, "] must be between 1 and ", n, ", got ", seq_len
        );
    }
}

torch::Tensor resolve_eisner_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int n,
    torch::Device device
) {
    if (!lengths_opt.has_value() || !lengths_opt->defined()) {
        return {};
    }

    torch::Tensor lengths = lengths_opt.value();
    validate_eisner_lengths_cuda(lengths, B, n, device);
    return lengths;
}

void validate_eisner_tangent_cuda(
    const torch::Tensor& arc_scores,
    const torch::Tensor& tangent
) {
    ORIHIME_CHECK_INPUT_CUDA(tangent);
    TORCH_CHECK(
        tangent.sizes() == arc_scores.sizes(),
        "tangent must have same shape as arc_scores"
    );
    TORCH_CHECK(
        tangent.device() == arc_scores.device(),
        "tangent must be on same device as arc_scores, got ",
        tangent.device(),
        " vs ",
        arc_scores.device()
    );
}

}  // namespace

// =============================================================================
// Autograd Function
// =============================================================================

class SoftEisnerCUDAFunction : public torch::autograd::Function<SoftEisnerCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor arc_scores,
        torch::Tensor temperature,
        c10::optional<torch::Tensor> lengths_opt
    ) {
        // r71: leave the unused marginals output-grad undefined so a first-order
        // backward skips the zero-contribution marginals path, matching the CPU
        // Function (r70) so both backends run the same guarded logic.
        ctx->set_materialize_grads(false);

        validate_eisner_arc_scores_cuda(arc_scores);
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");
        ORIHIME_CUDA_GUARD(arc_scores);

        int B = arc_scores.size(0);
        int n = arc_scores.size(1);

        float temp_val = temperature.cpu().item<float>();
        auto options = arc_scores.options();

        torch::Tensor lengths = resolve_eisner_lengths_cuda(lengths_opt, B, n, arc_scores.device());
        const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

        // Allocate tables
        torch::Tensor C_R = torch::zeros({B, n, n}, options);
        torch::Tensor C_L = torch::zeros({B, n, n}, options);
        torch::Tensor I_R = torch::zeros({B, n, n}, options);
        torch::Tensor I_L = torch::zeros({B, n, n}, options);
        torch::Tensor partition = torch::zeros({B}, options);

        // Forward pass
        orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &partition});
        orihime::eisner::forward(
            arc_scores.data_ptr<float>(),
            C_R.data_ptr<float>(),
            C_L.data_ptr<float>(),
            I_R.data_ptr<float>(),
            I_L.data_ptr<float>(),
            partition.data_ptr<float>(),
            lengths_ptr,
            B, n, temp_val
        );

        // Backward pass (compute marginals)
        torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
        torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
        torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
        torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
        torch::Tensor marginals = torch::zeros({B, n, n}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &beta_C_R, &beta_C_L, &beta_I_R, &beta_I_L, &marginals, &grad_T});
        orihime::eisner::backward(
            arc_scores.data_ptr<float>(),
            C_R.data_ptr<float>(),
            C_L.data_ptr<float>(),
            I_R.data_ptr<float>(),
            I_L.data_ptr<float>(),
            beta_C_R.data_ptr<float>(),
            beta_C_L.data_ptr<float>(),
            beta_I_R.data_ptr<float>(),
            beta_I_L.data_ptr<float>(),
            marginals.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths_ptr,
            B, n, temp_val
        );

        // Save for backward
        ctx->saved_data["temperature"] = temp_val;
        if (lengths.defined()) {
            ctx->saved_data["has_lengths"] = true;
            ctx->save_for_backward({arc_scores.clone(), C_R, C_L, I_R, I_L, grad_T, lengths});
        } else {
            ctx->saved_data["has_lengths"] = false;
            ctx->save_for_backward({arc_scores.clone(), C_R, C_L, I_R, I_L, grad_T});
        }

        return {partition, marginals};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor arc_scores = saved[0];
        torch::Tensor C_R = saved[1];
        torch::Tensor C_L = saved[2];
        torch::Tensor I_R = saved[3];
        torch::Tensor I_L = saved[4];
        torch::Tensor grad_T_fwd = saved[5];
        ORIHIME_CUDA_GUARD(arc_scores);

        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());
        bool has_lengths = ctx->saved_data["has_lengths"].toBool();

        const int* lengths_ptr = nullptr;
        if (has_lengths && saved.size() > 6) {
            torch::Tensor lengths = saved[6];
            lengths_ptr = lengths.data_ptr<int>();
        }

        int B = arc_scores.size(0);
        int n = arc_scores.size(1);
        auto options = arc_scores.options();

        torch::Tensor grad_partition = grad_outputs[0];
        torch::Tensor grad_marginals = grad_outputs[1];

        // Initialize gradients
        torch::Tensor grad_arc = torch::zeros({B, n, n}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from partition function
        if (grad_partition.defined() && grad_partition.numel() > 0) {
            // Recompute marginals
            torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor marginals = torch::zeros({B, n, n}, options);
            torch::Tensor grad_T = torch::zeros({B}, options);

            orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &beta_C_R, &beta_C_L, &beta_I_R, &beta_I_L, &marginals, &grad_T});
            orihime::eisner::backward(
                arc_scores.data_ptr<float>(),
                C_R.data_ptr<float>(),
                C_L.data_ptr<float>(),
                I_R.data_ptr<float>(),
                I_L.data_ptr<float>(),
                beta_C_R.data_ptr<float>(),
                beta_C_L.data_ptr<float>(),
                beta_I_R.data_ptr<float>(),
                beta_I_L.data_ptr<float>(),
                marginals.data_ptr<float>(),
                grad_T.data_ptr<float>(),
                lengths_ptr,
                B, n, temp_val
            );

            grad_arc += grad_partition.view({B, 1, 1}) * marginals;
            total_grad_T += (grad_partition * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from marginals via HVP
        if (grad_marginals.defined() && grad_marginals.numel() > 0) {
            grad_marginals = grad_marginals.contiguous().to(torch::kFloat32);

            torch::Tensor d_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor d_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor HVP = torch::zeros({B, n, n}, options);

            orihime::common::record_streams_current({&arc_scores, &grad_marginals, &C_R, &C_L, &I_R, &I_L, &d_C_R, &d_C_L, &d_I_R, &d_I_L, &beta_C_R, &beta_C_L, &beta_I_R, &beta_I_L, &d_beta_C_R, &d_beta_C_L, &d_beta_I_R, &d_beta_I_L, &HVP});
            orihime::eisner::hvp(
                arc_scores.data_ptr<float>(),
                grad_marginals.data_ptr<float>(),
                C_R.data_ptr<float>(),
                C_L.data_ptr<float>(),
                I_R.data_ptr<float>(),
                I_L.data_ptr<float>(),
                d_C_R.data_ptr<float>(),
                d_C_L.data_ptr<float>(),
                d_I_R.data_ptr<float>(),
                d_I_L.data_ptr<float>(),
                beta_C_R.data_ptr<float>(),
                beta_C_L.data_ptr<float>(),
                beta_I_R.data_ptr<float>(),
                beta_I_L.data_ptr<float>(),
                d_beta_C_R.data_ptr<float>(),
                d_beta_C_L.data_ptr<float>(),
                d_beta_I_R.data_ptr<float>(),
                d_beta_I_L.data_ptr<float>(),
                HVP.data_ptr<float>(),
                lengths_ptr,
                B, n, temp_val
            );

            grad_arc += HVP;

            // r71: marginals -> temperature. The marginals depend on the
            // temperature, so a loss through the marginals output contributes
            // to the temperature gradient via the dP/dT parameter Jacobian.
            // Mirrors the partition-path term above and the CPU Function, so
            // CPU and CUDA return the same complete temperature gradient.
            torch::Tensor U_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor U_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor U_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor U_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_R_w = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_L_w = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_R_w = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_L_w = torch::zeros({B, n, n}, options);
            torch::Tensor W_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor W_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor W_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor W_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor dP_dT = torch::zeros({B, n, n}, options);

            orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &U_C_R, &U_C_L, &U_I_R, &U_I_L, &beta_C_R_w, &beta_C_L_w, &beta_I_R_w, &beta_I_L_w, &W_C_R, &W_C_L, &W_I_R, &W_I_L, &dP_dT});
            orihime::eisner::param_grad(
                arc_scores.data_ptr<float>(),
                C_R.data_ptr<float>(),
                C_L.data_ptr<float>(),
                I_R.data_ptr<float>(),
                I_L.data_ptr<float>(),
                U_C_R.data_ptr<float>(),
                U_C_L.data_ptr<float>(),
                U_I_R.data_ptr<float>(),
                U_I_L.data_ptr<float>(),
                beta_C_R_w.data_ptr<float>(),
                beta_C_L_w.data_ptr<float>(),
                beta_I_R_w.data_ptr<float>(),
                beta_I_L_w.data_ptr<float>(),
                W_C_R.data_ptr<float>(),
                W_C_L.data_ptr<float>(),
                W_I_R.data_ptr<float>(),
                W_I_L.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths_ptr,
                B, n, temp_val
            );

            total_grad_T += (grad_marginals * dP_dT).sum().reshape({1});
        }

        return {grad_arc, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Wrapper Functions
// =============================================================================

std::vector<torch::Tensor> soft_eisner_cuda(
    torch::Tensor arc_scores,
    torch::Tensor temperature,
    c10::optional<torch::Tensor> lengths
) {
    return SoftEisnerCUDAFunction::apply(arc_scores, temperature, lengths);
}

torch::Tensor soft_eisner_float_cuda(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    ORIHIME_CUDA_GUARD(arc_scores);
    auto options = arc_scores.options();
    auto temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    auto results = SoftEisnerCUDAFunction::apply(arc_scores, temp_t, lengths);
    return results[0];
}

std::tuple<torch::Tensor, torch::Tensor> soft_eisner_with_grads_cuda(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cuda(arc_scores);
    ORIHIME_CUDA_GUARD(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cuda(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &partition});
    orihime::eisner::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor marginals = torch::zeros({B, n, n}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &beta_C_R, &beta_C_L, &beta_I_R, &beta_I_L, &marginals, &grad_T});
    orihime::eisner::backward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        beta_C_R.data_ptr<float>(),
        beta_C_L.data_ptr<float>(),
        beta_I_R.data_ptr<float>(),
        beta_I_L.data_ptr<float>(),
        marginals.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return std::make_tuple(partition, marginals);
}

torch::Tensor soft_eisner_hvp_cuda(
    torch::Tensor arc_scores,
    torch::Tensor V,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cuda(arc_scores);
    validate_eisner_tangent_cuda(arc_scores, V);
    ORIHIME_CUDA_GUARD(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cuda(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &partition});
    orihime::eisner::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    torch::Tensor d_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor d_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor HVP = torch::zeros({B, n, n}, options);

    orihime::common::record_streams_current({&arc_scores, &V, &C_R, &C_L, &I_R, &I_L, &d_C_R, &d_C_L, &d_I_R, &d_I_L, &beta_C_R, &beta_C_L, &beta_I_R, &beta_I_L, &d_beta_C_R, &d_beta_C_L, &d_beta_I_R, &d_beta_I_L, &HVP});
    orihime::eisner::hvp(
        arc_scores.data_ptr<float>(),
        V.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        d_C_R.data_ptr<float>(),
        d_C_L.data_ptr<float>(),
        d_I_R.data_ptr<float>(),
        d_I_L.data_ptr<float>(),
        beta_C_R.data_ptr<float>(),
        beta_C_L.data_ptr<float>(),
        beta_I_R.data_ptr<float>(),
        beta_I_L.data_ptr<float>(),
        d_beta_C_R.data_ptr<float>(),
        d_beta_C_L.data_ptr<float>(),
        d_beta_I_R.data_ptr<float>(),
        d_beta_I_L.data_ptr<float>(),
        HVP.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return HVP;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> soft_eisner_backward_full_cuda(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cuda(arc_scores);
    ORIHIME_CUDA_GUARD(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cuda(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &partition});
    orihime::eisner::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor marginals = torch::zeros({B, n, n}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &beta_C_R, &beta_C_L, &beta_I_R, &beta_I_L, &marginals, &grad_T});
    orihime::eisner::backward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        beta_C_R.data_ptr<float>(),
        beta_C_L.data_ptr<float>(),
        beta_I_R.data_ptr<float>(),
        beta_I_L.data_ptr<float>(),
        marginals.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return std::make_tuple(partition, marginals, grad_T);
}

torch::Tensor soft_eisner_param_jacobian_cuda(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cuda(arc_scores);
    ORIHIME_CUDA_GUARD(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cuda(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    // Forward pass
    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &partition});
    orihime::eisner::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    // Workspace for param_grad
    torch::Tensor U_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor U_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor U_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor U_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_R_w = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L_w = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R_w = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L_w = torch::zeros({B, n, n}, options);
    torch::Tensor W_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor W_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor W_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor W_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor dP_dT = torch::zeros({B, n, n}, options);

    orihime::common::record_streams_current({&arc_scores, &C_R, &C_L, &I_R, &I_L, &U_C_R, &U_C_L, &U_I_R, &U_I_L, &beta_C_R_w, &beta_C_L_w, &beta_I_R_w, &beta_I_L_w, &W_C_R, &W_C_L, &W_I_R, &W_I_L, &dP_dT});
    orihime::eisner::param_grad(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        U_C_R.data_ptr<float>(),
        U_C_L.data_ptr<float>(),
        U_I_R.data_ptr<float>(),
        U_I_L.data_ptr<float>(),
        beta_C_R_w.data_ptr<float>(),
        beta_C_L_w.data_ptr<float>(),
        beta_I_R_w.data_ptr<float>(),
        beta_I_L_w.data_ptr<float>(),
        W_C_R.data_ptr<float>(),
        W_C_L.data_ptr<float>(),
        W_I_R.data_ptr<float>(),
        W_I_L.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return dP_dT;
}

// =============================================================================
// Namespaced API Wrappers (eisner_*)
// =============================================================================

// eisner::forward - returns (value, marginals)
std::vector<torch::Tensor> eisner_forward_cuda(
    torch::Tensor arc_scores,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    ORIHIME_CUDA_GUARD(arc_scores);
    auto temp_t = torch::tensor(
        {static_cast<float>(temp)},
        arc_scores.options()
    );
    return soft_eisner_cuda(arc_scores, temp_t, lengths);
}

// eisner::forward_t - tensor parameter version
std::vector<torch::Tensor> eisner_forward_t_cuda(
    torch::Tensor arc_scores,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_eisner_cuda(arc_scores, temp, lengths);
}

// eisner::value_grad_params - returns grad_temp per batch
torch::Tensor eisner_value_grad_params_cuda(
    torch::Tensor arc_scores,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_eisner_backward_full_cuda(arc_scores, temp, lengths);
    return std::get<2>(result);
}

// eisner::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor> eisner_marginals_backward_cuda(
    torch::Tensor arc_scores,
    torch::Tensor grad_marginals,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    torch::Tensor grad_arc_scores =
        soft_eisner_hvp_cuda(arc_scores, grad_marginals, temp, lengths);
    torch::Tensor dP_dT =
        soft_eisner_param_jacobian_cuda(arc_scores, temp, lengths);
    torch::Tensor grad_temp =
        (grad_marginals * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_arc_scores, grad_temp);
}

// eisner::marginals_hvp - Hessian-vector product
torch::Tensor eisner_marginals_hvp_cuda(
    torch::Tensor arc_scores,
    torch::Tensor v,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_eisner_hvp_cuda(arc_scores, v, temp, lengths);
}

// eisner::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor eisner_marginals_grad_temp_cuda(
    torch::Tensor arc_scores,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_eisner_param_jacobian_cuda(arc_scores, temp, lengths);
}

// =============================================================================
// Library Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(orihime, CUDA, m) {
    m.impl("soft_eisner", soft_eisner_cuda);
    m.impl("soft_eisner_float", soft_eisner_float_cuda);
    m.impl("soft_eisner_with_grads", soft_eisner_with_grads_cuda);
    m.impl("soft_eisner_hvp", soft_eisner_hvp_cuda);
    m.impl("soft_eisner_backward_full", soft_eisner_backward_full_cuda);
    m.impl("soft_eisner_param_jacobian", soft_eisner_param_jacobian_cuda);

    // Namespaced API
    m.impl("eisner_forward", eisner_forward_cuda);
    m.impl("eisner_forward_t", eisner_forward_t_cuda);
    m.impl("eisner_value_grad_params", eisner_value_grad_params_cuda);
    m.impl("eisner_marginals_backward", eisner_marginals_backward_cuda);
    m.impl("eisner_marginals_hvp", eisner_marginals_hvp_cuda);
    m.impl("eisner_marginals_grad_temp", eisner_marginals_grad_temp_cuda);
}

TORCH_LIBRARY_IMPL(orihime, AutogradCUDA, m) {
    m.impl("soft_eisner", soft_eisner_cuda);
    m.impl("soft_eisner_float", soft_eisner_float_cuda);

    // Namespaced API - autograd versions
    m.impl("eisner_forward", eisner_forward_cuda);
    m.impl("eisner_forward_t", eisner_forward_t_cuda);
}

#endif // USE_TORCH_LIBRARY
