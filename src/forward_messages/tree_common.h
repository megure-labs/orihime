// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "forward_messages/torch_common.h"

namespace orihime::forward_messages {
struct TreeShape { int b, n, planes; };

inline TreeShape validate_tree(const Tensor& x, const Tensor& leaves,
                              const Tensor& temperature, const Tensor& lengths,
                              int64_t mode, bool gpu) {
  TORCH_CHECK(mode == 0 || mode == 1, "invalid tree algorithm");
  TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous(),
              "tree scores must be contiguous float32");
  TORCH_CHECK(gpu ? x.is_cuda() : x.device().is_cpu(), "unsupported input device");
  TORCH_CHECK(x.dim() == (mode == 0 ? 4 : 3), "invalid tree score rank");
  const int64_t b=x.size(0), n=x.size(1);
  TORCH_CHECK(b <= INT32_MAX && n > 0 && n <= 46340,
              "tree dimensions exceed native indexing limits");
  TORCH_CHECK(x.size(2) == n && (mode != 0 || x.size(3) == n),
              "tree score axes must have equal lengths");
  const int planes=mode == 0 ? 1 : 4;
  const int64_t per_batch=mode == 0 ? n*n*n : planes*n*n;
  TORCH_CHECK(b == 0 || per_batch <= INT32_MAX / b,
              "tree tables exceed int32 indexing");
  TORCH_CHECK(leaves.device() == x.device() && leaves.scalar_type() == at::kFloat &&
              leaves.is_contiguous(), "leaf scores must be contiguous float32 on the input device");
  TORCH_CHECK(mode == 0 ? leaves.sizes() == at::IntArrayRef({b,n}) : leaves.numel() == 0,
              "CKY requires leaf scores [B,N]; Eisner has no leaf scores");
  TORCH_CHECK(temperature.device() == x.device() && temperature.scalar_type() == at::kFloat &&
              temperature.is_contiguous(), "temperature must be contiguous float32 on the input device");
  TORCH_CHECK(temperature.numel() == 1 || (mode == 0 &&
              temperature.sizes() == at::IntArrayRef({b,n,n})),
              "temperature must be scalar, or [B,N,N] for CKY");
  TORCH_CHECK(at::isfinite(temperature).all().item<bool>() &&
              (temperature > 0).all().item<bool>(), "temperature must be finite and positive");
  TORCH_CHECK(lengths.device() == x.device() && lengths.scalar_type() == at::kInt &&
              lengths.is_contiguous() && lengths.sizes() == at::IntArrayRef({b}),
              "tree lengths must be contiguous int32 [B] on the input device");
  auto host=lengths.cpu(); const int* l=host.data_ptr<int>();
  for (int64_t i=0; i<b; ++i)
    TORCH_CHECK(l[i] >= 1 && l[i] <= n, "tree lengths must be between 1 and N");
  for (const Tensor* q : {&x,&leaves})
    TORCH_CHECK(!at::isnan(*q).any().item<bool>() && !at::isposinf(*q).any().item<bool>(),
                "tree scores contain NaN or positive infinity");
  return {int(b),int(n),planes};
}

inline std::tuple<Tensor,Tensor> finish_tree(const Tensor& planes, Tensor value,
                                           const Tensor& lengths, int64_t mode) {
  if (mode == 0)
    value=planes.select(0,0).select(1,0).gather(1,(lengths.to(at::kLong)-1).unsqueeze(1)).squeeze(1);
  return {planes.permute({1,0,2,3}).contiguous(),value};
}
} // namespace orihime::forward_messages
