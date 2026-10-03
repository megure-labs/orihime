// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <cmath>
#include <limits>
#include <tuple>
#include <torch/library.h>

namespace orihime::forward_messages {
using at::Tensor;
struct Shape { int b, n, m, planes; int64_t cells; };

inline Shape validate(const Tensor& x, const Tensor& lengths,
                      const Tensor& topology, int64_t algorithm,
                      double p0, double p1, double p2, double t,
                      int64_t band, bool gpu) {
  TORCH_CHECK(x.dim() == 3 && x.scalar_type() == at::kFloat && x.is_contiguous(),
              "inputs must be contiguous float32 [B,N,M]");
  TORCH_CHECK(gpu ? x.is_cuda() : x.device().is_cpu(), "unsupported input device");
  TORCH_CHECK(algorithm >= 0 && algorithm < 12, "invalid grid algorithm");
  TORCH_CHECK(std::isfinite(t) && float(t) > 0 && t <= std::numeric_limits<float>::max(),
              "temperature must be finite and positive in float32");
  TORCH_CHECK(std::isfinite(p0) && std::isfinite(p1) && std::isfinite(p2) &&
              std::abs(p0) <= std::numeric_limits<float>::max() &&
              std::abs(p1) <= std::numeric_limits<float>::max() &&
              std::abs(p2) <= std::numeric_limits<float>::max(),
              "parameters must be finite in float32");
  TORCH_CHECK(band >= -1 && band <= std::numeric_limits<int>::max(), "invalid bandwidth");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() &&
              x.size(1) < std::numeric_limits<int>::max() &&
              x.size(2) < std::numeric_limits<int>::max(), "dimensions exceed int32");
  const int64_t cells = (x.size(1)+1)*(x.size(2)+1);
  const int planes = algorithm == 1 || algorithm == 2 || algorithm == 3 || algorithm == 5 ? 3 : 1;
  TORCH_CHECK(cells <= std::numeric_limits<int>::max()/planes &&
              x.size(0)*planes*cells <= std::numeric_limits<int>::max(), "table exceeds int32 indexing");
  TORCH_CHECK(lengths.device() == x.device() && lengths.scalar_type() == at::kInt &&
              lengths.is_contiguous() && lengths.sizes() == at::IntArrayRef({x.size(0),2}),
              "lengths must be contiguous int32 [B,2] on the input device");
  auto host = lengths.cpu();
  const int* l = host.data_ptr<int>();
  for (int64_t b=0; b<x.size(0); ++b) {
    TORCH_CHECK(l[2*b] >= 0 && l[2*b] <= x.size(1) && l[2*b+1] >= 0 && l[2*b+1] <= x.size(2),
                "lengths entries must be within the input shape");
    if (algorithm == 6 || algorithm == 11)
      TORCH_CHECK((l[2*b] == 0) == (l[2*b+1] == 0), "one-sided empty problem has no path");
    if (algorithm == 11)
      TORCH_CHECK(l[2*b+1] > 0 && l[2*b] >= l[2*b+1], "MAS requires frame length >= token length > 0");
  }
  TORCH_CHECK(topology.device() == x.device() && topology.is_contiguous(),
              "topology must be contiguous on the input device");
  if (algorithm == 9) {
    TORCH_CHECK(topology.scalar_type() == at::kFloat && topology.sizes() == x.sizes(),
                "OSA topology must be float32 [B,N,M]");
    auto h = topology.cpu(); const float* q = h.data_ptr<float>();
    for (int64_t b=0; b<x.size(0); ++b)
      for (int64_t i=0; i<x.size(1); ++i)
        for (int64_t j=0; j<x.size(2); ++j) {
          const float v = q[(b*x.size(1)+i)*x.size(2)+j];
          TORCH_CHECK(v == 0 || v == 1, "OSA topology must be binary");
          TORCH_CHECK(v == 0 || (i > 0 && j > 0 && i < l[2*b] && j < l[2*b+1]),
                      "OSA topology is active at a boundary or padded cell");
        }
  } else if (algorithm == 10) {
    TORCH_CHECK(topology.scalar_type() == at::kInt && topology.sizes() ==
                at::IntArrayRef({x.size(0),x.size(1),x.size(2),2}), "invalid Damerau topology");
    auto h = topology.cpu(); const int* q = h.data_ptr<int>();
    for (int64_t b=0; b<x.size(0); ++b)
      for (int64_t i=0; i<x.size(1); ++i)
        for (int64_t j=0; j<x.size(2); ++j) {
          const int64_t k = 2*((b*x.size(1)+i)*x.size(2)+j);
          const int pi = q[k], pj = q[k+1];
          TORCH_CHECK((pi == -1 && pj == -1) ||
                      (pi >= 0 && pj >= 0 && pi < i && pj < j && i < l[2*b] && j < l[2*b+1]),
                      "Damerau topology must contain (-1,-1) or an earlier active predecessor");
        }
  } else {
    TORCH_CHECK(topology.numel() == 0, "this algorithm has no explicit topology");
  }
  const bool cost = algorithm == 6 || algorithm == 8 || algorithm == 9 || algorithm == 10;
  TORCH_CHECK(!at::isnan(x).any().item<bool>() &&
              !(cost ? at::isneginf(x) : at::isposinf(x)).any().item<bool>(),
              "inputs contain NaN or an incorrectly oriented infinity");
  return {int(x.size(0)), int(x.size(1)), int(x.size(2)), planes, cells};
}
} // namespace orihime::forward_messages
