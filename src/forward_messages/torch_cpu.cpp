// SPDX-License-Identifier: Apache-2.0
#include "forward_messages/torch_common.h"
#include "sw/kernels_cpu.h"
#include "sw_affine/kernels_cpu.h"
#include "sv_linear/kernels_cpu.h"
#include "sv_affine/kernels_cpu.h"
#include "nw/kernels_cpu.h"
#include "nw_affine/kernels_cpu.h"
#include "dtw/kernels_cpu.h"
#include "lcs/kernels_cpu.h"
#include "lev/kernels_cpu.h"
#include "osa/kernels_cpu.h"
#include "damerau/kernels_cpu.h"
#include "mas/kernels_cpu.h"

namespace orihime::forward_messages {
namespace {
std::tuple<Tensor,Tensor> run(const Tensor& x, const Tensor& lengths,
                            const Tensor& topology, int64_t mode,
                            double p0, double p1, double p2, double t, int64_t band) {
  const auto z = validate(x,lengths,topology,mode,p0,p1,p2,t,band,false);
  auto alpha = mode == 11 ? at::zeros_like(x) : at::zeros({z.b,z.planes*z.cells},x.options());
  auto value = at::zeros({z.b},x.options());
  if (z.b == 0) return {alpha,value};
  const float* s=x.data_ptr<float>(); float* a=alpha.data_ptr<float>(); float* v=value.data_ptr<float>();
  const int* l=lengths.data_ptr<int>();
  // Launch the original forward only: no posterior buffers or reverse calls.
  switch(mode) {
    case 0: sw_regular_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,t); break;
    case 1: sw_affine_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 2: sv_linear_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,t); break;
    case 3: sv_affine_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 4: nw_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,t); break;
    case 5: nw_affine_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 6: dtw_forward_cpu(s,a,v,l,z.b,z.n,z.m,t,band); break;
    case 7: orihime::lcs::cpu::lcs_forward_cpu(s,a,v,l,z.b,z.n,z.m,t); break;
    case 8: orihime::lev::lev_forward_cpu(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 9: orihime::osa::cpu::osa_forward_cpu(s,topology.data_ptr<float>(),a,v,l,p0,p1,p2,z.b,z.n,z.m,t); break;
    case 10: orihime::damerau::cpu::damerau_forward_cpu(s,topology.data_ptr<int>(),a,v,l,p0,p1,p2,z.b,z.n,z.m,t); break;
    case 11: orihime::mas::cpu::forward(s,a,v,l,z.b,z.n,z.m,t); break;
  }
  return {alpha,value};
}
}
}

#ifdef USE_TORCH_LIBRARY
TORCH_LIBRARY_IMPL(orihime, CPU, m) {
  m.impl("grid_forward_messages", orihime::forward_messages::run);
}
#endif
