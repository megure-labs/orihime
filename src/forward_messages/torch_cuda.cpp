// SPDX-License-Identifier: Apache-2.0
#include "forward_messages/torch_common.h"
#include "sw/kernels_gpu.cuh"
#include "sw_affine/kernels_gpu.cuh"
#include "sv_linear/kernels_gpu.cuh"
#include "sv_affine/kernels_gpu.cuh"
#include "nw/kernels_gpu.cuh"
#include "nw_affine/kernels_gpu.cuh"
#include "dtw/kernels_gpu.cuh"
#include "lcs/kernels_gpu.cuh"
#include "lev/kernels_gpu.cuh"
#include "osa/kernels_gpu.cuh"
#include "damerau/kernels_gpu.cuh"
#include "mas/kernels_gpu.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>

namespace orihime::forward_messages {
namespace {
std::tuple<Tensor,Tensor> run(const Tensor& x, const Tensor& lengths,
                            const Tensor& topology, int64_t mode,
                            double p0, double p1, double p2, double t, int64_t band) {
  const auto z = validate(x,lengths,topology,mode,p0,p1,p2,t,band,true);
  const c10::cuda::CUDAGuard guard(x.device());
  auto alpha = mode == 11 ? at::zeros_like(x) : at::zeros({z.b,z.planes*z.cells},x.options());
  auto value = at::zeros({z.b},x.options());
  if (z.b == 0) return {alpha,value};
  for (const Tensor* q : std::initializer_list<const Tensor*>{&x,&lengths,&topology,&alpha,&value})
    c10::cuda::CUDACachingAllocator::recordStream(q->storage().data_ptr(),at::cuda::getCurrentCUDAStream());
  const float* s=x.data_ptr<float>(); float* a=alpha.data_ptr<float>(); float* v=value.data_ptr<float>();
  const int* l=lengths.data_ptr<int>();
  // Launch the original forward only: no posterior buffers or reverse calls.
  switch(mode) {
    case 0: sw_regular_forward(s,a,v,l,z.b,z.n,z.m,p0,t); break;
    case 1: sw_affine_forward(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 2: sv_linear_forward(s,a,v,l,z.b,z.n,z.m,p0,t); break;
    case 3: sv_affine_forward(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 4: nw_forward(s,a,v,l,z.b,z.n,z.m,p0,t); break;
    case 5: nw_affine_forward(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 6: dtw_forward(s,a,v,l,z.b,z.n,z.m,t,band); break;
    case 7: orihime::lcs::lcs_forward(s,a,v,l,z.b,z.n,z.m,t); break;
    case 8: orihime::lev::lev_forward(s,a,v,l,z.b,z.n,z.m,p0,p1,t); break;
    case 9: orihime::osa::osa_forward(s,topology.data_ptr<float>(),a,v,l,p0,p1,p2,z.b,z.n,z.m,t); break;
    case 10: orihime::damerau::damerau_forward(s,topology.data_ptr<int>(),a,v,l,p0,p1,p2,z.b,z.n,z.m,t); break;
    case 11: orihime::mas::forward(s,a,v,l,z.b,z.n,z.m,t); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {alpha,value};
}
}
}

#ifdef USE_TORCH_LIBRARY
TORCH_LIBRARY_IMPL(orihime, CUDA, m) {
  m.impl("grid_forward_messages", orihime::forward_messages::run);
}
#endif
