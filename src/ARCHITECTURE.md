# Source architecture

This document defines the source layout and registration pattern for Orihime
operators. Follow it when adding an operator.

## Directory structure

Each operator lives in its own self-contained directory:

```
src/
├── ARCHITECTURE.md          # This file
├── common/                  # Shared utilities (all operators use these)
│   ├── numerics.cuh         # CUDA numerical utilities (LSE, safe_exp, etc.)
│   ├── numerics.hiph        # HIP numerical utilities
│   ├── numerics.h           # CPU numerical utilities
│   ├── softmax.cuh          # CUDA softmax primitives
│   ├── softmax.hiph         # HIP softmax primitives
│   ├── reduce.cuh           # CUDA reduction primitives
│   ├── reduce.hiph          # HIP reduction primitives
│   ├── torch_utils.h        # PyTorch tensor validation macros
│   ├── cuda_utils.h         # CUDA error checking, stream utilities
│   └── hip_utils.h          # HIP error checking, stream utilities
│
├── sw/                      # Smith-Waterman (linear gap) - reference implementation
│   ├── README.md            # Algorithm documentation
│   ├── registry.cpp         # Operator schema definitions (m.def)
│   ├── kernels_gpu.cu       # CUDA kernel implementations
│   ├── kernels_gpu.cuh      # CUDA kernel declarations
│   ├── kernels_gpu.hip      # HIP kernel implementations
│   ├── kernels_gpu.hiph     # HIP kernel declarations
│   ├── kernels_cpu.cpp      # CPU kernel implementations
│   ├── kernels_cpu.h        # CPU kernel declarations
│   ├── torch_cuda.cpp       # CUDA PyTorch bindings + autograd
│   ├── torch_hip.cpp        # HIP PyTorch bindings + autograd
│   └── torch_cpu.cpp        # CPU PyTorch bindings + autograd
│
├── sw_affine/               # Smith-Waterman (affine gap) - same structure
│   ├── README.md
│   ├── registry.cpp
│   ├── kernels_gpu.cu
│   ├── kernels_gpu.cuh
│   ├── kernels_gpu.hip
│   ├── kernels_gpu.hiph
│   ├── kernels_cpu.cpp
│   ├── kernels_cpu.h
│   ├── torch_cuda.cpp
│   ├── torch_hip.cpp
│   └── torch_cpu.cpp
│
└── <other_operator>/        # Future operators follow same pattern
```

## File responsibilities

### `registry.cpp` - operator schemas

Defines the native dispatcher schemas with `TORCH_LIBRARY_FRAGMENT`:

```cpp
#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(orihime, m) {
    // Core operators
    m.def("soft_<op>(Tensor scores, ...) -> Tensor[]");
    m.def("soft_<op>_float(Tensor scores, float param, ...) -> Tensor[]");
    m.def("soft_<op>_with_grads(...) -> (Tensor, ...)");
    m.def("soft_<op>_hvp(...) -> Tensor");
    m.def("soft_<op>_param_jacobian(...) -> Tensor");
    m.def("soft_<op>_backward_full(...) -> (Tensor, ...)");

    // Explicit-operation schemas
    m.def("<op>_forward(...) -> Tensor[]");
    m.def("<op>_forward_t(...) -> Tensor[]");  // tensor params version
    m.def("<op>_marginals_backward(...) -> (Tensor, ...)");
    m.def("<op>_marginals_hvp(...) -> Tensor");
}

#endif
```

### `kernels_gpu.cu` / `kernels_gpu.cuh` - CUDA kernels

These files implement the CUDA kernels without PyTorch dependencies:

```cpp
// kernels_gpu.cuh - declarations
void <op>_forward(const float* scores, float* alpha, float* partition,
                  const int* lengths, int B, int L1, int L2,
                  float param, float temperature);

void <op>_backward(const float* alpha, const float* scores, ...);

void <op>_hvp(const float* alpha, const float* scores, const float* tangent, ...);

void <op>_param_grad(const float* alpha, ..., int param_type);
```

### `kernels_gpu.hip` / `kernels_gpu.hiph` - HIP kernels

The AMD implementation uses the same `kernels_gpu` basename and operator
interface, retaining `.hip` and `.hiph` so Meson can compile it with `hipcc`.
All fourteen public operator families ship a HIP implementation.

### `kernels_cpu.cpp` / `kernels_cpu.h` - CPU kernels

The CPU kernels use the corresponding interface:

```cpp
// kernels_cpu.h - declarations (same signatures as CUDA)
void <op>_forward_cpu(...);
void <op>_backward_cpu(...);
void <op>_hvp_cpu(...);
void <op>_param_grad_cpu(...);
```

### `torch_cuda.cpp` - CUDA PyTorch bindings

Connects the CUDA kernels to PyTorch autograd and the dispatcher:

```cpp
#include <torch/extension.h>
#include "common/torch_utils.h"
#include "common/cuda_utils.h"
#include "<op>/kernels_gpu.cuh"

// 1. Autograd Function class
class Soft<Op>CUDAFunction : public torch::autograd::Function<Soft<Op>CUDAFunction> {
    static tensor_list forward(AutogradContext* ctx, ...);
    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

// 2. Python interface functions
std::vector<torch::Tensor> soft_<op>_cuda(...);
std::vector<torch::Tensor> soft_<op>_cuda_float(...);
// ... other functions ...

// 3. Namespaced API wrappers
std::vector<torch::Tensor> <op>_forward_cuda(...);
// ... other wrappers ...

// 4. Dispatcher registration
#ifdef USE_TORCH_LIBRARY
TORCH_LIBRARY_IMPL(orihime, CUDA, m) {
    m.impl("soft_<op>", soft_<op>_cuda);
    m.impl("<op>_forward", <op>_forward_cuda);
    // ...
}

TORCH_LIBRARY_IMPL(orihime, AutogradCUDA, m) {
    m.impl("soft_<op>", soft_<op>_cuda);
    // ...
}
#endif
```

### `torch_hip.cpp` - HIP PyTorch bindings

Connects HIP kernels through PyTorch's CUDA and AutogradCUDA dispatch keys,
which PyTorch intentionally also uses for ROCm builds. CUDA-spelled PyTorch
types and checks in this file are compatibility API, not CUDA source.

### `torch_cpu.cpp` - CPU PyTorch bindings

Uses the same binding pattern for the CPU kernels:

```cpp
#include "common/torch_utils.h"
#include "<op>/kernels_cpu.h"

// Same pattern: Autograd class, interface functions, registration
TORCH_LIBRARY_IMPL(orihime, CPU, m) { ... }
TORCH_LIBRARY_IMPL(orihime, AutogradCPU, m) { ... }
```

### `README.md` - implementation documentation

Documents the recurrence, native operations, memory layout, files, and
backends needed to maintain the operator:

```markdown
# <Operator name> implementation

Brief description.

## Recurrence

Recurrence relations with mathematical notation.

## State and memory layout

Input, chart, workspace, and output shapes.

## Native operations

Forward, backward, HVP, and parameter-sensitivity responsibilities.

## Files and backends

| File | Description |
|------|-------------|
| ... | ... |

## See also

Link to the public algorithm guide and source architecture.
```

## Shared utilities (`common/`)

Use the shared utilities below instead of duplicating them in an operator.

### `torch_utils.h`
- `ORIHIME_CHECK_INPUT_CUDA(x)` - Validate CUDA tensor
- `ORIHIME_CHECK_INPUT_CPU(x)` - Validate CPU tensor
- `ORIHIME_CHECK_CONTIGUOUS(x)` - Check contiguity
- `make_default_lengths_2d(B, L1, L2, device)` - Create default lengths tensor

### `numerics.cuh` / `numerics.hiph` / `numerics.h`
- `NINF` - Negative infinity constant (-1e30f)
- `safe_exp(x)` - Clamped exponential
- `lse2(a, b)` - Two-argument log-sum-exp
- `lse3(a, b, c)` - Three-argument log-sum-exp
- `lse4(a, b, c, d)` - Four-argument log-sum-exp
- `lse_T(T, ...)` - Temperature-scaled log-sum-exp

### `cuda_utils.h`
- `CUDA_CHECK(expr)` - CUDA error checking
- Stream and device utilities

### `hip_utils.h`
- HIP launch checking through the PyTorch ROCm compatibility API
- HIP stream and device utilities

## Adding an operator

1. Create `src/<op>/`.

2. Add the kernels:
   - `kernels_gpu.cu` + `kernels_gpu.cuh` (CUDA)
   - `kernels_gpu.hip` + `kernels_gpu.hiph` (HIP)
   - `kernels_cpu.cpp` + `kernels_cpu.h` (CPU)

3. Add `registry.cpp` with the dispatcher schemas.

4. Add the bindings:
   - `torch_cuda.cpp` (CUDA bindings + autograd)
   - `torch_hip.cpp` (HIP bindings + autograd)
   - `torch_cpu.cpp` (CPU bindings + autograd)

5. Document the algorithm in `README.md`.

6. Add the files to `meson.build`:
   ```meson
   cpp_sources = files(
     # <op> module (fully self-contained)
     'src/<op>/registry.cpp',
     'src/<op>/torch_cuda.cpp',
     'src/<op>/torch_cpu.cpp',
     'src/<op>/kernels_cpu.cpp',
   )

   cuda_sources = files(
     'src/<op>/kernels_gpu.cu',
   )
   ```

7. Add `orihime/<op>.py` and `orihime/ops/<op>.py`.

## Design principles

1. Keep each operator's implementation in its own directory.
2. Use the same file names across operators.
3. Put common code in `common/` rather than duplicating it.
4. Keep PyTorch dependencies out of kernels and kernel logic out of bindings.
5. Implement the derivatives required by the public API.
6. Provide CPU, CUDA, and HIP implementations for every public operator.

## PyTorch integration

Orihime uses `TORCH_LIBRARY_FRAGMENT` and `TORCH_LIBRARY_IMPL` for backend
dispatch, autograd registration, and the documented `torch.compile` support.

Each module registers in the same `orihime` namespace, which PyTorch merges at
runtime.
