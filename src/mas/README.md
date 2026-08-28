# Monotonic Alignment Search implementation

Native implementation of the score-native frame-to-token alignment operator
available as `ohm.mas`. Each frame either remains on the current token or
advances exactly one token, and the terminal state requires every active token
to be covered.

## Recurrence

For frame/token score `s[t,j]` and temperature `T`:

```text
A[0,0] = s[0,0]
A[t,0] = A[t-1,0] + s[t,0]
A[t,j] = s[t,j] + LSE_T(A[t-1,j], A[t-1,j-1])
Z      = A[frames-1,tokens-1]
```

States with more tokens than consumed frames are unreachable. A valid active
instance therefore requires `frames >= tokens`.

## State and memory layout

- `scores`: row-major `[B, frames, tokens]`.
- `alpha` and `beta`: row-major `[B, frames, tokens]` tables.
- value and temperature gradient: `[B]`.
- assignment map, map cotangent, and temperature sensitivity:
  `[B, frames, tokens]`.

Lengths provide active frame and token prefixes. Padded rows and columns are
zero in the public map.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and terminal soft score |
| backward | frame-to-token map and value gradient for temperature |
| HVP | map directional derivative with respect to scores |
| parameter sensitivity | full map derivative for temperature |

Public API: `ohm.mas`, `ohm.mas_value`, `ohm.mas_entropy`, and `ohm.ops.mas`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and advances the chart one frame at
a time. CUDA and HIP preserve the same temporal dependency order on the
current PyTorch stream while evaluating legal token states in parallel.

## See also

- [Monotonic Alignment Search guide](../../docs/algorithms/mas.md)
- [Source architecture](../ARCHITECTURE.md)
