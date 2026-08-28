# Smith–Waterman affine-gap implementation

Native implementation of the score-native local-alignment operator available as
`ohm.sw_affine`. Separate match, insertion, and deletion states distinguish
opening a gap from extending one.

## Recurrence

For pair score `s[i,j]`, gap-open score `g_o`, gap-extension score `g_e`, and
temperature `T`:

```text
M[i,j] = s[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
I[i,j] = LSE_T(M[i-1,j] + g_o, I[i-1,j] + g_e)
D[i,j] = LSE_T(M[i,j-1] + g_o, D[i,j-1] + g_e)
S      = LSE_T({M[i,j]})
```

A gap of length `k` receives `g_o + (k - 1) * g_e`. The restart option and
match-state termination make the alignment local.

## State and memory layout

- `pair_scores`: row-major `[B, L1, L2]`.
- `alpha` and `beta`: three state planes, flattened as
  `[B, 3 * (L1 + 1) * (L2 + 1)]` in `M`, `I`, `D` order.
- partition and scalar-parameter gradients: `[B]`.
- alignment map, map cotangents, and map sensitivities: `[B, L1, L2]`.

Lengths select an active prefix for each sequence. Padded map cells are zero.

## Native operations

| Operation | Result |
| --- | --- |
| forward | three alpha tables and partition value |
| backward | alignment map plus value gradients for both gaps and temperature |
| HVP | map directional derivative with respect to pair scores |
| parameter sensitivity | full map derivative for gap open, gap extension, or temperature |

Public API: `ohm.sw_affine`, `ohm.sw_affine_value`,
`ohm.sw_affine_entropy`, and `ohm.ops.sw_affine`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and uses compensated summation.
CUDA and HIP evaluate each state plane in anti-diagonal dependency order on
the current PyTorch stream. CPU, CUDA, and HIP use the same state layout and
numerical rules.

## See also

- [Smith–Waterman guide](../../docs/algorithms/sw.md)
- [Linear-gap implementation](../sw/README.md)
- [Source architecture](../ARCHITECTURE.md)
