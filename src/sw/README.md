# Smith–Waterman linear-gap implementation

Native implementation of the score-native local-alignment operator available as
`ohm.sw`. A path may restart inside either sequence, and every unmatched symbol
receives the same `gap_score`.

## Recurrence

For pair score `s[i,j]`, gap score `g`, and temperature `T`:

```text
A[i,j] = LSE_T(
    A[i-1,j-1] + s[i,j],
    A[i-1,j]   + g,
    A[i,j-1]   + g,
    0
)
S = LSE_T({A[i,j]})
```

The zero-valued option starts a new local alignment. The partition `S` ranges
over every terminal cell, so a path may also end anywhere.

## State and memory layout

- `pair_scores`: row-major `[B, L1, L2]`.
- `alpha` and `beta`: flattened row-major
  `[B, (L1 + 1) * (L2 + 1)]` tables.
- partition and scalar-parameter gradients: `[B]`.
- alignment map, map cotangents, and map sensitivities: `[B, L1, L2]`.

Lengths select an active prefix for each sequence. Padded map cells are zero.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and partition value |
| backward | alignment map plus value gradients for gap and temperature |
| HVP | map directional derivative with respect to pair scores |
| parameter sensitivity | full map derivative for gap or temperature |

Public API: `ohm.sw`, `ohm.sw_value`, `ohm.sw_entropy`, and `ohm.ops.sw`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and uses compensated summation.
CUDA and HIP evaluate the two-dimensional table in anti-diagonal dependency
order on the current PyTorch stream. CPU, CUDA, and HIP use the same state
layout and numerical rules.

## See also

- [Smith–Waterman guide](../../docs/algorithms/sw.md)
- [Affine-gap implementation](../sw_affine/README.md)
- [Source architecture](../ARCHITECTURE.md)
