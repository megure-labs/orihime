# Needleman–Wunsch linear-gap implementation

Native implementation of the score-native global-alignment operator available
as `ohm.nw`. Every active position belongs to an end-to-end path, and every
unmatched symbol receives the same `gap_score`.

## Recurrence

For pair score `s[i,j]`, gap score `g`, and temperature `T`:

```text
A[i,j] = LSE_T(
    A[i-1,j-1] + s[i,j],
    A[i-1,j]   + g,
    A[i,j-1]   + g
)
A[i,0] = i * g
A[0,j] = j * g
S      = A[L1,L2]
```

Unlike Smith–Waterman, there is no restart transition and only the final cell
terminates a path.

## State and memory layout

- `pair_scores`: row-major `[B, L1, L2]`.
- `alpha` and `beta`: flattened row-major
  `[B, (L1 + 1) * (L2 + 1)]` tables.
- value and scalar-parameter gradients: `[B]`.
- alignment map, map cotangents, and map sensitivities: `[B, L1, L2]`.

Lengths select an active prefix for each sequence. Padded map cells are zero.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and terminal value |
| backward | alignment map plus value gradients for gap and temperature |
| HVP | map directional derivative with respect to pair scores |
| parameter sensitivity | full map derivative for gap or temperature |

Public API: `ohm.nw`, `ohm.nw_value`, `ohm.nw_entropy`, and `ohm.ops.nw`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and uses compensated summation.
CUDA and HIP evaluate the table in anti-diagonal dependency order on the
current PyTorch stream. All three backends share the same boundary conditions.

## See also

- [Needleman–Wunsch guide](../../docs/algorithms/nw.md)
- [Affine-gap implementation](../nw_affine/README.md)
- [Source architecture](../ARCHITECTURE.md)
