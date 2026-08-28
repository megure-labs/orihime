# Longest Common Subsequence implementation

Native implementation of the score-native operator available as `ohm.lcs`.
Each path may match a pair of positions or skip a position from either input.

## Recurrence

For match score `s[i,j]` and temperature `T`:

```text
A[i,j] = LSE_T(
    A[i-1,j-1] + s[i,j],
    A[i-1,j],
    A[i,j-1]
)
A[i,0] = A[0,j] = 0
S      = A[L1,L2]
```

The returned map contains posterior mass for match transitions. Skip
transitions have no corresponding map entry.

## State and memory layout

- `match_scores`: row-major `[B, L1, L2]`.
- `alpha` and `beta`: flattened row-major
  `[B, (L1 + 1) * (L2 + 1)]` tables.
- value and temperature gradient: `[B]`.
- match map, map cotangent, and temperature sensitivity: `[B, L1, L2]`.

Lengths select an active prefix for each input. Padded map cells are zero.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and terminal soft score |
| backward | match map and value gradient for temperature |
| HVP | map directional derivative with respect to match scores |
| parameter sensitivity | full map derivative for temperature |

Public API: `ohm.lcs`, `ohm.lcs_value`, `ohm.lcs_entropy`, and `ohm.ops.lcs`.

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
current PyTorch stream. All three backends use the same score-native boundary
conditions.

## See also

- [Edit-distance family guide](../../docs/algorithms/edit-distance.md)
- [Source architecture](../ARCHITECTURE.md)
