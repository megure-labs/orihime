# Dynamic Time Warping implementation

Native implementation of the cost-native global-warping operator available as
`ohm.dtw`. Paths move diagonally, vertically, or horizontally through a pairwise
cost matrix and may be restricted by a Sakoe–Chiba bandwidth.

## Recurrence

For cost `c[i,j]` and temperature `T`:

```text
A[i,j] = c[i,j] + softmin_T(A[i-1,j-1], A[i-1,j], A[i,j-1])
A[0,0] = 0
A[i,0] = +inf  for i > 0
A[0,j] = +inf  for j > 0
S      = A[L1,L2]
```

When a bandwidth is present, cells outside the permitted diagonal band are
unreachable.

## State and memory layout

- `costs`: row-major `[B, L1, L2]`.
- `alpha` and `beta`: flattened row-major
  `[B, (L1 + 1) * (L2 + 1)]` tables.
- value and temperature gradient: `[B]`.
- warping map, map cotangent, and temperature sensitivity: `[B, L1, L2]`.

Lengths select active prefixes. A one-sided empty instance has no valid
corner-to-corner path.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and terminal soft cost |
| backward | warping map and value gradient for temperature |
| HVP | map directional derivative with respect to costs |
| parameter sensitivity | full map derivative for temperature |

Public API: `ohm.dtw`, `ohm.dtw_value`, `ohm.dtw_entropy`, and `ohm.ops.dtw`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and uses compensated soft-min
reductions. CUDA and HIP evaluate the table in anti-diagonal dependency order
on the current PyTorch stream. Banding changes reachability but not the stored
table shape.

## See also

- [Dynamic Time Warping guide](../../docs/algorithms/dtw.md)
- [Source architecture](../ARCHITECTURE.md)
