# Saigo–Vert affine-gap implementation

Native implementation of the affine-gap Saigo–Vert local-alignment operator
available as `ohm.sv_affine`. It uses separate match, insertion, and deletion
states while counting each monotone matched-pair skeleton once.

## Recurrence

For pair score `s[i,j]`, gap-open score `g_o`, gap-extension score `g_e`, and
temperature `T`:

```text
M[i,j] = s[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
I[i,j] = LSE_T(M[i-1,j] + g_o, I[i-1,j] + g_e)
D[i,j] = LSE_T(M[i,j-1] + g_o,
               I[i,j-1] + g_o,
               D[i,j-1] + g_e)
S      = LSE_T(0, {M[i,j]})
```

The `I -> D` edge is charged as a gap opening. There is no `D -> I` edge.
Paths terminate only in `M`, apart from the one explicit empty alignment.

## State and memory layout

- `pair_scores`: row-major `[B, L1, L2]`.
- `alpha` and derivative workspaces: three state planes flattened as
  `[B, 3 * (L1 + 1) * (L2 + 1)]` in `M`, `I`, `D` order.
- partition and scalar-parameter gradients: `[B]`.
- alignment map, map cotangents, and map sensitivities: `[B, L1, L2]`.

Lengths select an active prefix for each sequence. The empty alignment
contributes to the partition function but not the returned map.

## Native operations

| Operation | Result |
| --- | --- |
| forward | three alpha tables and partition value |
| backward | alignment map plus value gradients for both gaps and temperature |
| HVP | map directional derivative with respect to pair scores |
| parameter sensitivity | full map derivative for gap open, gap extension, or temperature |

Public API: `ohm.sv_affine`, `ohm.sv_affine_value`,
`ohm.sv_affine_entropy`, and `ohm.ops.sv_affine`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and uses compensated summation.
CUDA and HIP evaluate the state tables in anti-diagonal dependency order on
the current PyTorch stream. All three backends implement the same asymmetric
state graph.

## See also

- [Saigo–Vert guide](../../docs/algorithms/sv.md)
- [Linear-gap implementation](../sv_linear/README.md)
- [Source architecture](../ARCHITECTURE.md)
