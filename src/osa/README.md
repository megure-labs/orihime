# Optimal String Alignment implementation

Native implementation of the cost-native restricted-transposition operator
available as `ohm.osa`. It extends Levenshtein distance with selected adjacent
swaps that cannot overlap through repeated editing.

## Recurrence

For substitution cost `c[i,j]`, insertion cost `g_i`, deletion cost `g_d`,
transposition cost `g_t`, and temperature `T`:

```text
A[i,j] = softmin_T(
    A[i-1,j-1] + c[i,j],
    A[i-1,j]   + g_d,
    A[i,j-1]   + g_i,
    A[i-2,j-2] + g_t  if transposition[i,j] is allowed
)
```

Boundary rows and columns carry accumulated deletion and insertion costs, as
in Levenshtein distance. The transposition edge consumes two positions from
each input.

## State and memory layout

- `substitution_costs`: row-major `[B, L1, L2]`.
- native transposition mask: `[B, L1, L2]`; the public boolean
  `allowed_transpositions` argument is normalized before launch.
- `alpha` and `beta`: flattened row-major
  `[B, (L1 + 1) * (L2 + 1)]` tables.
- substitution map, map cotangents, and map sensitivities: `[B, L1, L2]`.

Lengths select an active prefix for each input. Padded map cells are zero.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and terminal soft distance |
| backward | substitution map plus value gradients for three costs and temperature |
| HVP | map directional derivative with respect to substitution costs |
| parameter sensitivity | full map derivative for insertion, deletion, transposition, or temperature |

Public API: `ohm.osa`, `ohm.osa_value`, `ohm.osa_entropy`, and `ohm.ops.osa`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items. CUDA and HIP evaluate the table in
anti-diagonal dependency order on the current PyTorch stream; the two-cell
transposition dependency lies on an earlier diagonal.

## See also

- [Edit-distance family guide](../../docs/algorithms/edit-distance.md)
- [Levenshtein implementation](../lev/README.md)
- [Damerau–Levenshtein implementation](../damerau/README.md)
- [Source architecture](../ARCHITECTURE.md)
