# Damerau–Levenshtein implementation

Native implementation of the cost-native unrestricted-transposition operator
available as `ohm.damerau`. A precomputed predecessor table identifies the
earlier chart cell from which each legal transposition begins.

## Recurrence

For substitution cost `c[i,j]`, insertion cost `g_i`, deletion cost `g_d`,
transposition cost `g_t`, and predecessor `(k,l)`:

```text
A[i,j] = softmin_T(
    A[i-1,j-1] + c[i,j],
    A[i-1,j]   + g_d,
    A[i,j-1]   + g_i,
    A[k,l] + g_t + (i-k-1)g_d + (j-l-1)g_i  if (k,l) is valid
)
```

Boundary rows and columns carry accumulated deletion and insertion costs.
Unlike OSA, the transposition source may be more than two cells away.

## State and memory layout

- `substitution_costs`: row-major `[B, L1, L2]`.
- `transposition_sources`: contiguous `int32 [B, L1, L2, 2]`; `(-1, -1)`
  disables the transposition edge for a cell.
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

Public API: `ohm.damerau`, `ohm.damerau_value`, `ohm.damerau_entropy`, and
`ohm.ops.damerau`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items. CUDA and HIP evaluate the chart in
anti-diagonal order; every accepted predecessor is validated to lie earlier in
that dependency order.

## See also

- [Edit-distance family guide](../../docs/algorithms/edit-distance.md)
- [Levenshtein implementation](../lev/README.md)
- [OSA implementation](../osa/README.md)
- [Source architecture](../ARCHITECTURE.md)
