# Levenshtein implementation

Native implementation of the cost-native soft edit-distance operator available
as `ohm.lev`. Paths substitute one position, delete from the first input, or
insert into it.

## Recurrence

For substitution cost `c[i,j]`, insertion cost `g_i`, deletion cost `g_d`, and
temperature `T`:

```text
A[i,j] = softmin_T(
    A[i-1,j-1] + c[i,j],
    A[i-1,j]   + g_d,
    A[i,j-1]   + g_i
)
A[i,0] = i * g_d
A[0,j] = j * g_i
S      = A[L1,L2]
```

The returned map is the posterior mass of the substitution transition;
insertions and deletions are represented by scalar-parameter derivatives.

## State and memory layout

- `substitution_costs`: row-major `[B, L1, L2]`.
- `alpha` and `beta`: flattened row-major
  `[B, (L1 + 1) * (L2 + 1)]` tables.
- value and scalar-parameter gradients: `[B]`.
- substitution map, map cotangents, and map sensitivities: `[B, L1, L2]`.

Lengths select an active prefix for each input. Padded map cells are zero.

## Native operations

| Operation | Result |
| --- | --- |
| forward | alpha table and terminal soft distance |
| backward | substitution map plus value gradients for insertion, deletion, and temperature |
| HVP | map directional derivative with respect to substitution costs |
| parameter sensitivity | full map derivative for insertion, deletion, or temperature |

Public API: `ohm.lev`, `ohm.lev_value`, `ohm.lev_entropy`, and `ohm.ops.lev`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `torch_bindings.h` | shared binding declarations |
| `registry.cpp` | internal dispatcher schemas |

CPU work is parallelized over batch items and uses compensated soft-min
reductions. CUDA and HIP evaluate the table in anti-diagonal dependency order
on the current PyTorch stream.

## See also

- [Edit-distance family guide](../../docs/algorithms/edit-distance.md)
- [OSA implementation](../osa/README.md)
- [Damerau–Levenshtein implementation](../damerau/README.md)
- [Source architecture](../ARCHITECTURE.md)
