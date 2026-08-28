# CKY implementation

Native implementation of the score-native binary constituency parser available
as `ohm.cky`. The inside chart sums over every split of every span; the backward
chart recovers merge and leaf derivatives.

## Recurrence

For merge score `m[i,k,j]`, leaf score `l[i]`, and temperature `T`:

```text
Z[i,i] = l[i]
Z[i,j] = LSE_T({Z[i,k] + Z[k+1,j] + m[i,k,j] : i <= k < j})
S      = Z[0,N-1]
```

The public merge map is the posterior weight of each `(i,k,j)` split. Summing
over `k` gives the posterior mass of span `(i,j)`.

## State and memory layout

- `merge_scores`: row-major `[B, N, N, N]`, indexed `(i,k,j)`.
- `leaf_scores`: row-major `[B, N]`.
- inside chart `Z` and outside chart `beta`: `[B, N, N]`.
- conditional and joint split tables: `[B, N, N, N]`.
- merge map: `[B, N, N, N]`; leaf derivative view: `[B, N]`.

Only entries with `i <= k < j` participate. CKY has no `lengths` argument;
padded charts must be excluded through their scores and mask.

## Native operations

| Operation | Result |
| --- | --- |
| forward | inside chart and full-span value |
| backward | merge map, leaf derivative, and value gradient for temperature |
| HVP | merge-map directional derivative for merge and leaf tangents |
| parameter sensitivity | full merge-map derivative for temperature |

Public API: `ohm.cky`, `ohm.cky_value`, `ohm.cky_entropy`,
`ohm.cky_leaf_map`, and `ohm.ops.cky`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU inside, outside, and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `torch_api.h` | shared binding declarations |
| `registry.cpp` | internal dispatcher schemas |

All backends build the chart by increasing span width and traverse it in the
reverse order for outside messages. CPU work is parallelized over batch items;
CUDA and HIP parallelize independent spans and split calculations on the
current PyTorch stream.

## See also

- [CKY guide](../../docs/algorithms/cky.md)
- [Source architecture](../ARCHITECTURE.md)
