# Eisner implementation

Native implementation of the score-native projective dependency parser
available as `ohm.eisner`. Four span states represent complete and incomplete
subtrees with heads on the left or right boundary.

## Recurrence

For arc score `a[h,d]` and temperature `T`:

```text
I_R[i,j] = a[i,j] + LSE_T({C_R[i,k] + C_L[k+1,j] : i <= k < j})
I_L[i,j] = a[j,i] + LSE_T({C_R[i,k] + C_L[k+1,j] : i <= k < j})
C_R[i,j] = LSE_T({C_R[i,k] + I_R[k,j] : i <= k < j})
C_L[i,j] = LSE_T({I_L[i,k] + C_L[k,j] : i < k <= j})
S        = C_R[0,N-1]
```

`C_R` and `C_L` are complete spans; `I_R` and `I_L` add the directed boundary
arc. Position zero is the artificial root.

## State and memory layout

- `arc_scores`: row-major `[B, N, N]`, indexed `[head, dependent]`.
- `C_R`, `C_L`, `I_R`, and `I_L`: separate `[B, N, N]` charts.
- outside and derivative workspaces mirror those four charts.
- value and temperature gradient: `[B]`.
- arc map, map cotangent, and temperature sensitivity: `[B, N, N]`.

Lengths are contiguous `int32 [B]` values that include the root position.
Self-arcs and arcs into the root do not participate.

## Native operations

| Operation | Result |
| --- | --- |
| forward | four inside charts and projective-tree value |
| backward | arc map and value gradient for temperature |
| HVP | map directional derivative with respect to arc scores |
| parameter sensitivity | full arc-map derivative for temperature |

Public API: `ohm.eisner`, `ohm.eisner_value`, `ohm.eisner_entropy`, and
`ohm.ops.eisner`.

## Files and backends

| Files | Role |
| --- | --- |
| `kernels_cpu.cpp`, `kernels_cpu.h` | CPU span recurrence and derivative kernels |
| `kernels_gpu.cu`, `kernels_gpu.cuh` | NVIDIA CUDA implementation |
| `kernels_gpu.hip`, `kernels_gpu.hiph` | AMD HIP implementation |
| `torch_cpu.cpp`, `torch_cuda.cpp`, `torch_hip.cpp` | PyTorch validation, allocation, and dispatch |
| `registry.cpp` | internal dispatcher schemas |

All backends advance by increasing span width, computing incomplete states
before complete states at each width. The backward and sensitivity passes
reverse that dependency order. CPU work is parallelized over batch items;
CUDA and HIP parallelize independent spans on the current PyTorch stream.

## See also

- [Eisner guide](../../docs/algorithms/eisner.md)
- [Source architecture](../ARCHITECTURE.md)
