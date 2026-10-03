# Forward messages implementation

## Recurrence

This interface calls the existing CPU, CUDA or HIP forward launcher for each
of the twelve grid algorithms. It returns the native alpha table and the soft
Bellman value without running a marginal backward pass. It defines no new DP
recurrence, decoder or autograd kernel. Consumers own arbitrary message VJPs.

## State and memory layout

The Python result describes score/cost orientation, state planes, prefix
coordinates, lengths and explicit edit topology. MAS uses its native unpadded
frame/token table; every other algorithm uses a padded prefix grid. Existing
map/value/entropy operations retain their autograd contract.

## Native operations

`grid_forward_messages` returns `(alpha, value)`. It validates dimensions,
lengths, scalar parameters and explicit edit topology before calling a native
launcher. Structural infinity masks use the native unreachable-state sentinel;
the Python metadata retains their original infinities for consumers.

## Files and backends

`registry.cpp` declares the schema. `torch_cpu.cpp`, `torch_cuda.cpp` and
`torch_hip.cpp` own tensor validation and call the respective existing native
forward functions. `torch_common.h` holds shared validation. This interface
supports CPU, CUDA and HIP; it has no Metal implementation.

## See also

[Forward-message API](../../docs/forward-messages.md) and
[native architecture](../ARCHITECTURE.md).
