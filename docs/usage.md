# Usage guide

Orihime has 14 public algorithms. Each has top-level functions that return a
structured attention map, value, or entropy, plus matching modules and explicit
derivative operations. The same rules for trainable parameters,
variable-length batches, mixed precision, and `torch.compile` apply throughout.
See the [algorithm guides](algorithms/) for individual recurrences and
[examples](examples.md) for runnable tasks.

## Map, value, and entropy functions

| Family | Map function | Keyword parameters | Structural keywords |
| --- | --- | --- | --- |
| local alignment | `ohm.sw` | `gap_score`, `temperature` | `lengths` |
| affine local alignment | `ohm.sw_affine` | `gap_open_score`, `gap_extend_score`, `temperature` | `lengths` |
| Saigo–Vert local alignment | `ohm.sv` | `gap_score`, `temperature` | `lengths` |
| affine Saigo–Vert local alignment | `ohm.sv_affine` | `gap_open_score`, `gap_extend_score`, `temperature` | `lengths` |
| global alignment | `ohm.nw` | `gap_score`, `temperature` | `lengths` |
| affine global alignment | `ohm.nw_affine` | `gap_open_score`, `gap_extend_score`, `temperature` | `lengths` |
| DTW | `ohm.dtw` | `temperature` | `lengths`, `bandwidth` |
| LCS | `ohm.lcs` | `temperature` | `lengths` |
| Levenshtein | `ohm.lev` | `insertion_cost`, `deletion_cost`, `temperature` | `lengths` |
| OSA | `ohm.osa` | `insertion_cost`, `deletion_cost`, `transposition_cost`, `temperature` | `lengths`, `allowed_transpositions` |
| Damerau | `ohm.damerau` | `insertion_cost`, `deletion_cost`, `transposition_cost`, `temperature` | `lengths`, `transposition_sources` |
| MAS | `ohm.mas` | `temperature` | `lengths` |
| CKY | `ohm.cky` | `temperature` | none |
| Eisner | `ohm.eisner` | `temperature` | `lengths` |

Append `_value` or `_entropy` to any map-function name for the other two
observables. The value is the soft Bellman value for the complete instance.
Entropy is the Shannon entropy, in nats, of the recurrence-defined distribution
over complete structures, not the elementwise entropy of the map. Orihime
computes it as `+dV/dT` for score-native operators and `-dV/dT` for cost-native
operators. See [Concepts](concepts.md#what-every-operator-returns) for the full
definition.

Every map, value, and entropy function also accepts `mask`, a boolean tensor
shaped like the primary input. `True` excludes a cell. Orihime uses `-inf` for
score-native operators and `+inf` for cost-native operators, including for
CKY's `merge_scores`.

```python
import torch
import orihime as ohm

torch.manual_seed(11)
pair_scores = torch.randn(2, 4, 5, requires_grad=True)

alignment = ohm.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = ohm.sw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = ohm.sw_entropy(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)
```

Score-native families maximize scores; cost-native DTW, Levenshtein, OSA,
and Damerau minimize costs.

## Saigo–Vert alignment

The linear- and affine-gap Saigo–Vert operators use the same top-level API as
the other algorithms:

```python
sv_scores = torch.randn(2, 4, 5, requires_grad=True)

linear_map = ohm.sv(
    sv_scores,
    gap_score=-0.7,
    temperature=1.0,
)
affine_value = ohm.sv_affine_value(
    sv_scores,
    gap_open_score=-0.7,
    gap_extend_score=-0.2,
    temperature=1.0,
)

assert linear_map.shape == sv_scores.shape
assert affine_value.shape == (2,)
```

The matching modules are `ohm.nn.SaigoVertLinear` and
`ohm.nn.SaigoVertAffine`. Use `ohm.ops.sv` and `ohm.ops.sv_affine` for explicit
forward, backward, and sensitivity operations. See the
[Saigo–Vert guide](algorithms/sv.md) for the recurrences.

## Differentiation

Tensor-valued scalar parameters may be 0-D tensors or one-element vectors.
Python numbers are frozen. Backpropagating through a value produces the map
on the primary input and expected-count gradients on learnable scalar
parameters.

```python
gap = torch.tensor(-0.7, requires_grad=True)
temperature = torch.tensor(0.9, requires_grad=True)

train_value = ohm.sw_value(
    pair_scores,
    gap_score=gap,
    temperature=temperature,
)
train_value.sum().backward()

assert pair_scores.grad is not None
assert gap.grad is not None
assert temperature.grad is not None
```

Backpropagating through the map uses orihime's hand-written second-order kernels.
Map and value support reverse mode and standalone `torch.func.jvp`, `vjp`,
and `jacrev` over their differentiable inputs and tensor parameters.

Entropy differentiates only through its primary score or cost input.
Scalar-parameter derivatives and CKY `leaf_scores` derivatives raise
`NotImplementedError` instead of returning a silent zero. `jacfwd` and nested
forward-mode compositions are not supported in 0.1.0.

## Explicit operations

Every algorithm has an object under `ohm.ops` with three methods:

- `forward` selects `map`, `value`, and `entropy`.
- `backward` contracts `grad_map` against the map derivative and selects
  gradients for tensor inputs and scalar scoring parameters.
- `sensitivity` selects the full map derivative for each scalar scoring
  parameter.

All three use `output=` consistently. A single field name returns a tensor. A
sequence of names returns exactly those fields in a dictionary and preserves
their order. `forward(output=None)` returns the standard `map`, `value`, and
`entropy` fields; it does not include every algorithm-specific statistic that
can be derived from the value. For `backward` and `sensitivity`, `output=None`
returns all derivative fields supported by that method.

```python
pair_scores = pair_scores.detach()
grad_map = torch.randn_like(pair_scores).contiguous()

observables = ohm.ops.sw.forward(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
map_only = ohm.ops.sw.forward(
    pair_scores,
    output="map",
    gap_score=-0.7,
    temperature=0.9,
)
selected_grads = ohm.ops.sw.backward(
    pair_scores,
    grad_map=grad_map,
    output=("pair_scores", "temperature"),
    gap_score=-0.7,
    temperature=0.9,
)
parameter_maps = ohm.ops.sw.sensitivity(
    pair_scores,
    output=("gap_score", "temperature"),
    gap_score=-0.7,
    temperature=0.9,
)

assert tuple(observables) == ("map", "value", "entropy")
assert map_only.shape == pair_scores.shape
assert tuple(selected_grads) == ("pair_scores", "temperature")
assert parameter_maps["temperature"].shape == pair_scores.shape
```

`grad_map` must be a contiguous FP32 tensor with the same shape and device as
the map. A non-contiguous, wrong-shaped, wrong-dtype, wrong-device, or
non-tensor cotangent is rejected before native dispatch.
Tensor-parameter gradients retain the parameter's scalar shape; gradients for
Python-number parameters are zero-dimensional tensors.

## Modules

Modules under `ohm.nn` store selected model parameters as `nn.Parameter`
objects and the rest as buffers. Calling a module returns the map;
`.value(...)` and `.entropy(...)` return the other observables.

```python
layer = ohm.nn.SmithWaterman(
    gap_score=-0.7,
    temperature=0.9,
    learnable=("gap_score",),
)
layer_map = layer(pair_scores)
layer_value = layer.value(pair_scores)

assert layer_map.shape == pair_scores.shape
assert layer_value.shape == (2,)
assert set(dict(layer.named_parameters())) == {"gap_score"}
assert "temperature" in dict(layer.named_buffers())
```

## Variable-length batching and shapes

The native kernels accept exactly one leading batch dimension:

- pairwise alignment, DTW, MAS, and edit-distance inputs: `[B, L1, L2]`
- Eisner arc scores: `[B, N, N]`
- CKY merge and leaf scores: `[B, N, N, N]` and `[B, N]`

Pairwise `lengths` has shape `[B, 2]`; Eisner `lengths` has shape `[B]`.
Lengths are contiguous `torch.int32` tensors on the input device. Map entries
outside the declared active lengths are zero. DTW rejects one-sided empty
instances. MAS requires the first active length to be at least the second.

```python
lengths = torch.tensor(
    [[4, 5], [3, 4]],
    dtype=torch.int32,
)
ragged_map = ohm.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
    lengths=lengths,
)
assert torch.count_nonzero(ragged_map[1, 3:]) == 0
assert torch.count_nonzero(ragged_map[1, :, 4:]) == 0
```

Orihime 0.1.0 does not accept an empty leading batch (`B=0`) on CPU or CUDA.
Handle empty minibatches before calling `ohm`.

OSA uses a boolean `allowed_transpositions` shaped like its substitution costs.
Damerau uses `torch.int32` predecessor coordinates shaped
`[B, L1, L2, 2]`; build them with
`ohm.build_damerau_transposition_sources(...)`.

## Scalar dtype and device policy

For portable eager, transform, compile, CPU, and CUDA behavior:

- pass a Python number, a 0-D tensor, or a one-element vector;
- tensor-valued scalar parameters must be floating point;
- keep tensor-valued scalars on the same device and in FP32 with the primary
  input;
- use `ohm.nn` constructor `device=` and `dtype=` when a module should manage
  those scalars.

Other singleton shapes and per-batch parameter broadcasting are rejected.

## Numerical domain and masks

Temperature must be finite and strictly positive. Every finite score, cost,
and scoring parameter must satisfy:

```text
abs(value) / temperature <= 80
```

The bound keeps the FP32 recurrences away from exponential overflow and
low-temperature cancellation. Calls outside it raise in eager and compiled
execution.

To exclude cells, pass a boolean `mask=` to any map, value, or entropy function.
`True` marks excluded cells. Orihime inserts `-inf` for score-native operators
or `+inf` for cost-native operators, then converts it to an equivalent finite
sentinel. Writing the corresponding infinity into the input directly has the
same effect; the opposite infinity and every NaN are rejected.

CKY has no `lengths` argument; mask padded chart entries in its scores.

## Mixed precision (AMP)

Native dynamic programs accumulate in FP32. Passing `dtype=torch.float32`
explicitly requests FP32 accumulation; every other explicit dtype is rejected.
CUDA autocast promotes FP16/BF16 inputs to FP32 internally, returns FP32
observables, and restores input gradients to their original dtype.

```python
explicit_fp32 = ohm.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
    dtype=torch.float32,
)
assert explicit_fp32.dtype == torch.float32
```

## `torch.compile`

Map, value, and entropy support fullgraph AOT-eager and the documented
Inductor matrix. Dynamic tensor parameters and valid dynamic lengths remain
runtime values; invalid values raise graph-safe runtime errors.

```python
def compiled_alignment(scores):
    return ohm.sw(
        scores,
        gap_score=-0.7,
        temperature=0.9,
    )

compiled = torch.compile(
    compiled_alignment,
    backend="aot_eager",
    fullgraph=True,
)
compiled_map = compiled(pair_scores)
torch.testing.assert_close(compiled_map, alignment)
```

## Implementation boundary

The compiled dispatcher schemas under `torch.ops.orihime.*` and Orihime's
Python kernel adapters are private implementation details. They use native
schema order and may change without a public API migration. Use the top-level
functions, `ohm.nn`, or `ohm.ops`.

## See also

[Concepts](concepts.md) · [Algorithm guides](algorithms/) ·
[Examples](examples.md) · [FAQ](faq.md)
