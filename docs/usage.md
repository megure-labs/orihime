# Usage guide

d2p's v3 surface consists of plain, keyword-oriented functions. Each of the
12 algorithms has a map function, a `_value` function, and an `_entropy`
function. The primary tensor input remains positional; all scalar parameters
and structural arguments are keyword-only.

## Functions and parameters

| Family | Map function | Keyword parameters | Structural keywords |
| --- | --- | --- | --- |
| local alignment | `d2p.sw` | `gap_score`, `temperature` | `lengths` |
| affine local alignment | `d2p.sw_affine` | `gap_open_score`, `gap_extend_score`, `temperature` | `lengths` |
| global alignment | `d2p.nw` | `gap_score`, `temperature` | `lengths` |
| affine global alignment | `d2p.nw_affine` | `gap_open_score`, `gap_extend_score`, `temperature` | `lengths` |
| DTW | `d2p.dtw` | `temperature` | `lengths`, `bandwidth` |
| LCS | `d2p.lcs` | `temperature` | `lengths` |
| Levenshtein | `d2p.lev` | `insertion_cost`, `deletion_cost`, `temperature` | `lengths` |
| OSA | `d2p.osa` | `insertion_cost`, `deletion_cost`, `transposition_cost`, `temperature` | `lengths`, `allowed_transpositions` |
| Damerau | `d2p.damerau` | `insertion_cost`, `deletion_cost`, `transposition_cost`, `temperature` | `lengths`, `transposition_sources` |
| MAS | `d2p.mas` | `temperature` | `lengths` |
| CKY | `d2p.cky` | `temperature` | none |
| Eisner | `d2p.eisner` | `temperature` | `lengths` |

Append `_value` or `_entropy` to any map-function name for the other two
observables. Every map/value/entropy function also accepts `mask` (a boolean
tensor shaped like the primary input; `True` marks cells to exclude, applied as
the orientation-correct infinity internally — including CKY's `merge_scores`).

```python
import torch
import d2p

torch.manual_seed(11)
pair_scores = torch.randn(2, 4, 5, requires_grad=True)

alignment = d2p.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = d2p.sw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = d2p.sw_entropy(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)
```

Score-native families maximize scores; cost-native DTW, Levenshtein, OSA,
and Damerau minimize costs.

## Autograd

Tensor-valued scalar parameters may be 0-D tensors or one-element vectors.
Python numbers are frozen. Backpropagating through a value produces the map
on the primary input and expected-count gradients on learnable scalar
parameters.

```python
gap = torch.tensor(-0.7, requires_grad=True)
temperature = torch.tensor(0.9, requires_grad=True)

train_value = d2p.sw_value(
    pair_scores,
    gap_score=gap,
    temperature=temperature,
)
train_value.sum().backward()

assert pair_scores.grad is not None
assert gap.grad is not None
assert temperature.grad is not None
```

Backpropagating through the map uses d2p's hand-written second-order kernels.
Map and value support reverse mode and standalone `torch.func.jvp`, `vjp`,
and `jacrev` over their differentiable inputs and tensor parameters.

Entropy is intentionally narrower: its primary score or cost input is the
only differentiable direction. Scalar-parameter derivatives, and CKY
`leaf_scores` derivatives, raise `NotImplementedError` instead of returning a
silent zero. `jacfwd` and nested forward-mode compositions are not part of
the 0.1.0 contract.

## Selected no-graph VJPs

`d2p.raw.<op>` exposes detached, type-stable VJPs for the fields listed by
that module's `vjp_fields`. `vjp_one` always returns a tensor; `vjp` always
returns a dictionary. `wrt` is mandatory.

```python
pair_scores = pair_scores.detach()
map_cotangent = torch.randn_like(pair_scores)

gap_vjp = d2p.raw.sw.vjp_one(
    pair_scores,
    wrt="gap_score",
    cotangent=map_cotangent,
    gap_score=-0.7,
    temperature=0.9,
)
selected_vjps = d2p.raw.sw.vjp(
    pair_scores,
    wrt=("gap_score", "temperature"),
    cotangent=map_cotangent,
    gap_score=-0.7,
    temperature=0.9,
)

assert gap_vjp.numel() == 1
assert set(selected_vjps) == {"gap_score", "temperature"}
```

The supported raw-cotangent contract is exact: the cotangent must be a
contiguous FP32 tensor with the same shape and device as the map. Broadcasting
a scalar or singleton tensor is outside the public contract, even if a
particular native primitive happens to accept it.

The named low-level derivative bindings exposed by the same module use the
same strict vector contract. `marginals_hvp` calls its derivative vector a
`tangent`; `marginals_backward` calls it a `cotangent`. Across all twelve
operators, each vector must be a contiguous FP32 tensor with the exact primary
map shape and device. CKY validates both the merge and leaf tangents. A
non-contiguous, wrong-shaped, wrong-dtype, wrong-device, or non-tensor vector
is rejected before native dispatch; user vectors are never implicitly copied.

Use ordinary autograd when the primary-input VJP is required.

## Stateful modules

The `d2p.nn` tier stores selected model parameters as `nn.Parameter` objects
and the rest as buffers. Each layer's call returns the map; `.value(...)` and
`.entropy(...)` return the other observables.

```python
layer = d2p.nn.SmithWaterman(
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

## Shapes, batching, and structural inputs

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
ragged_map = d2p.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
    lengths=lengths,
)
assert torch.count_nonzero(ragged_map[1, 3:]) == 0
assert torch.count_nonzero(ragged_map[1, :, 4:]) == 0
```

An empty leading batch (`B=0`) is outside the 0.1.0 public contract on both
CPU and CUDA. Split upstream code around empty minibatches instead of passing
them to d2p.

OSA uses a boolean `allowed_transpositions` shaped like its substitution costs.
Damerau uses `torch.int32` predecessor coordinates shaped
`[B, L1, L2, 2]`; build them with
`d2p.build_damerau_transposition_sources(...)`.

## Scalar dtype and device policy

For portable eager, transform, compile, CPU, and CUDA behavior:

- pass a Python number, a 0-D tensor, or a one-element vector;
- tensor-valued scalar parameters must be floating point;
- keep tensor-valued scalars on the same device and in FP32 with the primary
  input;
- use `d2p.nn` constructor `device=` and `dtype=` when a module should manage
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

To exclude cells, pass a boolean `mask=` (`True` marks excluded cells) to any
map/value/entropy function; d2p applies the orientation-correct infinity
internally, so you never handle infinities and entropy stays finite. The
equivalent low-level form is writing `-inf` (score-native) or `+inf`
(cost-native) into the inputs directly; the opposite infinity and every NaN
are rejected.

CKY has no `lengths` argument; mask padded chart entries in its scores.

## FP32 and AMP

Native dynamic programs accumulate in FP32. `dtype=torch.float32` is an
explicit accumulation escape hatch; every other explicit dtype is rejected.
CUDA autocast promotes FP16/BF16 inputs to FP32 internally, returns FP32
observables, and restores input gradients to their original dtype.

```python
explicit_fp32 = d2p.sw(
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
    return d2p.sw(
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

## Kernel-level bindings

`d2p.raw.<op>` exposes named wrappers around the compiled primitives (the same
module that provides the VJPs), and `torch.ops.d2p.*` is the dispatcher floor.
The named derivative-vector arguments follow the strict tangent/cotangent
contract above. These interfaces use kernel vocabulary and positional C++
schema order. Prefer
the high-level functions or `d2p.nn` unless direct primitive control is required.
