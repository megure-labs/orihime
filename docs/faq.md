# FAQ

## What does a top-level function return?

The unsuffixed function returns the structured map. Append `_value` or
`_entropy` for the other observables.

```python
import torch
import d2p

pair_scores = torch.randn(2, 4, 5)
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

## Why are model parameters keyword-only?

Names make score/cost orientation and affine-gap order explicit. The primary
tensor input remains positional; model and structural arguments use keywords.
Defaults are part of the versioned v3 contract.

## How do I learn a gap or temperature?

Pass a scalar tensor directly, or use `d2p.nn`.

```python
gap = torch.tensor(-0.7, requires_grad=True)
temperature = torch.tensor(0.9, requires_grad=True)
train_value = d2p.sw_value(
    pair_scores,
    gap_score=gap,
    temperature=temperature,
)
train_value.sum().backward()

assert gap.grad is not None
assert temperature.grad is not None
```

`d2p.nn.SmithWaterman(learnable=("gap_score",))` provides the stateful layer
equivalent.

## How do I obtain a parameter VJP of the map?

Use the selected, detached raw tier:

```python
cotangent = torch.randn_like(pair_scores)
gap_vjp = d2p.raw.sw.vjp_one(
    pair_scores,
    wrt="gap_score",
    cotangent=cotangent,
    gap_score=-0.7,
    temperature=0.9,
)
assert gap_vjp.numel() == 1
```

Use ordinary autograd for primary-input derivatives. Raw cotangents must be
contiguous FP32 tensors with the exact map shape and device; broadcasting is
not a public contract.

## Why did entropy reject a parameter gradient?

Entropy supports differentiation through the primary score or cost tensor
only. Scalar-parameter directions and CKY `leaf_scores` require unshipped
second-derivative blocks and raise `NotImplementedError`. Detach those
directions or use finite differences.

## What shape does `lengths` use?

Pairwise alignment, DTW, MAS, and edit-distance functions use contiguous
`torch.int32` `[B, 2]` lengths. Eisner uses `[B]`. Lengths live on the input
device. CKY derives its size from its inputs and has no lengths argument.

## Are empty inputs supported?

An empty leading batch (`B=0`) is outside the 0.1.0 contract. DTW also rejects
one-sided empty lengths because no feasible path exists. Handle empty
minibatches before calling d2p.

## Why did MAS reject my shape or lengths?

MAS consumes one frame per step and must cover every token. Inputs are
`[B, T, S]`, and each active row requires `T >= S`.

```python
mas_scores = torch.randn(2, 5, 3)
mas_map = d2p.mas(mas_scores, temperature=1.0)
assert mas_map.shape == mas_scores.shape
```

## How do OSA and Damerau transpositions differ?

OSA takes a boolean `allowed_transpositions` shaped like the substitution costs.
Damerau takes `torch.int32` predecessor coordinates shaped
`[B, L1, L2, 2]`. Use
`d2p.build_damerau_transposition_sources(source_tokens, target_tokens)` to
construct them.

## How do infinite masks work?

Pass a boolean `mask=` (`True` marks cells to exclude) to any map/value/entropy
function; d2p applies the orientation-correct infinity internally and normalizes
it to an answer-preserving finite sentinel. Writing `-inf` (score-native) or
`+inf` (cost-native) directly is the equivalent low-level form; the opposite
infinity and NaN are rejected.

## Why was my temperature or score rejected?

Temperature must be finite and strictly positive. Every finite score, cost,
and scoring parameter must satisfy `abs(value) / temperature <= 80`. This is
the supported FP32 numerical domain.

## Which scalar tensor shapes and devices are portable?

Use a 0-D tensor or shape `[1]`, in FP32 on the primary input's device.
Other singleton shapes and per-batch scalar broadcasting are unsupported.
Python numbers are the simplest frozen-parameter representation.

## Does d2p support AMP?

CUDA autocast runs the native dynamic programs in FP32, returns FP32
observables, and restores gradients to the upstream FP16/BF16 dtype.
Outside autocast, inputs must meet the native FP32 contract unless
`dtype=torch.float32` is used as the explicit cast.

## Does `torch.compile` work?

Yes. The v3 compile matrix covers map, value, and entropy, including dynamic
scalar tensors and graph-safe validation. See [usage.md](usage.md) for the
exact AOT-eager and Inductor boundaries.

## When should I use the raw kernel bindings?

`d2p.raw.<op>` exposes the named kernel bindings (`forward`, `forward_t`,
`value_grad_params`, `marginals_backward`, `marginals_hvp`, `marginals_grad_*`)
next to its VJPs. Use them only for direct named-kernel control; normal model
code should use the top-level functions or `d2p.nn`.
