# FAQ and troubleshooting

These are common questions and failure modes when using Orihime. See the
[usage guide](usage.md) for the complete API and the
[algorithm guides](algorithms/) for family-specific inputs and outputs.

## What does a top-level function return?

The unsuffixed function returns the structured attention map. Append `_value`
or `_entropy` for the other observables.

```python
import torch
import orihime as ohm

pair_scores = torch.randn(2, 4, 5)
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

## What do value and entropy mean?

`value` is the soft Bellman value for the complete problem instance. It is a
temperature-smoothed maximum over complete score-native structures or minimum
over complete cost-native structures.

`entropy` is the Shannon entropy, in nats, of that distribution over complete
paths, trees, alignments, or edit scripts. It is not the elementwise entropy of
the returned structured attention map. Orihime obtains it from the Bellman
value's temperature derivative: `+dV/dT` for score-native operators and
`-dV/dT` for cost-native operators. See
[Concepts](concepts.md#what-every-operator-returns) for the equations.

## Why are model parameters keyword-only?

Names make score/cost orientation and affine-gap order explicit. The primary
tensor input remains positional; model and structural arguments use keywords.
Defaults are part of the public API for each release.

## How do I learn a gap or temperature?

Pass a scalar tensor directly, or use `ohm.nn`.

```python
gap = torch.tensor(-0.7, requires_grad=True)
temperature = torch.tensor(0.9, requires_grad=True)
train_value = ohm.sw_value(
    pair_scores,
    gap_score=gap,
    temperature=temperature,
)
train_value.sum().backward()

assert gap.grad is not None
assert temperature.grad is not None
```

`ohm.nn.SmithWaterman(learnable=("gap_score",))` provides the stateful layer
equivalent.

## How do I obtain expected gap counts or gap contributions?

A gap score or cost is an input parameter. Differentiating the soft Bellman
value with respect to that parameter gives the expected number of corresponding
gap transitions. These counts are generally fractional at finite temperature.

```python
gap_score = torch.tensor(-0.7, requires_grad=True)
values = ohm.sw_value(
    pair_scores,
    gap_score=gap_score,
    temperature=0.9,
)

per_example_gap_counts = torch.func.jacrev(
    lambda gap: ohm.sw_value(
        pair_scores,
        gap_score=gap,
        temperature=0.9,
    )
)(gap_score)
expected_gap_score_contribution = gap_score * per_example_gap_counts

assert per_example_gap_counts.shape == values.shape
```

For an affine-gap operator, derivatives with respect to `gap_open_score` and
`gap_extend_score` give the expected open and extension counts. The same rule
applies to cost parameters such as insertion and deletion costs. These are
derived statistics rather than separate `forward` fields in 0.1.0.

## How do I obtain a parameter VJP of the map?

Use `ohm.ops`:

```python
grad_map = torch.randn_like(pair_scores).contiguous()
gap_vjp = ohm.ops.sw.backward(
    pair_scores,
    grad_map=grad_map,
    output="gap_score",
    gap_score=-0.7,
    temperature=0.9,
)
assert gap_vjp.numel() == 1
```

The same call can select `pair_scores` for the primary-input derivative.
`grad_map` must be contiguous FP32 with the exact map shape and device.
Broadcasting is not supported.

## Why does an entropy parameter gradient fail?

Entropy supports differentiation through the primary score or cost tensor
only. The required second-derivative blocks for scalar parameters and CKY
`leaf_scores` are not implemented in 0.1.0, so those gradients raise
`NotImplementedError`. Detach those directions or use finite differences.

## What shape does `lengths` use?

Pairwise alignment, DTW, MAS, and edit-distance functions use contiguous
`torch.int32` `[B, 2]` lengths. Eisner uses `[B]`. Lengths live on the input
device. CKY derives its size from its inputs and has no lengths argument.

## Are empty inputs supported?

Orihime 0.1.0 does not accept an empty leading batch (`B=0`). DTW also rejects
one-sided empty lengths because no feasible path exists. Handle empty
minibatches before calling `ohm`.

## Why did MAS reject my shape or lengths?

MAS consumes one frame per step and must cover every token. Inputs are
`[B, T, S]`, and each active row requires `T >= S`.

```python
mas_scores = torch.randn(2, 5, 3)
mas_map = ohm.mas(mas_scores, temperature=1.0)
assert mas_map.shape == mas_scores.shape
```

## How do OSA and Damerau transpositions differ?

OSA takes a boolean `allowed_transpositions` shaped like the substitution costs.
Damerau takes `torch.int32` predecessor coordinates shaped
`[B, L1, L2, 2]`. Use
`ohm.build_damerau_transposition_sources(source_tokens, target_tokens)` to
construct them.

## How do infinite masks work?

Pass a boolean `mask=` (`True` marks cells to exclude) to any map, value, or
entropy function. Orihime inserts `-inf` for score-native operators or `+inf`
for cost-native operators, then converts it to an equivalent finite sentinel.
Writing the corresponding infinity into the input directly has the same
effect; the opposite infinity and NaN are rejected.

## Why was my temperature or score rejected?

Temperature must be finite and strictly positive. Every finite score, cost,
and scoring parameter must satisfy `abs(value) / temperature <= 80`. This is
the supported FP32 numerical domain.

## Which scalar tensor shapes and devices are portable?

Use a 0-D tensor or shape `[1]`, in FP32 on the primary input's device.
Other singleton shapes and per-batch scalar broadcasting are unsupported.
Python numbers are the simplest frozen-parameter representation.

## Does Orihime support AMP?

CUDA autocast runs the native dynamic programs in FP32, returns FP32
observables, and restores gradients to the upstream FP16/BF16 dtype.
Outside autocast, inputs must meet the native FP32 requirements unless
`dtype=torch.float32` is used as the explicit cast.

## Does `torch.compile` work?

Yes. The documented compile support covers map, value, and entropy, including
dynamic scalar tensors and graph-safe validation. See [usage.md](usage.md) for
the supported AOT-eager and Inductor combinations.

## When should I use `ohm.ops`?

Use `ohm.ops` when you need several observables in one name-keyed result, an
explicit map VJP, or a full parameter sensitivity map. Normal model code can
usually use the top-level functions or `ohm.nn`.

## See also

[Concepts](concepts.md) · [Usage guide](usage.md) · [Examples](examples.md)
