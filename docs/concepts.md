# Concepts

d²p turns classic dynamic programs into differentiable PyTorch functions.
The functions sum over alignments, parses, or edit paths instead of committing
to one hard structure.

## From hard to soft dynamic programming

A hard recurrence chooses one predecessor with `max` or `min`. A soft
recurrence replaces that choice with a temperature-scaled log-sum-exp:

- score-native families use a soft maximum;
- cost-native families use a soft minimum;
- the gradient of the value is a structured map of expected cell, split, or
  arc occupancy.

This makes a dynamic-program value usable as a loss and makes the map usable
as a structured, attention-like intermediate.

## The three observables

Every algorithm exposes three plain functions:

1. `d2p.<op>(...)` returns the map.
2. `d2p.<op>_value(...)` returns one soft DP value per batch item.
3. `d2p.<op>_entropy(...)` returns one entropy per batch item.

The map has the same shape as the primary score or cost tensor. CKY returns a
merge map shaped like `merge_scores`; `d2p.cky_leaf_map(...)` is a separate
detached leaf derivative view.

The key identity is that the map is the value gradient with respect to the
primary input:

```python
import torch
import d2p

torch.manual_seed(31)
pair_scores = 0.2 * torch.randn(2, 4, 5, requires_grad=True)
value = d2p.sw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
expected_map = d2p.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

(actual_map,) = torch.autograd.grad(value.sum(), pair_scores)
torch.testing.assert_close(actual_map, expected_map)
```

The map is therefore not a visualization heuristic. It is the expected soft
structure induced by the same recurrence that produced the value.

## Temperature

Temperature controls how strongly alternative structures contribute. Lower
values sharpen the distribution; higher values spread mass across more
structures. Temperature is differentiable when supplied as a tensor.

The FP32 kernels intentionally define a conservative supported domain:

```text
temperature is finite and positive
abs(finite score, cost, or scoring parameter) / temperature <= 80
```

Do not anneal below that ratio. Calls outside it raise instead of returning a
plausible but numerically invalid map or entropy.

## Score and cost orientation

Score-native families treat higher values as better:

- Smith-Waterman and Needleman-Wunsch, including affine variants
- LCS and MAS
- CKY and Eisner

Cost-native families treat lower values as better:

- DTW
- Levenshtein
- OSA
- Damerau-Levenshtein

Parameter names encode the convention (`gap_score`, `insertion_cost`, and so
on). There is no runtime orientation flag.

## Choosing a family

- Use Smith-Waterman for local sequence alignment and Needleman-Wunsch for
  end-to-end alignment. Affine variants distinguish opening from extending a
  gap.
- Use DTW for monotone time warping and MAS for frame-to-token monotonic
  alignment.
- Use CKY for binary constituency parsing and Eisner for projective dependency
  parsing.
- Use Levenshtein, OSA, or Damerau for progressively richer edit models. Use
  LCS when match scores, rather than edit costs, are the natural input.

## Differentiation boundary

Map and value functions differentiate through their tensor inputs and
tensor-valued scalar parameters. Entropy differentiates through the primary
score or cost input only. Parameter directions for entropy, plus CKY
`leaf_scores`, raise explicitly because the required second-derivative blocks
are not shipped.

Selected detached parameter VJPs are available under `d2p.raw`; trainable
modules are available under `d2p.nn`.

See [usage.md](usage.md) for exact signatures and policies, and
[algorithms/](algorithms/) for per-family examples.
