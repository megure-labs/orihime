# CKY

`ohm.cky(...)` implements a differentiable inside algorithm for binary
constituency parsing. It replaces the hard choice of one parse tree with a
temperature-controlled log-sum-exp over every valid tree.

## What it computes

Over a sequence of length `N`, CKY builds each span `(i, j)` by choosing a split
`k`, joining `(i, k)` to `(k + 1, j)`. Orihime returns the posterior weight of
every such merge. The top-level functions return:

- `ohm.cky(...)` → merge map `[B, N, N, N]`
- `ohm.cky_value(...)` → soft Bellman value `[B]`
- `ohm.cky_entropy(...)` → entropy `[B]`
- `ohm.cky_leaf_map(...)` → detached leaf derivative `[B, N]`

The merge map is the gradient of the value with respect to `merge_scores`.
Each entry is therefore the posterior weight of that merge.

## Soft inside recurrence

With temperature `T`, the recurrence is:

```text
inside[i, i] = leaf_scores[i]
inside[i, j] = T logsumexp_k(
    (merge_scores[i, k, j] + inside[i, k] + inside[k + 1, j]) / T
)
value = inside[0, N - 1]
```

As `T` approaches zero, the distribution sharpens toward the single best
tree. Larger temperatures spread mass across competing splits.

## Inputs

- `merge_scores`: floating-point `[B, N, N, N]`; only `i <= k < j` is read.
- `leaf_scores`: floating-point `[B, N]`.
- `temperature`: positive Python number or floating scalar tensor.
- `mask`: optional boolean tensor shaped like `merge_scores`; `True` excludes
  that merge.

CKY has no `lengths` argument. Mask padded merge cells, and write `-inf` into
padded `leaf_scores` if needed.

## Example

```python
import torch
import orihime as ohm

torch.manual_seed(53)
merge_scores = (0.2 * torch.randn(2, 5, 5, 5)).requires_grad_()
leaf_scores = (0.2 * torch.randn(2, 5)).requires_grad_()

merge_map = ohm.cky(merge_scores, leaf_scores, temperature=0.9)
value = ohm.cky_value(merge_scores, leaf_scores, temperature=0.9)
entropy = ohm.cky_entropy(merge_scores, leaf_scores, temperature=0.9)
leaf_map = ohm.cky_leaf_map(merge_scores, leaf_scores, temperature=0.9)
span_map = merge_map.sum(dim=2)

assert merge_map.shape == merge_scores.shape
assert span_map.shape == (2, 5, 5)
assert value.shape == entropy.shape == (2,)
assert leaf_map.shape == leaf_scores.shape

(value.sum()).backward()
assert merge_scores.grad is not None
assert leaf_scores.grad is not None
```

Read `span_map[b, i, j]` as the posterior mass assigned to span `(i, j)`,
summing over all ways to split it. The full-sentence span has mass one.

## Derivatives and modules

Pass temperature as a tensor to optimize it, or store it in `ohm.nn.CKY`:

```python
layer = ohm.nn.CKY(temperature=0.9, learnable=("temperature",))
layer_map = layer(merge_scores.detach(), leaf_scores.detach())
layer_value = layer.value(merge_scores.detach(), leaf_scores.detach())

assert layer_map.shape == merge_scores.shape
assert layer_value.shape == (2,)
```

Map and value differentiate through both score tensors and tensor-valued
temperature. Entropy differentiates only through `merge_scores`; gradients
with respect to `leaf_scores` or temperature raise `NotImplementedError`.
`ohm.ops.cky` provides explicit map VJPs for both score tensors and a
temperature-sensitivity map.

## Common pitfalls

- The merge map is `[B, N, N, N]`, not `[B, N, N]`; sum over `dim=2` for span
  posteriors.
- `k` is the last index of the left child, so the valid split condition is
  `i <= k < j`.
- Every tree uses every leaf exactly once. Leaf scores move the value but do
  not change the distribution over tree structure.

## Complexity

CKY takes `O(B N^3)` time and `O(B N^2)` working memory. The materialized
`merge_scores` and merge map are themselves `O(B N^3)`.

## See also

[Usage guide](../usage.md) · [Examples](../examples.md) ·
[Performance](../performance.md)
