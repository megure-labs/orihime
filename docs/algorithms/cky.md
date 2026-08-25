# CKY

CKY is a score-native soft binary constituency parser. It is the only d2p
family with two differentiable tensor inputs:

- `merge_scores`: `[B, N, N, N]`
- `leaf_scores`: `[B, N]`

The primary map matches `merge_scores`.

```python
import torch
import d2p

torch.manual_seed(53)
merge_scores = 0.2 * torch.randn(2, 3, 3, 3, requires_grad=True)
leaf_scores = 0.2 * torch.randn(2, 3, requires_grad=True)

merge_map = d2p.cky(
    merge_scores,
    leaf_scores,
    temperature=0.9,
)
value = d2p.cky_value(
    merge_scores,
    leaf_scores,
    temperature=0.9,
)
entropy = d2p.cky_entropy(
    merge_scores,
    leaf_scores,
    temperature=0.9,
)
leaf_map = d2p.cky_leaf_map(
    merge_scores,
    leaf_scores,
    temperature=0.9,
)

assert merge_map.shape == merge_scores.shape
assert value.shape == entropy.shape == (2,)
assert leaf_map.shape == leaf_scores.shape
assert not leaf_map.requires_grad
```

`cky_leaf_map` is a detached first-derivative view of the value, not a second
differentiable map.

CKY has no `lengths` argument. Pass `mask=` (a boolean tensor shaped like
`merge_scores`, `True` = exclude) to exclude padded chart cells; writing `-inf`
into `merge_scores` is the equivalent low-level form. (`mask=` covers
`merge_scores`; mask `leaf_scores` padding with `-inf` directly.)

Map and value differentiate through both tensor inputs and a tensor-valued
temperature. Entropy differentiates through `merge_scores` only. Entropy
directions for `leaf_scores` and temperature raise explicitly.

`d2p.raw.cky.vjp_fields` contains `leaf_scores` and `temperature`. The stateful
layer is `d2p.nn.CKY`.

Temperature must be finite and positive. Every finite chart score must satisfy
`abs(value) / temperature <= 80`.
