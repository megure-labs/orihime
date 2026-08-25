# Edit-distance families

The edit-distance group contains one score-native family (LCS) and three
cost-native families (Levenshtein, OSA, and Damerau-Levenshtein). Every input
and map has shape `[B, L1, L2]`.

## Levenshtein

```python
import torch
import orihime as ori

torch.manual_seed(67)
substitution_costs = torch.rand(2, 4, 5, requires_grad=True)

lev_map = ori.lev(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.9,
)
lev_value = ori.lev_value(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.9,
)
lev_entropy = ori.lev_entropy(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.9,
)

assert lev_map.shape == substitution_costs.shape
assert lev_value.shape == lev_entropy.shape == (2,)
```

Insertion and deletion costs default to one.

## LCS

LCS is score-native and has only a temperature parameter.

```python
match_scores = 0.2 * torch.randn(2, 4, 5)
lcs_map = ori.lcs(match_scores, temperature=0.9)
lcs_value = ori.lcs_value(match_scores, temperature=0.9)
assert lcs_map.shape == match_scores.shape
assert lcs_value.shape == (2,)
```

## Optimal String Alignment

OSA adds restricted adjacent transpositions. A boolean
`allowed_transpositions` enables valid edges; `None` disables them. The name
makes its polarity explicit: `True` allows an adjacent-transposition edge,
while the separate cell-exclusion `mask=True` excludes a DP cell.

```python
allowed_transpositions = torch.zeros_like(
    substitution_costs,
    dtype=torch.bool,
)
allowed_transpositions[:, 1:, 1:] = True

osa_map = ori.osa(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    transposition_cost=1.0,
    temperature=0.9,
    allowed_transpositions=allowed_transpositions,
)
osa_value = ori.osa_value(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    transposition_cost=1.0,
    temperature=0.9,
    allowed_transpositions=allowed_transpositions,
)
assert osa_map.shape == substitution_costs.shape
assert osa_value.shape == (2,)
```

## Damerau-Levenshtein

Damerau uses predecessor coordinates for unrestricted transpositions. Build
them from token IDs.

```python
source_tokens = torch.tensor(
    [[1, 2, 3, 4], [4, 3, 2, 1]],
    dtype=torch.int64,
)
target_tokens = torch.tensor(
    [[2, 1, 3, 4, 5], [4, 2, 3, 1, 0]],
    dtype=torch.int64,
)
transposition_sources = ori.build_damerau_transposition_sources(
    source_tokens,
    target_tokens,
)
damerau_map = ori.damerau(
    substitution_costs,
    temperature=0.9,
    transposition_sources=transposition_sources,
)
assert damerau_map.shape == substitution_costs.shape
```

## Shared policies

All four accept contiguous `torch.int32` `[B, 2]` lengths. Map and value
differentiate through their tensor inputs and tensor-valued scalar parameters.
Entropy differentiates through the primary score or cost input only.

Selected detached parameter VJPs are exposed under `ori.raw.lev`,
`ori.raw.lcs`, `ori.raw.osa`, and `ori.raw.damerau`. Matching stateful layers
are under `ori.nn`.

Temperature must be finite and positive. Every finite score, cost, and scoring
parameter must satisfy `abs(value) / temperature <= 80`. Use `-inf` to mask LCS
scores and `+inf` to mask cost-native inputs.
