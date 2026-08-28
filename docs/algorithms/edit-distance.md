# Edit-distance families

Orihime provides four edit-distance operators:

- `ohm.lev(...)`: insert, delete, and substitute
- `ohm.lcs(...)`: longest common subsequence, expressed as a score
- `ohm.osa(...)`: Levenshtein plus restricted adjacent transposition
- `ohm.damerau(...)`: Levenshtein plus unrestricted transposition sources

Use these when the question is “how much editing turns one sequence into the
other?” rather than “where do they best align?”

## What they compute

Levenshtein, OSA, and Damerau–Levenshtein minimize edit costs. LCS differs: it
maximizes rewards for an ordered common subsequence. Every family
takes a primary matrix `[B, L1, L2]` and returns a map of the same shape, plus
separate value and entropy functions returning `[B]`.

For the cost families, the map is the gradient of the soft distance with
respect to substitution costs. For LCS, it is the gradient of the soft score
with respect to match scores.

## Soft recurrence

The cost recurrence is schematically:

```text
D[i, j] = softmin_T(
    D[i - 1, j - 1] + substitution_costs[i, j],
    D[i - 1, j]     + deletion_cost,
    D[i, j - 1]     + insertion_cost,
    optional transposition edge
)
```

LCS uses the analogous soft maximum over match, skip-left, and skip-right.
As temperature approaches zero, each soft recurrence approaches its hard
minimum or maximum.

## Levenshtein

`ohm.lev` uses insertion, deletion, and substitution costs:

```python
import torch
import orihime as ohm

torch.manual_seed(67)
substitution_costs = torch.rand(2, 4, 5).requires_grad_()

lev_map = ohm.lev(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.9,
)
lev_value = ohm.lev_value(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.9,
)
lev_entropy = ohm.lev_entropy(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.9,
)

assert lev_map.shape == substitution_costs.shape
assert lev_value.shape == lev_entropy.shape == (2,)

lev_value.sum().backward()
torch.testing.assert_close(substitution_costs.grad, lev_map)
```

## Longest common subsequence

LCS consumes rewards, not costs. Higher `match_scores` make a paired position
more desirable:

```python
match_scores = torch.rand(2, 4, 5)
lcs_map = ohm.lcs(match_scores, temperature=0.9)
lcs_value = ohm.lcs_value(match_scores, temperature=0.9)
lcs_entropy = ohm.lcs_entropy(match_scores, temperature=0.9)

assert lcs_map.shape == match_scores.shape
assert lcs_value.shape == lcs_entropy.shape == (2,)
```

## Optimal String Alignment

OSA adds one restricted adjacent-transposition edge. Supply a boolean
`allowed_transpositions` matrix with the same shape as substitution costs;
`True` permits a swap ending at that cell. This polarity differs from
`mask=True`, which excludes a cell.

```python
osa_costs = torch.rand(2, 4, 5)
allowed_transpositions = torch.zeros_like(osa_costs, dtype=torch.bool)
allowed_transpositions[:, 1:, 1:] = True

osa_map = ohm.osa(
    osa_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    transposition_cost=1.0,
    temperature=0.9,
    allowed_transpositions=allowed_transpositions,
)
osa_value = ohm.osa_value(
    osa_costs,
    transposition_cost=1.0,
    temperature=0.9,
    allowed_transpositions=allowed_transpositions,
)

assert osa_map.shape == osa_costs.shape
assert osa_value.shape == (2,)
```

## Damerau–Levenshtein

Damerau uses an `int32 [B, L1, L2, 2]` table naming the earlier source cell of
each permitted transposition. Build it from token IDs rather than constructing
the predecessor table by hand:

```python
source_tokens = torch.tensor(
    [[1, 2, 3, 4], [4, 3, 2, 1]],
    dtype=torch.int64,
)
target_tokens = torch.tensor(
    [[2, 1, 3, 4, 5], [4, 2, 3, 1, 0]],
    dtype=torch.int64,
)
transposition_sources = ohm.build_damerau_transposition_sources(
    source_tokens,
    target_tokens,
)
damerau_costs = (
    source_tokens[:, :, None] != target_tokens[:, None, :]
).to(torch.float32)

damerau_map = ohm.damerau(
    damerau_costs,
    temperature=0.9,
    transposition_sources=transposition_sources,
)
assert damerau_map.shape == damerau_costs.shape
assert transposition_sources.shape == (2, 4, 5, 2)
assert transposition_sources.dtype == torch.int32
```

OSA permits only fixed adjacent swaps and does not allow a transposed pair to
be edited again. Damerau can point to any valid earlier source cell and charges
the insertion/deletion cost of characters spanned by the jump.

## Worked string example

```python
def mismatch_costs(source: str, target: str) -> torch.Tensor:
    source_ids = torch.tensor([ord(char) for char in source])
    target_ids = torch.tensor([ord(char) for char in target])
    return (
        source_ids[:, None] != target_ids[None, :]
    ).to(torch.float32).unsqueeze(0)

kitten_sitting = mismatch_costs("kitten", "sitting")
soft_distance = ohm.lev_value(
    kitten_sitting,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.1,
)
kitten_sitting_map = ohm.lev(
    kitten_sitting,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.1,
)

assert soft_distance.shape == (1,)
assert kitten_sitting_map.shape == (1, 6, 7)
```

At low temperature, the soft value approaches the familiar hard edit distance
of three while retaining gradients through the substitution matrix.

## Derivatives, modules, and batching

Insertion, deletion, transposition, and temperature parameters may be scalar
tensors with gradients. The stateful layers are
`ohm.nn.LongestCommonSubsequence`, `ohm.nn.Levenshtein`,
`ohm.nn.OptimalStringAlignment`, and `ohm.nn.DamerauLevenshtein`. Explicit map
VJPs and parameter sensitivity maps are under `ohm.ops.lcs`, `ohm.ops.lev`,
`ohm.ops.osa`, and `ohm.ops.damerau`.

All four families accept contiguous `torch.int32 [B, 2]` lengths:

```python
padded_costs = torch.rand(3, 10, 12)
lengths = torch.tensor([[10, 12], [6, 7], [8, 5]], dtype=torch.int32)
ragged_map = ohm.lev(padded_costs, lengths=lengths, temperature=0.5)

assert torch.count_nonzero(ragged_map[1, 6:, :]) == 0
assert torch.count_nonzero(ragged_map[1, :, 7:]) == 0
```

## Common pitfalls

- Cost families prefer lower values; LCS prefers higher match scores.
- `allowed_transpositions` is boolean `[B, L1, L2]`, while Damerau's
  `transposition_sources` is `int32 [B, L1, L2, 2]`.
- `-1` disables a Damerau source; zero is a valid coordinate.
- `lengths` must be contiguous `torch.int32 [B, 2]` on the input device.
- Masked cost cells use `+inf`; masked LCS score cells use `-inf`.

## Complexity

All four families take `O(B L1 L2)` time and memory. The transposition
transition adds a small constant factor to OSA and Damerau. CPU work
parallelizes across batch items.

## See also

[Usage guide](../usage.md) · [Examples](../examples.md) ·
[Performance](../performance.md)
