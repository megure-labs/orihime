# Needleman–Wunsch

`ohm.nw(...)` and `ohm.nw_affine(...)` implement differentiable global sequence
alignment. Unlike local Smith–Waterman, every active position in both sequences
participates in an end-to-end path.

## What it computes

Needleman–Wunsch scores monotone paths through a pairwise similarity matrix.
Diagonal steps align two positions; horizontal and vertical steps create
gaps. Orihime replaces the hard maximum over paths with a
temperature-controlled log-sum-exp.

Each variant has map, value, and entropy functions:

- `ohm.nw(...)`, `ohm.nw_value(...)`, `ohm.nw_entropy(...)`
- `ohm.nw_affine(...)`, `ohm.nw_affine_value(...)`,
  `ohm.nw_affine_entropy(...)`

The map is the value gradient with respect to `pair_scores` and has the same
shape `[B, L1, L2]`.

## Linear and affine gaps

The linear recurrence uses one `gap_score` for every horizontal or vertical
step. The affine recurrence keeps match, insertion, and deletion states so it
can charge `gap_open_score` once and `gap_extend_score` for later positions in
the same run.

Affine gaps are usually preferable when one long insertion or deletion should
cost less than many short ones. Both gap arguments are scores: negative values
penalize gaps.

## Inputs and outputs

- `pair_scores`: floating-point `[B, L1, L2]`, where higher means a better
  match.
- linear gap: `gap_score`.
- affine gaps: `gap_open_score` and `gap_extend_score`.
- `temperature`: positive Python number or floating scalar tensor.
- `lengths`: optional contiguous `torch.int32 [B, 2]` true lengths.
- `mask`: optional boolean matrix (`True` excludes a match cell).

Map functions return `[B, L1, L2]`; value and entropy functions return `[B]`.

## Example

```python
import torch
import orihime as ohm

torch.manual_seed(43)
pair_scores = (0.2 * torch.randn(2, 6, 7)).requires_grad_()

alignment = ohm.nw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = ohm.nw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = ohm.nw_entropy(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)

value.sum().backward()
torch.testing.assert_close(pair_scores.grad, alignment)
```

The affine call changes only the gap model:

```python
affine_map = ohm.nw_affine(
    pair_scores.detach(),
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)
affine_value = ohm.nw_affine_value(
    pair_scores.detach(),
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)

assert affine_map.shape == pair_scores.shape
assert affine_value.shape == (2,)
```

## Derivatives and modules

Pass any scalar parameter as a tensor, or store selected parameters in an
`ohm.nn` module:

```python
layer = ohm.nn.NeedlemanWunschAffine(
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
    learnable=("gap_open_score", "gap_extend_score"),
)
layer_map = layer(pair_scores.detach())
assert layer_map.shape == pair_scores.shape
```

Map and value differentiate through pair scores and tensor-valued model
parameters. Entropy differentiates through pair scores only. Explicit map
VJPs and parameter sensitivity maps are available under `ohm.ops.nw` and
`ohm.ops.nw_affine`.

## Variable-length batching

```python
padded_scores = torch.randn(2, 10, 12)
lengths = torch.tensor([[10, 12], [6, 7]], dtype=torch.int32)
padded_map = ohm.nw(
    padded_scores,
    gap_score=-0.7,
    lengths=lengths,
)

assert torch.count_nonzero(padded_map[1, 6:, :]) == 0
assert torch.count_nonzero(padded_map[1, :, 7:]) == 0
```

## Common pitfalls

- Needleman–Wunsch is global. Use [Smith–Waterman](sw.md) when unmatched ends
  should be ignored.
- Gap arguments are scores, so penalties are normally negative.
- `lengths` must be contiguous `torch.int32 [B, 2]` on the score device.
- Masking match cells cannot remove the need for a feasible end-to-end path.

## Complexity

Both variants take `O(B L1 L2)` time and memory. The extra gap states add a
small constant factor to affine alignment. CPU work parallelizes across batch
items.

## See also

[Smith–Waterman](sw.md) · [Usage guide](../usage.md) ·
[Performance](../performance.md)
