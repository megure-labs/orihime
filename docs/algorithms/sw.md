# Smith–Waterman

`ohm.sw(...)` and `ohm.sw_affine(...)` implement differentiable local sequence
alignment. A path may start and end inside the two inputs, so poorly matched
flanks can be ignored.

## What it computes

Smith–Waterman scores monotone paths through a pairwise similarity matrix.
Diagonal steps align positions; horizontal and vertical steps create gaps; a
restart state lets the best local block begin anywhere. Orihime replaces each
hard maximum with a temperature-controlled log-sum-exp.

Each variant has map, value, and entropy functions:

- `ohm.sw(...)`, `ohm.sw_value(...)`, `ohm.sw_entropy(...)`
- `ohm.sw_affine(...)`, `ohm.sw_affine_value(...)`,
  `ohm.sw_affine_entropy(...)`

The map has shape `[B, L1, L2]` and is exactly the gradient of the value with
respect to `pair_scores`.

## Linear and affine gaps

The linear recurrence charges one `gap_score` for each gap position. The
affine recurrence tracks match, insertion, and deletion states so a run pays
`gap_open_score` once and `gap_extend_score` as it grows.

Affine gaps better model one long insertion or deletion; linear gaps are
simpler when every skipped position should cost the same. Gap arguments are
scores, so negative values penalize gaps.

## Inputs and outputs

- `pair_scores`: floating-point `[B, L1, L2]`, where higher means a better
  match.
- linear gap: `gap_score`.
- affine gaps: `gap_open_score` and `gap_extend_score`.
- `temperature`: positive Python number or floating scalar tensor.
- `lengths`: optional contiguous `torch.int32 [B, 2]` true lengths.
- `mask`: optional boolean matrix (`True` excludes a match cell).

Map functions return `[B, L1, L2]`; value and entropy functions return `[B]`.

## Quick example

```python
import torch
import orihime as ohm

torch.manual_seed(41)
pair_scores = (0.2 * torch.randn(2, 6, 7)).requires_grad_()

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

value.sum().backward()
torch.testing.assert_close(pair_scores.grad, alignment)
```

## Worked example

The motif `ACGTA` appears inside two different flanking contexts:

```python
seq1 = "TTACGTAGG"
seq2 = "CACGTATT"
a = torch.tensor([ord(char) for char in seq1])
b = torch.tensor([ord(char) for char in seq2])
motif_scores = (
    2 * (a[:, None] == b[None, :]).to(torch.float32) - 1
).unsqueeze(0)

motif_map = ohm.sw(
    motif_scores,
    gap_score=-1.0,
    temperature=0.5,
)
assert motif_map.shape == (1, len(seq1), len(seq2))
```

The high-mass diagonal through `motif_map[0, 2:7, 1:6]` identifies the shared
interior motif, while the unmatched ends carry little mass. A global alignment
would also have to account for those ends.

## Derivatives and modules

```python
layer = ohm.nn.SmithWatermanAffine(
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
    learnable=("gap_open_score", "gap_extend_score"),
)
affine_map = layer(pair_scores.detach())
affine_value = layer.value(pair_scores.detach())

assert affine_map.shape == pair_scores.shape
assert affine_value.shape == (2,)
```

Map and value differentiate through pair scores and tensor-valued model
parameters. Entropy differentiates through pair scores only. Explicit map
VJPs and parameter sensitivity maps are available under `ohm.ops.sw` and
`ohm.ops.sw_affine`.

## Variable-length batching

```python
padded_scores = torch.randn(2, 10, 12)
lengths = torch.tensor([[10, 12], [6, 7]], dtype=torch.int32)
padded_map = ohm.sw(
    padded_scores,
    gap_score=-0.7,
    lengths=lengths,
)

assert torch.count_nonzero(padded_map[1, 6:, :]) == 0
assert torch.count_nonzero(padded_map[1, :, 7:]) == 0
```

## Common pitfalls

- Smith–Waterman is local. Use [Needleman–Wunsch](nw.md) when both inputs must
  be covered end to end.
- Gap arguments are scores, so penalties are normally negative.
- `lengths` must be contiguous `torch.int32 [B, 2]` on the score device.
- `mask=True` excludes individual match cells and applies `-inf` internally;
  it is not a valid-position prefix mask.

## Complexity

Both variants take `O(B L1 L2)` time and memory. The extra gap states add a
small constant factor to affine alignment. CPU work parallelizes across batch
items.

## See also

[Needleman–Wunsch](nw.md) · [Usage guide](../usage.md) ·
[Performance](../performance.md)
