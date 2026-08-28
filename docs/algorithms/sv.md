# Saigo–Vert local alignment

`ohm.sv(...)` and `ohm.sv_affine(...)` implement the differentiable Saigo–Vert
local-alignment kernel. Like Smith–Waterman, they align subsequences rather than
requiring both inputs to be covered end to end.

## What it computes

A Saigo–Vert path is an ordered sequence of matched position pairs. Its state
graph counts each monotone matched-pair skeleton once, including one explicit
empty alignment. This path space produces the positive-semidefinite alignment
kernel described by Saigo and Vert. Ordinary Smith–Waterman uses a different
path space.

Each variant has map, value, and entropy functions:

- `ohm.sv(...)`, `ohm.sv_value(...)`, `ohm.sv_entropy(...)`
- `ohm.sv_affine(...)`, `ohm.sv_affine_value(...)`,
  `ohm.sv_affine_entropy(...)`

The map has shape `[B, L1, L2]` and is the value gradient with respect to
`pair_scores`.

## Linear and affine gaps

The recurrence tracks match (`M`), insertion (`I`), and deletion (`D`) states.
It permits an `I -> D` transition but no `D -> I` transition. Allowing both
would count both possible orders of skipping positions from the two sequences.
Paths terminate only in a match state, apart from the single empty alignment.

The linear variant applies one `gap_score` to every unmatched symbol. The
affine variant charges
`gap_open_score + (k - 1) * gap_extend_score` for a gap of length `k`.

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

torch.manual_seed(23)
pair_scores = torch.randn(2, 5, 7, requires_grad=True)

alignment = ohm.sv(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = ohm.sv_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = ohm.sv_entropy(
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
affine_map = ohm.sv_affine(
    pair_scores.detach(),
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)
assert affine_map.shape == pair_scores.shape
```

## Derivatives and modules

Map and value differentiate through pair scores and tensor-valued model
parameters. Entropy differentiates through pair scores only. Stateful layers
are `ohm.nn.SaigoVertLinear` and `ohm.nn.SaigoVertAffine`.

`ohm.ops.sv` and `ohm.ops.sv_affine` expose explicit map VJPs and full map
sensitivities to gap parameters and temperature. The public linear-gap name is
`sv`; `sv_linear` is an internal kernel-family name.

## Variable-length batching

```python
padded_scores = torch.randn(2, 10, 12)
lengths = torch.tensor([[10, 12], [6, 7]], dtype=torch.int32)
padded_map = ohm.sv(
    padded_scores,
    gap_score=-0.7,
    lengths=lengths,
)

assert torch.count_nonzero(padded_map[1, 6:, :]) == 0
assert torch.count_nonzero(padded_map[1, :, 7:]) == 0
```

## Common pitfalls

- Saigo–Vert and Smith–Waterman are different distributions even when their
  gap parameters have the same values.
- Gap arguments are scores, so penalties are normally negative.
- The empty alignment contributes to the partition function but not the
  returned map.
- `lengths` must be contiguous `torch.int32 [B, 2]` on the score device.
- `mask=True` excludes individual match cells and applies `-inf` internally.

## Complexity

Both variants take `O(B L1 L2)` time and memory on CPU, NVIDIA CUDA, and AMD
HIP. The additional gap parameters add a small constant factor to the affine
variant.

## See also

[Smith–Waterman](sw.md) · [Usage guide](../usage.md) ·
[Linear implementation](../../src/sv_linear/README.md) ·
[Affine implementation](../../src/sv_affine/README.md)
