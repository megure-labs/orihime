# Needleman-Wunsch

Needleman-Wunsch produces a soft global alignment that covers both sequences
end to end. It is score-native.

## Linear gaps

```python
import torch
import d2p

torch.manual_seed(43)
pair_scores = 0.2 * torch.randn(2, 4, 5, requires_grad=True)

alignment = d2p.nw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = d2p.nw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = d2p.nw_entropy(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)
```

## Affine gaps

```python
affine_alignment = d2p.nw_affine(
    pair_scores,
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)
affine_value = d2p.nw_affine_value(
    pair_scores,
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)

assert affine_alignment.shape == pair_scores.shape
assert affine_value.shape == (2,)
```

Pass contiguous `torch.int32` `[B, 2]` `lengths` for ragged batches. Global
alignment must remain feasible for each active pair.

Map and value differentiate through `pair_scores` and tensor-valued model
parameters. Entropy differentiates through `pair_scores` only. Selected
parameter VJPs are available under `d2p.raw.nw` and `d2p.raw.nw_affine`;
stateful layers are `d2p.nn.NeedlemanWunsch` and
`d2p.nn.NeedlemanWunschAffine`.

Temperature must be finite and positive. Every finite score and scoring
parameter must satisfy `abs(value) / temperature <= 80`. Pass `mask=` (a boolean
tensor, `True` = exclude) to exclude cells; `-inf` in the score cells is the
equivalent low-level form.
