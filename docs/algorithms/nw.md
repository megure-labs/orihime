# Needleman-Wunsch

Needleman-Wunsch produces a soft global alignment that covers both sequences
end to end. It is score-native.

## Linear gaps

```python
import torch
import orihime as ori

torch.manual_seed(43)
pair_scores = 0.2 * torch.randn(2, 4, 5, requires_grad=True)

alignment = ori.nw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = ori.nw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = ori.nw_entropy(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)
```

## Affine gaps

```python
affine_alignment = ori.nw_affine(
    pair_scores,
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)
affine_value = ori.nw_affine_value(
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
parameter VJPs are available under `ori.raw.nw` and `ori.raw.nw_affine`;
stateful layers are `ori.nn.NeedlemanWunsch` and
`ori.nn.NeedlemanWunschAffine`.

Temperature must be finite and positive. Every finite score and scoring
parameter must satisfy `abs(value) / temperature <= 80`. Pass `mask=` (a boolean
tensor, `True` = exclude) to exclude cells; `-inf` in the score cells is the
equivalent low-level form.
