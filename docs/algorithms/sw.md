# Smith-Waterman

Smith-Waterman finds a soft local alignment: a path may start and end inside
the two sequences. Both variants are score-native, so higher pair and gap
scores are preferred.

## Linear gaps

```python
import torch
import d2p

torch.manual_seed(41)
pair_scores = 0.2 * torch.randn(2, 4, 5, requires_grad=True)

alignment = d2p.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
value = d2p.sw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
entropy = d2p.sw_entropy(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)
```

`gap_score=0.0` and `temperature=1.0` are the defaults.

## Affine gaps

The affine variant distinguishes opening a gap from extending it.

```python
affine_map = d2p.sw_affine(
    pair_scores,
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)
affine_value = d2p.sw_affine_value(
    pair_scores,
    gap_open_score=-1.0,
    gap_extend_score=-0.25,
    temperature=0.9,
)

assert affine_map.shape == pair_scores.shape
assert affine_value.shape == (2,)
```

Both functions accept contiguous `torch.int32` `[B, 2]` `lengths`. Padded
map cells are zero.

## Derivatives and modules

Autograd through the value yields the alignment map on `pair_scores`.
Autograd through the map yields contracted second-order derivatives. Entropy
differentiates through `pair_scores` only.

Selected detached parameter VJPs are exposed by `d2p.raw.sw` and
`d2p.raw.sw_affine`.

```python
cotangent = torch.randn_like(alignment)
temperature_vjp = d2p.raw.sw.vjp_one(
    pair_scores.detach(),
    wrt="temperature",
    cotangent=cotangent,
    gap_score=-0.7,
    temperature=0.9,
)
assert temperature_vjp.numel() == 1

layer = d2p.nn.SmithWaterman(
    gap_score=-0.7,
    temperature=0.9,
    learnable=("gap_score",),
)
assert layer(pair_scores.detach()).shape == pair_scores.shape
```

Temperature must be finite and positive. Every finite score and scoring
parameter must satisfy `abs(value) / temperature <= 80`. Pass `mask=` (a boolean
tensor, `True` = exclude) to exclude cells; writing `-inf` into excluded score
cells is the equivalent low-level form.
