# Eisner

Eisner computes a score-native soft projective dependency parse. Its input and
arc-marginal map both have shape `[B, N, N]`.

```python
import torch
import orihime as ori

torch.manual_seed(59)
arc_scores = 0.2 * torch.randn(2, 4, 4, requires_grad=True)

arc_map = ori.eisner(
    arc_scores,
    temperature=0.9,
)
value = ori.eisner_value(
    arc_scores,
    temperature=0.9,
)
entropy = ori.eisner_entropy(
    arc_scores,
    temperature=0.9,
)

assert arc_map.shape == arc_scores.shape
assert value.shape == entropy.shape == (2,)
```

Variable sentence lengths use a contiguous `torch.int32` tensor shaped `[B]`
on the score device.

```python
lengths = torch.tensor([4, 3], dtype=torch.int32)
ragged_map = ori.eisner(
    arc_scores,
    temperature=0.9,
    lengths=lengths,
)
assert ragged_map.shape == arc_scores.shape
```

Map and value differentiate through arc scores and a tensor-valued
temperature. Entropy differentiates through arc scores only. The detached
temperature VJP lives under `ori.raw.eisner`; the stateful layer is
`ori.nn.Eisner`.

Temperature must be finite and positive. Every finite arc score must satisfy
`abs(value) / temperature <= 80`. Pass `mask=` (a boolean tensor, `True` =
exclude) to exclude arcs; `-inf` in the excluded arc scores is the low-level form.
