# Monotonic Alignment Search

MAS computes a score-native frame-to-token monotonic alignment. Inputs and
maps have shape `[B, T, S]`. Every path consumes one frame at each step and
must cover every active token, so each active row requires `T >= S`.

```python
import torch
import d2p

torch.manual_seed(61)
scores = 0.2 * torch.randn(2, 5, 3, requires_grad=True)

alignment = d2p.mas(scores, temperature=0.9)
value = d2p.mas_value(scores, temperature=0.9)
entropy = d2p.mas_entropy(scores, temperature=0.9)
durations = alignment.sum(dim=1)

assert alignment.shape == scores.shape
assert value.shape == entropy.shape == (2,)
assert durations.shape == (2, 3)
```

Ragged batches use contiguous `torch.int32` `[B, 2]` lengths:

```python
lengths = torch.tensor([[5, 3], [4, 2]], dtype=torch.int32)
ragged_alignment = d2p.mas(
    scores,
    temperature=0.9,
    lengths=lengths,
)
assert ragged_alignment.shape == scores.shape
```

Map and value differentiate through scores and a tensor-valued temperature.
Entropy differentiates through scores only. The detached temperature VJP is
available under `d2p.raw.mas`; the stateful layer is
`d2p.nn.MonotonicAlignmentSearch`.

Temperature must be finite and positive. Every finite score must satisfy
`abs(value) / temperature <= 80`. Pass `mask=` (a boolean tensor, `True` =
exclude) to exclude cells; `-inf` in the excluded score cells is the low-level form.
