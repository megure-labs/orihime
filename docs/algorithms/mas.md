# Monotonic Alignment Search

`ohm.mas(...)` is differentiable Monotonic Alignment Search for assigning a
long frame sequence to a shorter ordered token sequence.

## What it computes

MAS follows a monotone path through scores shaped `[B, T, S]`, where `T` is
the number of frames and `S` the number of tokens. Every frame is assigned to
one token, token order is preserved, and every active token must be covered.
That last condition requires `T >= S` for every batch item.

The top-level functions return:

- `ohm.mas(...)` → frame-to-token assignment map `[B, T, S]`
- `ohm.mas_value(...)` → soft Bellman value `[B]`
- `ohm.mas_entropy(...)` → alignment entropy `[B]`

The map is the value gradient with respect to the input scores. Summing it over
the frame axis gives expected token durations.

## Soft recurrence

At frame `t` and token `s`, a valid path either remains on token `s` or
advances from `s - 1`. The soft recurrence uses a temperature-scaled
log-sum-exp over those choices. As temperature approaches zero, the map
concentrates on one hard monotone assignment.

## Inputs and outputs

- `scores`: floating-point `[B, T, S]`, where higher means a better assignment.
- `temperature`: positive Python number or floating scalar tensor.
- `lengths`: optional contiguous `torch.int32 [B, 2]` frame/token lengths.
- `mask`: optional boolean score-shaped tensor (`True` excludes a cell).

Each active row of the map sums to one. Padded rows and columns contain zero.

## Example

```python
import torch
import orihime as ohm

torch.manual_seed(61)
scores = (0.2 * torch.randn(2, 8, 4)).requires_grad_()

alignment = ohm.mas(scores, temperature=0.9)
value = ohm.mas_value(scores, temperature=0.9)
entropy = ohm.mas_entropy(scores, temperature=0.9)
durations = alignment.sum(dim=1)

assert alignment.shape == scores.shape
assert value.shape == entropy.shape == (2,)
assert durations.shape == (2, 4)
torch.testing.assert_close(alignment.sum(dim=2), torch.ones(2, 8))

value.sum().backward()
torch.testing.assert_close(scores.grad, alignment)
```

`alignment.argmax(dim=2)` gives the highest-mass token at each frame, while
`durations` retains the soft expected duration of every token.

## Variable-length batches

```python
padded_scores = torch.randn(2, 8, 4)
lengths = torch.tensor([[8, 4], [6, 3]], dtype=torch.int32)
ragged_map = ohm.mas(
    padded_scores,
    temperature=0.9,
    lengths=lengths,
)

assert torch.count_nonzero(ragged_map[1, 6:, :]) == 0
assert torch.count_nonzero(ragged_map[1, :, 3:]) == 0
```

## Derivatives and modules

```python
layer = ohm.nn.MonotonicAlignmentSearch(
    temperature=0.9,
    learnable=("temperature",),
)
layer_map = layer(scores.detach())
assert layer_map.shape == scores.shape
```

Map and value differentiate through scores and tensor-valued temperature.
Entropy differentiates through scores only. Explicit map VJPs and temperature
sensitivity maps are available under `ohm.ops.mas`.

## Common pitfalls

- The input order is frames first, tokens second: `[B, T, S]`.
- Every active row must satisfy `T >= S`; otherwise no full-coverage path
  exists.
- Map rows, not columns, sum to one. Column sums are expected durations.
- `lengths` must be contiguous `torch.int32 [B, 2]` on the score device.
- Masked score cells use `-inf`; `mask=True` applies that safely.

## Complexity

MAS takes `O(B T S)` time and memory. CPU work parallelizes across batch
items.

## See also

[Usage guide](../usage.md) · [Examples](../examples.md) ·
[Performance](../performance.md)
