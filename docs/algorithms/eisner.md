# Eisner

`ohm.eisner(...)` implements differentiable projective dependency parsing. Use
it when a model needs soft head-to-dependent arc marginals over every
projective tree rather than one hard parse.

## What it computes

A dependency tree gives every non-root token exactly one head. It is
projective when no arcs cross with the sentence laid out in order. Eisner's
chart represents complete and incomplete spans so it can sum over all
projective trees in cubic time.

The top-level functions return three views of that distribution:

- `ohm.eisner(...)` → arc-marginal map `[B, N, N]`
- `ohm.eisner_value(...)` → soft Bellman value `[B]`
- `ohm.eisner_entropy(...)` → tree entropy `[B]`

The map entry `[b, h, d]` is the posterior probability of arc `h -> d` and is
the value gradient with respect to `arc_scores[b, h, d]`.

## Soft inside recurrence

Eisner combines complete and incomplete spans over every legal split point,
adding an arc score when a head attaches a dependent. Orihime replaces each
hard maximum with `T * logsumexp(scores / T)`. As `T` approaches zero, the
distribution sharpens toward the Viterbi projective tree; larger temperatures
spread mass across alternatives.

## Inputs and outputs

- `arc_scores`: floating-point `[B, N, N]`, indexed head first.
- `temperature`: positive Python number or floating scalar tensor.
- `lengths`: optional contiguous `torch.int32 [B]`, including position zero.
- `mask`: optional boolean matrix (`True` excludes an arc).

Position zero is the artificial root. Self-arcs and arcs into the root do not
belong to a valid tree and are ignored. For every active non-root dependent,
incoming arc mass sums to one.

## Example

```python
import torch
import orihime as ohm

torch.manual_seed(59)
arc_scores = (0.2 * torch.randn(2, 6, 6)).requires_grad_()

arc_map = ohm.eisner(arc_scores, temperature=0.9)
value = ohm.eisner_value(arc_scores, temperature=0.9)
entropy = ohm.eisner_entropy(arc_scores, temperature=0.9)
incoming_mass = arc_map.sum(dim=1)

assert arc_map.shape == arc_scores.shape
assert value.shape == entropy.shape == (2,)
torch.testing.assert_close(incoming_mass[:, 0], torch.zeros(2))
torch.testing.assert_close(incoming_mass[:, 1:], torch.ones(2, 5))

value.sum().backward()
torch.testing.assert_close(arc_scores.grad, arc_map)
```

The most probable head for each real token is
`arc_map[:, :, 1:].argmax(dim=1)`. That argmax is a lossy summary; train
against the full marginal map when possible.

## Variable-length batches

```python
padded_scores = torch.randn(2, 8, 8)
lengths = torch.tensor([8, 5], dtype=torch.int32)
padded_map = ohm.eisner(
    padded_scores,
    temperature=1.0,
    lengths=lengths,
)

assert torch.count_nonzero(padded_map[1, :, 5:]) == 0
assert torch.count_nonzero(padded_map[1, 5:, :]) == 0
```

## Derivatives and modules

```python
layer = ohm.nn.Eisner(
    temperature=0.9,
    learnable=("temperature",),
)
layer_map = layer(arc_scores.detach())
assert layer_map.shape == arc_scores.shape
```

Map and value differentiate through arc scores and tensor-valued temperature.
Entropy differentiates through arc scores only. Explicit map VJPs and
temperature sensitivity maps are available under `ohm.ops.eisner`.

## Common pitfalls

- The index order is `[head, dependent]`, not `[dependent, head]`.
- Position zero is root and is included in each length.
- The marginal matrix is not itself a hard tree; independent per-token
  argmaxes need not be the highest-probability globally valid tree.
- `lengths` must be contiguous `torch.int32 [B]` on the score device.
- Masked score cells use `-inf`; `mask=True` applies that safely.

## Complexity

Eisner takes `O(B N^3)` time and `O(B N^2)` working memory. CPU work
parallelizes across batch items.

## See also

[Usage guide](../usage.md) · [Examples](../examples.md) ·
[Performance](../performance.md)
