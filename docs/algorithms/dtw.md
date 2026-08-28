# Dynamic Time Warping

`ohm.dtw(...)` is differentiable Dynamic Time Warping with an optional
Sakoe–Chiba band.

## What it computes

DTW aligns sequences whose time axes advance at different rates. A monotone
path crosses a pairwise cost matrix from one corner to the other, moving right,
down, or diagonally. That lets one frame in either sequence align to several
consecutive frames in the other.

Classic DTW takes a hard minimum at each cell. Orihime uses a
temperature-controlled soft minimum instead:

- `ohm.dtw(...)` → soft path-occupancy map `[B, L1, L2]`
- `ohm.dtw_value(...)` → soft warping cost `[B]`
- `ohm.dtw_entropy(...)` → path entropy `[B]`

The map is the gradient of the value with respect to the input costs.

## Soft recurrence

```text
D[i, j] = costs[i, j] + softmin_T(
    D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
)
```

As `T` approaches zero, the value approaches hard DTW and the map concentrates
on one least-cost path. Larger temperatures spread occupancy across competing
warps.

## Inputs and outputs

- `costs`: floating-point `[B, L1, L2]`, where lower means more similar.
- `temperature`: positive Python number or floating scalar tensor.
- `lengths`: optional contiguous `torch.int32 [B, 2]` true lengths.
- `bandwidth`: `None` for the full matrix, or a non-negative integer limiting
  distance from the diagonal.
- `mask`: optional boolean matrix (`True` excludes a cell).

The map matches `costs`; value and entropy contain one scalar per batch item.
DTW rejects one-sided empty inputs because no corner-to-corner path exists.

## Example

```python
import torch
import orihime as ohm

torch.manual_seed(47)
x = torch.randn(1, 12, 4)
y = torch.randn(1, 16, 4)
costs = torch.cdist(x, y).requires_grad_()

warping_map = ohm.dtw(costs, temperature=0.5, bandwidth=None)
value = ohm.dtw_value(costs, temperature=0.5, bandwidth=None)
entropy = ohm.dtw_entropy(costs, temperature=0.5, bandwidth=None)

assert warping_map.shape == (1, 12, 16)
assert value.shape == entropy.shape == (1,)

value.sum().backward()
torch.testing.assert_close(costs.grad, warping_map)
```

`warping_map[0].sum(dim=1)` measures how much path mass each frame of `x`
absorbs. A row sum above one indicates a local stretch where the other
sequence advances while that frame is held.

## Ragged batches and banding

```python
ragged_costs = torch.rand(2, 12, 16)
lengths = torch.tensor([[12, 16], [9, 11]], dtype=torch.int32)

ragged_map = ohm.dtw(
    ragged_costs,
    temperature=0.8,
    lengths=lengths,
    bandwidth=8,
)
assert ragged_map.shape == ragged_costs.shape
```

A narrower band rejects implausible warps and prunes work. A band narrower than
the length difference can exclude every valid path. Start without a band, then
narrow it after checking that the value remains stable.

## Derivatives and modules

```python
layer = ohm.nn.DynamicTimeWarping(
    temperature=0.5,
    learnable=("temperature",),
)
layer_map = layer(costs.detach())
assert layer_map.shape == costs.shape
```

Map and value differentiate through costs and tensor-valued temperature.
Entropy differentiates through costs only. Explicit map VJPs and temperature
sensitivity maps are available under `ohm.ops.dtw`.

## Common pitfalls

- DTW consumes costs, not similarities. Negate a similarity matrix first.
- Cost scale and temperature interact through `cost / temperature`; rescaling
  one changes the effective softness of the other.
- `lengths` must be contiguous `torch.int32 [B, 2]` on the cost device.
- Masked cost cells use `+inf`, never `-inf`; `mask=True` applies that policy
  safely.

## Complexity

The unrestricted recurrence takes `O(B L1 L2)` time and memory. Banding prunes
transitions, but the current kernels still allocate full-size tables. CPU work
parallelizes across batch items.

## See also

[Usage guide](../usage.md) · [Examples](../examples.md) ·
[Performance](../performance.md)
