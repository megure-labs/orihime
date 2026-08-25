# Dynamic Time Warping

Dynamic Time Warping aligns two time axes with a monotone path. It is
cost-native: lower costs are preferred.

```python
import torch
import orihime as ori

torch.manual_seed(47)
costs = torch.rand(2, 4, 5, requires_grad=True)

warping_map = ori.dtw(
    costs,
    temperature=0.9,
    bandwidth=None,
)
value = ori.dtw_value(
    costs,
    temperature=0.9,
    bandwidth=None,
)
entropy = ori.dtw_entropy(
    costs,
    temperature=0.9,
    bandwidth=None,
)

assert warping_map.shape == costs.shape
assert value.shape == entropy.shape == (2,)
```

`bandwidth=None` is unrestricted. A non-negative integer limits the path's
distance from the diagonal.

Pairwise `lengths` is contiguous `torch.int32` `[B, 2]`. DTW rejects a
one-sided empty instance because it has no feasible path.

```python
lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32)
ragged_value = ori.dtw_value(
    costs,
    temperature=0.9,
    lengths=lengths,
)
assert ragged_value.shape == (2,)
```

Map and value differentiate through costs and a tensor-valued temperature.
Entropy differentiates through costs only. The detached temperature VJP is
available as `ori.raw.dtw.vjp_one(..., wrt="temperature", ...)`; the stateful
layer is `ori.nn.DynamicTimeWarping`.

Temperature must be finite and positive, and finite costs must satisfy
`abs(value) / temperature <= 80`. Pass `mask=` (a boolean tensor, `True` =
exclude) to exclude cells; `+inf` in the excluded cost cells is the equivalent
low-level form.
