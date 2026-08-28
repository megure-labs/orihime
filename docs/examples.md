# Examples

Each of these seven examples builds an input tensor, calls Orihime, and reads
the result. The examples use small CPU tensors, can be copied directly, and run
as part of the documentation test suite. See the [usage guide](usage.md) for
shared API rules and the [algorithm guides](algorithms/) for inputs and
recurrences.

## 1. Local sequence alignment

```python
import torch
import orihime as ohm

torch.manual_seed(21)
left = torch.randn(2, 4, 3, requires_grad=True)
right = torch.randn(2, 5, 3, requires_grad=True)
pair_scores = torch.einsum("bid,bjd->bij", left, right) / 3.0

local_map = ohm.sw(
    pair_scores,
    gap_score=-0.5,
    temperature=1.0,
)
local_value = ohm.sw_value(
    pair_scores,
    gap_score=-0.5,
    temperature=1.0,
)

assert local_map.shape == (2, 4, 5)
assert local_value.shape == (2,)

(-local_value.mean()).backward()
assert left.grad is not None and right.grad is not None
```

## 2. Global alignment with affine gaps

```python
global_scores = torch.randn(2, 4, 5)
gap_open = torch.tensor(-1.0, requires_grad=True)
gap_extend = torch.tensor(-0.25, requires_grad=True)

global_map = ohm.nw_affine(
    global_scores,
    gap_open_score=gap_open,
    gap_extend_score=gap_extend,
    temperature=1.0,
)
global_value = ohm.nw_affine_value(
    global_scores,
    gap_open_score=gap_open,
    gap_extend_score=gap_extend,
    temperature=1.0,
)
global_value.sum().backward()

assert global_map.shape == global_scores.shape
assert gap_open.grad is not None and gap_extend.grad is not None
```

## 3. Differentiable time-series cost

```python
x = torch.randn(2, 4, 2, requires_grad=True)
y = torch.randn(2, 5, 2, requires_grad=True)
pair_costs = torch.cdist(x, y).square() / 4.0

warping_map = ohm.dtw(
    pair_costs,
    temperature=0.5,
    bandwidth=None,
)
warping_value = ohm.dtw_value(
    pair_costs,
    temperature=0.5,
    bandwidth=None,
)
warping_value.mean().backward()

assert warping_map.shape == pair_costs.shape
assert x.grad is not None and y.grad is not None
```

The DTW output is a structured attention map over paths, not a row-normalized
attention matrix.

## 4. Monotonic alignment durations

```python
mas_scores = torch.randn(2, 5, 3, requires_grad=True)
mas_map = ohm.mas(mas_scores, temperature=1.0)
mas_value = ohm.mas_value(mas_scores, temperature=1.0)
durations = mas_map.sum(dim=1)

assert mas_value.shape == (2,)
assert durations.shape == (2, 3)
```

MAS requires each active frame length to be at least its active token length.

## 5. CKY and Eisner

```python
merge_scores = 0.2 * torch.randn(2, 3, 3, 3, requires_grad=True)
leaf_scores = 0.2 * torch.randn(2, 3, requires_grad=True)

merge_map = ohm.cky(
    merge_scores,
    leaf_scores,
    temperature=1.0,
)
parse_value = ohm.cky_value(
    merge_scores,
    leaf_scores,
    temperature=1.0,
)
leaf_map = ohm.cky_leaf_map(
    merge_scores,
    leaf_scores,
    temperature=1.0,
)

assert merge_map.shape == merge_scores.shape
assert leaf_map.shape == leaf_scores.shape
assert not leaf_map.requires_grad
assert parse_value.shape == (2,)
```

```python
arc_scores = 0.2 * torch.randn(2, 4, 4, requires_grad=True)
arc_map = ohm.eisner(arc_scores, temperature=1.0)
dependency_value = ohm.eisner_value(
    arc_scores,
    temperature=1.0,
)

assert arc_map.shape == arc_scores.shape
assert dependency_value.shape == (2,)
```

## 6. Fuzzy string matching

```python
def make_substitution_costs(left, right):
    result = torch.empty(1, len(left), len(right))
    for i, source in enumerate(left):
        for j, target in enumerate(right):
            result[0, i, j] = float(source != target)
    return result


substitution_costs = make_substitution_costs("kitten", "sitting")
lev_map = ohm.lev(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.5,
)
lev_value = ohm.lev_value(
    substitution_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.5,
)

assert lev_map.shape == substitution_costs.shape
assert lev_value.shape == (1,)
```

OSA uses a boolean mask to enable valid adjacent-transposition edges:

```python
osa_costs = make_substitution_costs("ca", "ac")
allowed_transpositions = torch.zeros_like(osa_costs, dtype=torch.bool)
allowed_transpositions[:, 1:, 1:] = True

osa_value = ohm.osa_value(
    osa_costs,
    insertion_cost=1.0,
    deletion_cost=1.0,
    transposition_cost=1.0,
    temperature=0.5,
    allowed_transpositions=allowed_transpositions,
)
assert osa_value.shape == (1,)
```

Build unrestricted Damerau predecessor coordinates from token IDs:

```python
source_tokens = torch.tensor([[1, 2, 3]], dtype=torch.int64)
target_tokens = torch.tensor([[2, 1, 3]], dtype=torch.int64)
damerau_costs = (source_tokens.unsqueeze(2) != target_tokens.unsqueeze(1)).float()
sources = ohm.build_damerau_transposition_sources(
    source_tokens,
    target_tokens,
)
damerau_value = ohm.damerau_value(
    damerau_costs,
    temperature=0.5,
    transposition_sources=sources,
)
assert damerau_value.shape == (1,)
```

## 7. Selected map VJPs

The `ohm.ops` API contracts a map cotangent against selected inputs or scalar
parameters without constructing an autograd graph:

```python
grad_map = torch.randn_like(lev_map).contiguous()
temperature_vjp = ohm.ops.lev.backward(
    substitution_costs,
    output="temperature",
    grad_map=grad_map,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.5,
)
all_parameter_vjps = ohm.ops.lev.backward(
    substitution_costs,
    output=("insertion_cost", "deletion_cost", "temperature"),
    grad_map=grad_map,
    insertion_cost=1.0,
    deletion_cost=1.0,
    temperature=0.5,
)

assert temperature_vjp.numel() == 1
assert tuple(all_parameter_vjps) == (
    "insertion_cost",
    "deletion_cost",
    "temperature",
)
```

## See also

[Concepts](concepts.md) · [Usage guide](usage.md) ·
[Algorithm guides](algorithms/)
