# Concepts

Orihime provides differentiable PyTorch versions of classic dynamic programs
for sequence alignment, parsing, monotonic attention, and edit distance. This
page explains the ideas shared by all of them. See the [usage guide](usage.md)
for the API and the [algorithm guides](algorithms/) for individual recurrences.

## Differentiable dynamic programming

A dynamic program solves a structured problem by filling a table and choosing
among a few predecessors at each cell. Its value may be useful inside a larger
model, but a hard choice discards every alternative and blocks useful gradients
to the network that produced the table.

## Hard DP vs soft DP

A hard recurrence chooses one predecessor with `max` or `min`. A soft
recurrence replaces that choice with a temperature-scaled log-sum-exp:

- score-native families use a soft maximum;
- cost-native families use a soft minimum;
- the gradient of the value is a structured attention map of expected cell,
  split, or arc occupancy.

The resulting value can be used as a loss, and the map as structured
attention inside a larger model.

## What every operator returns

Every algorithm has three top-level functions:

1. `ohm.<op>(...)` returns the structured attention map.
2. `ohm.<op>_value(...)` returns the soft Bellman value for each batch item.
3. `ohm.<op>_entropy(...)` returns the Shannon entropy of the induced
   distribution over complete structures.

The term `value` is used in the Bellman-equation sense: it is the solution of
the temperature-smoothed recurrence for the complete problem instance. If `𝒵`
is the set of complete structures counted by the recurrence, then:

```text
score-native: V_T =  T log Σ_{z∈𝒵} exp(S(z) / T)
cost-native:  V_T = -T log Σ_{z∈𝒵} exp(-C(z) / T)
```

Here `S(z)` and `C(z)` are the total score and total cost assigned to structure
`z`. The dynamic program evaluates these expressions without enumerating `𝒵`.

The map has the same shape as the primary score or cost tensor. CKY returns a
merge map shaped like `merge_scores`; `ohm.cky_leaf_map(...)` is a separate
detached leaf derivative view.

For every operator, the map is the value gradient with respect to the primary
input:

```python
import torch
import orihime as ohm

torch.manual_seed(31)
pair_scores = 0.2 * torch.randn(2, 4, 5, requires_grad=True)
value = ohm.sw_value(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)
expected_map = ohm.sw(
    pair_scores,
    gap_score=-0.7,
    temperature=0.9,
)

(actual_map,) = torch.autograd.grad(value.sum(), pair_scores)
torch.testing.assert_close(actual_map, expected_map)
```

The map is the expected soft structure under the same recurrence that produced
the value.

The same recurrence defines a Gibbs distribution over complete paths, trees,
alignments, or edit scripts. Orihime returns its Shannon entropy in nats:

```text
H = -Σ_z p_T(z) log p_T(z)

score-native: H =  dV_T/dT = (V_T - E_p[S]) / T
cost-native:  H = -dV_T/dT = (E_p[C] - V_T) / T
```

The native backward pass computes the temperature derivative while calculating
the structured attention map and parameter gradients. The entropy is therefore
global to the recurrence-defined distribution; it is not the elementwise
entropy of the returned map.

## Temperature

Temperature controls how strongly alternative structures contribute. Lower
values sharpen the distribution; higher values spread mass across more
structures. Temperature is differentiable when supplied as a tensor.

The FP32 kernels support the following numerical domain:

```text
temperature is finite and positive
abs(finite score, cost, or scoring parameter) / temperature <= 80
```

Calls outside this bound raise an error and return no map or entropy.

## Score and cost orientation

Score-native families treat higher values as better:

- Smith-Waterman and Needleman-Wunsch, including affine variants
- LCS and MAS
- CKY and Eisner

Cost-native families treat lower values as better:

- DTW
- Levenshtein
- OSA
- Damerau-Levenshtein

Parameter names encode the convention (`gap_score`, `insertion_cost`, and so
on). There is no runtime orientation flag.

## Choosing an operator

- Use Smith-Waterman for local sequence alignment and Needleman-Wunsch for
  end-to-end alignment. Affine variants distinguish opening from extending a
  gap.
- Use DTW for monotone time warping and MAS for frame-to-token monotonic
  alignment.
- Use CKY for binary constituency parsing and Eisner for projective dependency
  parsing.
- Use Levenshtein, OSA, or Damerau for progressively richer edit models. Use
  LCS when match scores, rather than edit costs, are the natural input.

## Supported derivatives

Map and value functions differentiate through their tensor inputs and
tensor-valued scalar parameters. Entropy differentiates only through the
primary score or cost input. Entropy gradients for scalar parameters or CKY
`leaf_scores` raise `NotImplementedError` because the required
second-derivative blocks are not included.

Explicit map VJPs and scalar-parameter sensitivity maps are available under
`ohm.ops`; trainable modules are available under `ohm.nn`.

## See also

[Usage guide](usage.md) · [Algorithm guides](algorithms/) ·
[Examples](examples.md)
