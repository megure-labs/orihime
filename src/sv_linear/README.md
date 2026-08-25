# Canonical Saigo–Vert Linear-Gap Alignment

`sv_linear` is a real three-state differentiable local-alignment operator. It
specializes the canonical Saigo–Vert affine recurrence to one penalty `gap`
per unmatched symbol; equivalently, it uses `gap_open == gap_ext == gap`
without aliasing or dispatching to `sv_affine`.

For scores `s[i,j]` and temperature `T`:

```text
M[i,j] = s[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
I[i,j] = LSE_T(M[i-1,j] + gap, I[i-1,j] + gap)
D[i,j] = LSE_T(M[i,j-1] + gap, I[i,j-1] + gap, D[i,j-1] + gap)
S      = LSE_T(0, {M[i,j] | i >= 1, j >= 1})
```

The state graph has exactly one `I -> D` cross and no `D -> I` cross. This
orientation enumerates each monotone matched-pair skeleton exactly once.
Termination is match-only, with exactly one explicit empty alignment.

This is not ordinary linear-gap Smith–Waterman. Ordinary `sw` uses a
single-state recurrence and therefore has a different path space and partition
function, even though both accept one `gap` parameter.

## Native surface

Both CPU and CUDA expose four launch units:

- `sv_linear_forward[_cpu]`
- `sv_linear_backward[_cpu]`
- `sv_linear_hvp[_cpu]`
- `sv_linear_param_grad[_cpu]`

The registered compatibility and namespaced APIs expose forward values and
marginals, value gradients for `gap` and `temperature`, marginal backward/HVP,
and marginal sensitivities to `gap` and `temperature`.

The implementation is `O(B * L1 * L2)`. CUDA uses compensated local, warp, and
block reductions; CPU uses Kahan accumulation. Native wrappers retain checked
sizes, lengths, devices, contiguity, current-stream ownership, initialized
workspaces, caching-allocator storage, and launch/runtime error checks.
