# Canonical Saigo-Vert affine local alignment

`sv_affine` is the canonical soft local-alignment kernel of Saigo and Vert. It
uses the same match/insertion/deletion state layout and affine cost convention
as `sw_affine`, but its path space enumerates every monotone matched-pair
skeleton exactly once. The resulting partition function is suitable as the
Saigo-Vert positive-semidefinite alignment kernel.

## Difference from `sw_affine`

The delete state has a one-way insertion-to-deletion transition:

```text
I[i,j-1] -- gap_open --> D[i,j]
```

Thus

```text
D[i,j] = LSE_T(M[i,j-1] + gap_open,
               I[i,j-1] + gap_open,
               D[i,j-1] + gap_ext)
```

There is deliberately no `D -> I` edge, which would count the two orders of a
both-sided skip twice. The partition terminates only at true match cells and
contains one explicit empty alignment:

```text
S = LSE_T(0, {M[i,j] : i >= 1, j >= 1})
```

Consequently, insertion/deletion cells receive no terminal beta seed, while
the empty term participates in the normalizer without owning a DP cell.

The forward, backward, score HVP, and parameter-Jacobian recurrences implement
these rules on both CPU and CUDA. The `I -> D` edge contributes to
`gap_open`, never `gap_ext`.

## Python API

```python
from d2p import ops

value, marginals = ops.sv_affine.forward(
    scores, gap_open=-2.0, gap_ext=-0.5, temp=1.0
)
hvp = ops.sv_affine.marginals_hvp(
    scores, tangent, gap_open=-2.0, gap_ext=-0.5, temp=1.0
)
```

The flat `torch.ops.d2p.sv_affine_*` names and the compatibility
`soft_sv_affine_*` names are also registered.

## Verification (2026-07-23)

The implementation was built CPU-only first and then with CUDA 13.0, both with
compile parallelism capped at 8 jobs. On an RTX 6000 Ada:

- canonical forward vs exhaustive monotone-skeleton reference: `1.28e-08`
  absolute error;
- score marginals vs central finite differences: `1.36e-04` maximum error;
- both-sided-skip SV/SW value difference: `0.205822`;
- score HVP vs directional finite differences: `2.64e-05` maximum error;
- value-gradient finite-difference errors for open/extension/temperature:
  `9.28e-05`, `1.49e-04`, and `1.19e-04`;
- marginal parameter-Jacobian maximum errors for
  open/extension/temperature: `6.98e-06`, `2.70e-05`, and `2.24e-05`;
- CPU/CUDA maximum differences: `4.77e-07` for values, `2.68e-07` for
  marginals, `2.09e-07` for HVP, and at most `2.68e-07` for parameter
  Jacobians;
- `tests/test_soft_sv_affine.py`: 7 passed;
- existing `tests/test_soft_sw_affine.py`: 63 passed, 10 skipped.
