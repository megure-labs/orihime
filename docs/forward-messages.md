# Forward messages for external decoders

`orihime.forward_messages` runs the existing native forward DP and returns its
tables without computing a structured attention map. It covers the twelve
grid algorithms plus CKY and Eisner on CPU, CUDA and ROCm. Metal, compilation and function
transforms are not supported by this interface.

```python
import torch
import orihime as ohm

scores = torch.randn(2, 16, 12)
messages = ohm.forward_messages(scores, algorithm="sv", gap_score=-1.0)
assert messages.alpha.shape == (2, 3, 17, 13)
assert messages.states == ("M", "I", "D")
```

The result uses layout version 1. SW and NW with linear gaps, DTW, LCS,
Levenshtein, OSA and Damerau return one padded prefix table. Affine SW/NW and
both SV variants return three padded state planes in M/I/D order. MAS returns
one unpadded frame/token table. `value` is the soft Bellman value for the complete
problem instance. Tables and value keep the public score/cost orientation.

The result also retains differentiable snapshots of the primary input, scalar
parameters and temperature, plus lengths, explicit edit topology and the DTW
band. `mask=True` excludes primary-input edges. Structural infinities remain in
the input metadata; the native forward receives its finite unreachable-state
sentinel. Unreachable entries are sentinels rather than probability values.
MAS requires positive token lengths and at least as many frames as tokens.

The tables and value are detached. A consumer can attach its own custom
autograd function to these outputs using the retained inputs. Greedy traceback,
stochastic traceback and arbitrary message VJPs belong in that consumer.
Orihime's ordinary map/value/entropy calls retain their existing derivatives.

A soft prefix message aggregates all prefixes reaching its state. Stochastic
traceback chooses a legal terminal according to its normalized prefix mass,
then repeatedly chooses a predecessor using its prefix mass plus the edge
score. This samples the recurrence's complete-path distribution. Greedy choices
on soft messages define a deterministic decode and generally differ from
maximum-score decoding. Maximum expected accuracy decoding instead optimizes
an additive objective on the structured attention map.

## Tree messages

CKY and Eisner also expose existing native inside tables on CPU, CUDA and HIP.
CKY scores [B,N,N,N] index inclusive (left,split,right) merges and require
`leaf_scores` [B,N]. Its alpha [B,1,N,N] contains inside scores for every span.
Scalar or per-span [B,N,N] temperature is supported. Eisner arc scores [B,N,N]
index head,dependent, with node zero as root. Its alpha [B,4,N,N] has state order
C_R,C_L,I_R,I_L; temperature is scalar. Both have positive sequence length.

`lengths` is int32 [B] and selects each root. CKY's native forward computes
all padded spans, while Eisner's existing native forward limits chart entries
to each true length. Neither returns tree indices, a posterior map or a VJP.
The leaf/input/temperature snapshots retain gradients for consumers that attach
an appropriate VJP. Tables and root values remain detached.

```python
merge_scores = torch.randn(2, 8, 8, 8) * 0.1
leaves = torch.zeros(2, 8)
inside = ohm.forward_messages(merge_scores, algorithm="cky", leaf_scores=leaves)
assert inside.states == ("I",)
assert inside.is_tree
```

A tree sampler chooses a root decomposition using child inside masses and
recursively samples its children. A scalar temperature produces a Gibbs tree
posterior. Per-span CKY temperatures define local hierarchical categoricals;
there need not be one global Gibbs temperature. Tree decoders, including exact
MEA, greedy choices and Gumbel sampling, belong to the consuming package.
