# Forward messages for external decoders

`orihime.forward_messages` runs the existing native forward DP and returns its
tables without computing a structured attention map. It covers the twelve
grid algorithms on CPU, CUDA and ROCm. CKY and Eisner use different structure
spaces and are outside this interface. Metal, compilation and function
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
