<h1 align="center">Orihime</h1>

<p align="center">
  <strong>Differentiable Dynamic Programming for PyTorch</strong><br>
  CPU, NVIDIA CUDA, and AMD HIP operators for alignment, parsing, monotonic attention, and edit distance.
</p>

<p align="center">
  <a href="https://github.com/megure-labs/orihime/actions/workflows/ci.yml"><img src="https://github.com/megure-labs/orihime/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://openreview.net/forum?id=ZPtK8LKcvi"><img src="https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg" alt="Paper on OpenReview"></a>
  <a href="https://icml.cc/virtual/2026/poster/63172"><img src="https://img.shields.io/badge/Conference-ICML%202026-4b44ce.svg" alt="ICML 2026 paper page"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10--3.14-blue.svg" alt="Python 3.10 through 3.14"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.5%2B-orange.svg" alt="PyTorch 2.5+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

Orihime is the re-release of d²p. It includes 14 differentiable
dynamic-programming algorithms for sequence alignment, parsing, monotonic
alignment, and edit distance, each implemented for CPU, NVIDIA CUDA, and AMD
HIP. The top-level API has three functions per algorithm:

- `ohm.<op>(...)` returns the structured attention map (marginals).
- `ohm.<op>_value(...)` returns the soft Bellman value for each batch item.
- `ohm.<op>_entropy(...)` returns the Shannon entropy, in nats, of the induced
  distribution over complete structures.

Entropy here is global: it measures uncertainty over complete paths, trees,
alignments, or edit scripts, not elementwise uncertainty in the attention map.

Model parameters and structural arguments are keyword-only. Stateful modules
are under `ohm.nn`. Use `ohm.ops` for explicit forward, backward, and
sensitivity operations. Autograd, selected standalone `torch.func` transforms,
and `torch.compile` are supported where documented.

> **API stability:** Orihime is pre-1.0 software. Releases before `1.0.0` are
> not guaranteed to be backward compatible; pin an exact version when a stable
> API is required.

## Installation

**PyPI and Conda packages are coming soon.** Until then, install Orihime from
source.

### Build from source

Install PyTorch first, then compile against that exact installation:

```bash
pip install meson-python meson ninja
git clone https://github.com/megure-labs/orihime.git
cd orihime
pip install . --no-build-isolation --no-cache-dir
```

`--no-build-isolation` prevents pip from compiling against a temporary,
different PyTorch. `--no-cache-dir` prevents it from reusing a wheel compiled
against another ABI.

Requirements:

- Python 3.10 through 3.14
- PyTorch 2.5 or newer
- for CUDA source builds, a CUDA toolkit matching the installed PyTorch build
- for HIP source builds, a ROCm PyTorch installation and `hipcc` from its
  matching ROCm toolchain

For editable installs, CPU-only builds, CUDA and HIP architecture selection,
and custom toolchains, see [Building from source](docs/source-build.md).

## Quick start

```python
import torch
import orihime as ohm

torch.manual_seed(7)
pair_scores = torch.randn(2, 4, 5, requires_grad=True)

alignment = ohm.sw(
    pair_scores,
    gap_score=-1.0,
    temperature=1.0,
)
value = ohm.sw_value(
    pair_scores,
    gap_score=-1.0,
    temperature=1.0,
)
entropy = ohm.sw_entropy(
    pair_scores,
    gap_score=-1.0,
    temperature=1.0,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)

(-value.mean()).backward()
assert pair_scores.grad is not None
```

The map has the same shape as the primary input. CKY is the exception: its
merge map matches `merge_scores`, while
`ohm.cky_leaf_map(...)` returns a detached leaf-derivative view.

## Algorithm guides

| Family | Top-level map functions | Guide |
| --- | --- | --- |
| Smith–Waterman | `ohm.sw`, `ohm.sw_affine` | [Smith–Waterman](docs/algorithms/sw.md) |
| Saigo–Vert local alignment | `ohm.sv`, `ohm.sv_affine` | [Saigo–Vert alignment](docs/algorithms/sv.md) |
| Needleman–Wunsch | `ohm.nw`, `ohm.nw_affine` | [Needleman–Wunsch](docs/algorithms/nw.md) |
| Dynamic Time Warping | `ohm.dtw` | [DTW](docs/algorithms/dtw.md) |
| CKY | `ohm.cky` | [CKY](docs/algorithms/cky.md) |
| Monotonic Alignment Search | `ohm.mas` | [MAS](docs/algorithms/mas.md) |
| Eisner | `ohm.eisner` | [Eisner](docs/algorithms/eisner.md) |
| Edit distance | `ohm.lev`, `ohm.lcs`, `ohm.osa`, `ohm.damerau` | [Edit distance](docs/algorithms/edit-distance.md) |

More documentation: [Concepts](docs/concepts.md) ·
[Usage](docs/usage.md) · [Examples](docs/examples.md) ·
[Compatibility](docs/compatibility.md) · [Testing](docs/testing.md) ·
[Performance](docs/performance.md) · [FAQ](docs/faq.md) ·
[Source builds](docs/source-build.md) · [Changelog](CHANGELOG.md).

## Numerical behavior and derivatives

- Native dynamic programs accumulate in FP32. `dtype=torch.float32` is the
  only explicit `dtype=` value.
- Temperature must be finite and strictly positive.
- Every finite score, cost, and scoring parameter must satisfy
  `abs(value) / temperature <= 80`.
- To exclude cells, pass a boolean `mask=` (`True` marks excluded cells).
  Orihime inserts `-inf` for score-native algorithms or `+inf` for cost-native
  algorithms, then normalizes it to an equivalent finite sentinel. Writing the
  corresponding infinity into the input directly has the same effect.
- Map and value functions differentiate through tensor inputs and
  tensor-valued scalar parameters. Entropy differentiates through the primary
  score or cost input only.
- Explicit `grad_map` cotangents must be contiguous FP32 tensors with the
  exact map shape and device. Nonconforming cotangents raise an error and are
  not silently copied into a valid layout.

The [usage guide](docs/usage.md) documents shapes, scalar parameters, empty
batches, transforms, modules, masks, and cotangents.

## API levels

- Top-level functions: `ohm.<op>`, `ohm.<op>_value`, and `ohm.<op>_entropy`.
- Modules: `ohm.nn.SmithWaterman`, `ohm.nn.SaigoVertLinear`,
  `ohm.nn.CKY`, and their peers.
- Explicit operations: `ohm.ops.<op>.forward` selects maps, values, and
  entropies; `backward` contracts a map cotangent against selected inputs and
  parameters; `sensitivity` returns full map derivatives with respect to
  scalar parameters.

Pass one field or a sequence of fields to `output=`. For `forward`, omitting
`output` returns the three standard observables—`map`, `value`, and `entropy`—in
a name-keyed dictionary. Algorithm-specific statistics are derived from value
gradients rather than added to this default result. Native dispatcher bindings
and compatibility adapters are implementation details.

## Citation

The original d²p paper is **d²p: Structured Soft Attention Is All You Need**.
Read it on [OpenReview](https://openreview.net/forum?id=ZPtK8LKcvi), or visit
the [ICML 2026 paper page](https://icml.cc/virtual/2026/poster/63172). If you
use Orihime in your research, please cite it:

```bibtex
@inproceedings{mogilevsky2026structured,
  title     = {{d\textsuperscript{2}p: Structured Soft Attention Is All You Need}},
  author    = {Mogilevsky, Casey Sumagaysay and Liang, Kimberly},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/forum?id=ZPtK8LKcvi},
}
```

## Contributing and provenance

Admission is actor-neutral: human-written and agent-written changes are subject
to identical provenance, clean-room, licensing, validation, and admission
requirements; maintainer authority can authorize work but does not exempt it
from those requirements.

External contributions are temporarily closed until Megure Labs deploys the
Kaname verifier. Forking and downstream modification remain permitted under
Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) for the admission policy and
[PROVENANCE.md](PROVENANCE.md) for the release history and trace policy.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Report
vulnerabilities through the private process described in
[SECURITY.md](SECURITY.md).
