<h1 align="center">Orihime</h1>

<p align="center">
  <strong>Differentiable Dynamic Programming for PyTorch</strong><br>
  CPU and CUDA operators for alignment, parsing, and edit-distance workloads.
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=ZPtK8LKcvi"><img src="https://img.shields.io/badge/Paper-ICML%202026-b31b1b.svg" alt="Paper: ICML 2026"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.5%2B-orange.svg" alt="PyTorch 2.5+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

Orihime provides differentiable dynamic-programming functions for PyTorch across
sequence alignment, parsing, monotonic alignment, and edit distance. Each
algorithm exposes three plain functions:

The Python distribution and import name remain `py-d2p` and `d2p` for API
compatibility.

- `d2p.<op>(...)` returns the structured map (marginals).
- `d2p.<op>_value(...)` returns one scalar value per batch item.
- `d2p.<op>_entropy(...)` returns one entropy per batch item.

Model parameters and structural arguments are keyword-only. Autograd,
standalone `torch.func` transforms, `torch.compile`, selected no-graph VJPs
under `d2p.raw`, and stateful layers under `d2p.nn` are part of the v3 API.

## Installation

### Prebuilt binaries (recommended)

Install a matching PyTorch first. Each binary is locked to the exact PyTorch
minor and CPU/CUDA lane it was built against. Then select that exact d2p build:

```bash
pip install "py-d2p==0.1.0" --only-binary py-d2p \
  --find-links https://download.orikata.ai/whl/torch-2.13+cu130.html
```

That example is for PyTorch 2.13 with CUDA 13.0. Replace `2.13` and `cu130`
with the values reported by your installed PyTorch. CPU builds use `cpu`, for
example `torch-2.13+cpu.html`. The selected page contains only that ABI lane;
`--only-binary` makes a missing binary fail instead of silently building the
PyPI source distribution.

Available Linux lanes are:

| PyTorch | CUDA lanes |
|---------|------------|
| 2.9  | cpu · cu126 · cu128 · cu129 · cu130 |
| 2.10 | cpu · cu126 · cu128 · cu129 · cu130 |
| 2.11 | cpu · cu126 · cu128 · cu129 · cu130 |
| 2.12 | cpu · cu126 · cu129 · cu130 · cu132 |
| 2.13 | cpu · cu126 · cu129 · cu130 · cu132 |

Linux wheels cover x86-64 and ARM64 with CPython 3.10 through 3.14. macOS
Apple-silicon wheels are CPU-only; the exact Python rows follow the
[compatibility matrix](docs/compatibility.md). Windows, Intel macOS, and ROCm
binaries are not provided in `0.1.0`.

The distribution is named `py-d2p`; the import name is `d2p`. Under PEP 440,
the public pin `==0.1.0` accepts the lane wheel's local version, such as
`0.1.0+torch213cu130`. `--find-links` keeps normal dependencies on PyPI while
adding only the chosen d2p wheel page. Do not flatten or combine lane pages.

Conda cannot inspect a PyTorch installation made by pip, so a generic
`conda install d2p` is unsafe. Select the PyTorch/CUDA lane in the build string;
Conda selects the matching Python build from the active environment:

```bash
conda install -c https://download.orikata.ai/conda \
  "d2p=0.1.0=0_torch213_cu130_py*"
```

Replace `213` and `cu130` with the installed PyTorch minor and lane. If no
compatible Python build exists, solving fails instead of choosing another
Torch/CUDA lane; use a supported lane or build from source.

### Build from source

Install PyTorch first, then build against that exact installation:

```bash
pip install meson-python meson ninja
git clone https://github.com/megure-labs/orihime.git
cd orihime
pip install . --no-build-isolation --no-cache-dir
```

`--no-build-isolation` prevents pip from compiling against a temporary,
different PyTorch. `--no-cache-dir` prevents reuse of a wheel compiled against
another PyTorch ABI.

Requirements:

- Python `3.10` through `3.14`
- `torch>=2.5`
- a CUDA toolkit matching the installed PyTorch build for CUDA source builds

For CPU-only builds, architecture selection, and custom toolchains, see the
[source-build guide](docs/source-build.md).

## Quick start

```python
import torch
import d2p

torch.manual_seed(7)
pair_scores = torch.randn(2, 4, 5, requires_grad=True)

alignment = d2p.sw(
    pair_scores,
    gap_score=-1.0,
    temperature=1.0,
)
value = d2p.sw_value(
    pair_scores,
    gap_score=-1.0,
    temperature=1.0,
)
entropy = d2p.sw_entropy(
    pair_scores,
    gap_score=-1.0,
    temperature=1.0,
)

assert alignment.shape == pair_scores.shape
assert value.shape == entropy.shape == (2,)

(-value.mean()).backward()
assert pair_scores.grad is not None
```

The map has the same shape as the primary input. CKY is the one multi-input
exception: its merge map matches `merge_scores`, while
`d2p.cky_leaf_map(...)` returns a detached leaf derivative view.

## Algorithm guides

| Family | Public map functions | Guide |
| --- | --- | --- |
| Smith-Waterman | `d2p.sw`, `d2p.sw_affine` | [Smith-Waterman](docs/algorithms/sw.md) |
| Needleman-Wunsch | `d2p.nw`, `d2p.nw_affine` | [Needleman-Wunsch](docs/algorithms/nw.md) |
| Dynamic Time Warping | `d2p.dtw` | [DTW](docs/algorithms/dtw.md) |
| CKY | `d2p.cky` | [CKY](docs/algorithms/cky.md) |
| Monotonic Alignment Search | `d2p.mas` | [MAS](docs/algorithms/mas.md) |
| Eisner | `d2p.eisner` | [Eisner](docs/algorithms/eisner.md) |
| Edit distance | `d2p.lev`, `d2p.lcs`, `d2p.osa`, `d2p.damerau` | [Edit distance](docs/algorithms/edit-distance.md) |

More documentation: [Concepts](docs/concepts.md) ·
[Usage](docs/usage.md) · [Examples](docs/examples.md) ·
[Compatibility](docs/compatibility.md) ·
[Performance](docs/performance.md) · [FAQ](docs/faq.md) ·
[Changelog](CHANGELOG.md).

## Numerical and differentiation contract

- Native dynamic programs accumulate in FP32. `dtype=torch.float32` is the
  only explicit `dtype=` value.
- Temperature must be finite and strictly positive.
- Every finite score, cost, and scoring parameter must satisfy
  `abs(value) / temperature <= 80`.
- To exclude cells, pass a boolean `mask=` (`True` marks excluded cells) to any
  map/value/entropy function; d2p applies the orientation-correct infinity
  internally and normalizes it to an answer-preserving finite sentinel, so you
  never handle infinities. Writing `-inf` (score-native) or `+inf` (cost-native)
  into the inputs is the equivalent low-level form.
- Map and value functions differentiate through tensor inputs and
  tensor-valued scalar parameters. Entropy differentiates through the primary
  score or cost input only.
- Raw `d2p.raw.<op>.vjp` and `vjp_one` cotangents, plus named
  `marginals_hvp` tangents and `marginals_backward` cotangents, must be
  contiguous FP32 tensors with the exact primary-map shape and device. These
  public low-level boundaries reject invalid layouts; they do not silently
  call `.contiguous()`.

See the [usage guide](docs/usage.md) for the complete shape, scalar, empty
batch, raw-cotangent, transform, module, and masking policies.

## API tiers

- High level: `d2p.<op>`, `d2p.<op>_value`, and
  `d2p.<op>_entropy`.
- Low-level tier — `d2p.raw.<op>`: no-graph VJPs (`vjp_one(...)`, `vjp(...)`,
  `vjp_fields`) plus the named kernel bindings (`forward`, `value_grad_params`,
  `marginals_backward`, ...).
- Stateful layers: `d2p.nn.SmithWaterman`,
  `d2p.nn.NeedlemanWunsch`, `d2p.nn.CKY`, and their peers.
- Dispatcher floor: `torch.ops.d2p.*`.

## Citation

d2p accompanies
**[d²p: Structured Soft Attention Is All You Need](https://openreview.net/forum?id=ZPtK8LKcvi)**
(ICML 2026).

```bibtex
@inproceedings{mogilevsky2026d2p,
  title     = {{d\textsuperscript{2}p: Structured Soft Attention Is All You Need}},
  author    = {Mogilevsky, Casey Sumagaysay and Liang, Kimberly},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/forum?id=ZPtK8LKcvi},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Report
vulnerabilities through the process in [SECURITY.md](SECURITY.md).

This repository is a fresh public release cut owned by Megure Labs and
attributed to Casey Mogilevsky. The complete private Kaname execution and
review traces are available on request; see [PROVENANCE.md](PROVENANCE.md).
