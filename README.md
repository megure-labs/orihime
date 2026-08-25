<h1 align="center">Orihime</h1>

<p align="center">
  <strong>Differentiable Dynamic Programming for PyTorch</strong><br>
  CPU, NVIDIA CUDA, and AMD HIP operators for alignment, parsing, monotonic attention, and edit distance.
</p>

<p align="center">
  <a href="https://github.com/megure-labs/orihime/actions/workflows/ci.yml"><img src="https://github.com/megure-labs/orihime/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://openreview.net/forum?id=ZPtK8LKcvi"><img src="https://img.shields.io/badge/Paper-ICML%202026-b31b1b.svg" alt="Paper: ICML 2026"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10--3.14-blue.svg" alt="Python 3.10 through 3.14"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.5%2B-orange.svg" alt="PyTorch 2.5+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

Orihime is Megure Labs' differentiable dynamic-programming runtime for
PyTorch. It ships **14 native operator families**: twelve stable public v3
operators on CPU, NVIDIA CUDA, and AMD HIP, plus two canonical Saigo–Vert
alignment kernels on CPU and NVIDIA CUDA. The
Python distribution and import package are both named `orihime`; examples use
the conventional short alias `ori`.

Every stable public operator exposes three plain functions:

- `ori.<op>(...)` returns the structured map or marginals;
- `ori.<op>_value(...)` returns one scalar value per batch item; and
- `ori.<op>_entropy(...)` returns one entropy per batch item.

The same functions work with autograd, selected `torch.func` transforms, and
`torch.compile`. Stateful layers live under `ori.nn`; no-graph VJPs and named
kernel bindings live under `ori.raw`.

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

- Python 3.10 through 3.14;
- PyTorch 2.5 or newer; and
- for CUDA source builds, a CUDA toolkit matching the installed PyTorch build;
  or
- for HIP source builds, a ROCm PyTorch installation and `hipcc` from its
  matching ROCm toolchain.

For editable installs, CPU-only builds, CUDA and HIP architecture selection,
and custom toolchains, see [Building from source](docs/source-build.md).

## Quick start

```python
import torch
import orihime as ori

torch.manual_seed(7)
scores = torch.randn(2, 100, 120, requires_grad=True)

alignment = ori.sw(scores, gap_score=-1.0, temperature=1.0)
value = ori.sw_value(scores, gap_score=-1.0, temperature=1.0)
entropy = ori.sw_entropy(scores, gap_score=-1.0, temperature=1.0)

print(alignment.shape)  # [2, 100, 120]
print(value.shape)      # [2]
print(entropy.shape)    # [2]

(-value.mean()).backward()
print(scores.grad.shape)  # [2, 100, 120]
```

The map has the same shape as the primary input. CKY is the one multi-input
exception: its merge map matches `merge_scores`, while
`ori.cky_leaf_map(...)` returns a detached leaf-derivative view.

## Fourteen native operator families

The stable v3 surface contains twelve operators across seven guide families:

| Family | Stable public map functions | Guide |
| --- | --- | --- |
| Smith–Waterman | `ori.sw`, `ori.sw_affine` | [Smith–Waterman](docs/algorithms/sw.md) |
| Needleman–Wunsch | `ori.nw`, `ori.nw_affine` | [Needleman–Wunsch](docs/algorithms/nw.md) |
| Dynamic Time Warping | `ori.dtw` | [DTW](docs/algorithms/dtw.md) |
| CKY | `ori.cky` | [CKY](docs/algorithms/cky.md) |
| Monotonic Alignment Search | `ori.mas` | [MAS](docs/algorithms/mas.md) |
| Eisner | `ori.eisner` | [Eisner](docs/algorithms/eisner.md) |
| Edit distance | `ori.lev`, `ori.lcs`, `ori.osa`, `ori.damerau` | [Edit distance](docs/algorithms/edit-distance.md) |

Two additional engine-level families implement the canonical Saigo–Vert
positive-semidefinite local-alignment kernel:

| Family | Engine API | Implementation note |
| --- | --- | --- |
| Saigo–Vert, linear gap | `ori.ops.sv_linear` | [Canonical linear-gap recurrence](src/sv_linear/README.md) |
| Saigo–Vert, affine gap | `ori.ops.sv_affine` | [Canonical affine-gap recurrence](src/sv_affine/README.md) |

Unlike ordinary soft Smith–Waterman, the Saigo–Vert state graph enumerates
each monotone matched-pair skeleton exactly once and includes one explicit
empty alignment. Both implementations provide CPU and CUDA forward passes,
score marginals, backward/HVP operations, and parameter sensitivities. They
are shipped and tested in `0.1.0`, but remain engine-level modules while their
high-level v3 API is finalized.

More documentation: [Concepts](docs/concepts.md) ·
[Usage](docs/usage.md) · [Examples](docs/examples.md) ·
[Compatibility](docs/compatibility.md) ·
[Performance](docs/performance.md) · [FAQ](docs/faq.md) ·
[Changelog](CHANGELOG.md).

## Features

- **Three API tiers.** Plain map/value/entropy functions for model code;
  stateful `ori.nn` modules for learned parameters; and explicit `ori.raw`
  VJP/kernel interfaces for systems work.
- **First- and second-order differentiation.** Gradients through primary
  inputs and tensor-valued scalar parameters, Hessian-vector products, and
  first-order parameter Jacobians where the family exposes them.
- **PyTorch-native transforms.** Stable operators support ordinary autograd,
  selected standalone `torch.func` transforms, `torch.compile`, and FP32
  autocast behavior.
- **Native CPU, CUDA, and HIP kernels.** The twelve stable families run on CPU,
  NVIDIA CUDA, and AMD HIP; Saigo–Vert linear and affine currently run on CPU
  and NVIDIA CUDA. CPU work is parallelized across batch elements through
  PyTorch's intra-op thread pool.
- **Explicit masking and lengths.** Boolean masks use `True` to exclude a
  cell; score- and cost-native families receive the correct internal infinity
  automatically. Variable-length batches do not require padded cells to
  participate in the recurrence.
- **Hardened boundaries.** Shape, length, dtype, device, contiguity, numerical
  domain, index width, and multi-GPU device ownership are checked before a
  native launch.

## Numerical and differentiation contract

- Native dynamic programs accumulate in FP32. `dtype=torch.float32` is the
  only explicit `dtype=` value.
- Temperature must be finite and strictly positive.
- Every finite score, cost, and scoring parameter must satisfy
  `abs(value) / temperature <= 80`.
- Map and value functions differentiate through tensor inputs and
  tensor-valued scalar parameters. Entropy differentiates through the primary
  score or cost input only.
- Raw `ori.raw.<op>.vjp` and `vjp_one` cotangents, plus named HVP and backward
  vectors, must be contiguous FP32 tensors with the exact primary-map shape
  and device. Invalid layouts are rejected rather than silently copied.

See the [usage guide](docs/usage.md) for the complete shape, scalar, empty
batch, transform, module, masking, and raw-cotangent policies.

## Citation

Orihime contains the software accompanying
**[d²p: Structured Soft Attention Is All You Need](https://openreview.net/forum?id=ZPtK8LKcvi)**
(ICML 2026).

```bibtex
@inproceedings{mogilevsky2026orihime,
  title     = {{d\textsuperscript{2}p: Structured Soft Attention Is All You Need}},
  author    = {Mogilevsky, Casey Sumagaysay and Liang, Kimberly},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/forum?id=ZPtK8LKcvi},
}
```

## Contributing, license, and provenance

Contributions from people and coding agents are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before changing
the implementation; both document the repository's clean-room and
third-party-source rules.

Orihime is licensed under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). The repository is a fresh public release cut owned by
Megure Labs and attributed to Casey Mogilevsky. Complete immutable,
append-only Kaname execution and review traces are retained privately and are
available on request; see [PROVENANCE.md](PROVENANCE.md).
