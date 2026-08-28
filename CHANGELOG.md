# Changelog

All notable public changes are recorded here. This project follows semantic
versioning from its first public release. While the version is below `1.0.0`,
minor and patch releases are not guaranteed to be backward compatible; pin an
exact version when API stability is required.

## 0.1.0 — 2026-08-28

Initial public release of differentiable dynamic-programming operators for
PyTorch.

### Included

- Smith-Waterman and Needleman-Wunsch with linear and affine gaps;
- Dynamic Time Warping;
- Levenshtein, Longest Common Subsequence, Optimal String Alignment, and
  unrestricted Damerau-Levenshtein;
- Monotonic Alignment Search;
- CKY constituency and Eisner projective dependency parsing;
- Saigo–Vert local alignment with linear and affine gaps;
- structured attention map, soft Bellman value, and Shannon entropy
  observables;
- autograd, selected `torch.func` transforms, `torch.compile`, FP32 autocast,
  explicit named operations, and stateful `orihime.nn` layers;
- CPU and NVIDIA CUDA implementations for all fourteen operator families;
- AMD HIP implementations for all fourteen public algorithms, with native
  builds, a multi-family RDNA2-through-RDNA4 release profile, and explicit
  custom targets;
- a source release for Linux and macOS; PyPI and Conda packages are coming
  soon.

### API notes

- The Python distribution is `orihime`; the import package is `orihime`.
- The conventional short import is `import orihime as ohm`.
- `orihime.ops.<algorithm>` exposes `forward`, `backward`, and `sensitivity`.
  Their shared `output=` selector returns one tensor for one string or an
  ordered name-keyed dictionary for a sequence. `forward(output=None)` returns
  `map`, `value`, and `entropy`; the derivative methods return all fields they
  support.
- Native builds compile against the active PyTorch minor and CPU/CUDA/ROCm lane.
- Explicit `grad_map` cotangents must be contiguous FP32 tensors matching the
  map's shape and device. Invalid cotangents are rejected and never normalized.
- Native dynamic programs run in FP32 and enforce the documented numerical
  domain.
- Boolean cell masks use `True` to exclude. OSA's separate
  `allowed_transpositions` topology uses `True` to allow a transposition edge.
- Windows, Intel macOS, musllinux, and prebuilt ROCm binaries are not included;
  ROCm environments build the committed HIP sources locally.
