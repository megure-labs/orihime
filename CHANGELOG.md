# Changelog

All notable public changes are recorded here. This project follows semantic
versioning from its first public release.

## 0.1.0 — 2026-08-25

Initial public release of differentiable dynamic-programming operators for
PyTorch.

### Included

- Smith-Waterman and Needleman-Wunsch with linear and affine gaps;
- Dynamic Time Warping;
- Levenshtein, Longest Common Subsequence, Optimal String Alignment, and
  unrestricted Damerau-Levenshtein;
- Monotonic Alignment Search;
- CKY constituency and Eisner projective dependency parsing;
- canonical Saigo–Vert linear- and affine-gap local-alignment engine kernels;
- map, value, and entropy observables;
- autograd, selected `torch.func` transforms, `torch.compile`, FP32 autocast,
  detached raw parameter VJPs, and stateful `orihime.nn` layers;
- CPU and NVIDIA CUDA implementations for all fourteen operator families;
- AMD HIP implementations for the twelve stable operator families, with native,
  multi-family RDNA2-through-RDNA4 release, and explicit custom target modes;
- a source release for Linux and macOS; PyPI and Conda packages are coming
  soon.

### Contract notes

- The Python distribution is `orihime`; the import package is `orihime`.
- Native builds compile against the active PyTorch minor and CPU/CUDA/ROCm lane.
- Raw VJP cotangents and named HVP/full-backward derivative vectors must be
  contiguous FP32 tensors matching the primary map's shape and device;
  invalid public low-level vectors are rejected rather than normalized.
- Native dynamic programs run in FP32 and enforce the documented numerical
  domain.
- Boolean cell masks use `True` to exclude. OSA's separate
  `allowed_transpositions` topology uses `True` to allow a transposition edge.
- Windows, Intel macOS, musllinux, and prebuilt ROCm binaries are not included;
  ROCm environments build the committed HIP sources locally.
