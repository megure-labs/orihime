# Changelog

All notable public changes are recorded here. This project follows semantic
versioning from its first public release.

## 0.1.0 — pending public tag

Initial public release of differentiable dynamic-programming operators for
PyTorch.

### Included

- Smith-Waterman and Needleman-Wunsch with linear and affine gaps;
- Dynamic Time Warping;
- Levenshtein, Longest Common Subsequence, Optimal String Alignment, and
  unrestricted Damerau-Levenshtein;
- Monotonic Alignment Search;
- CKY constituency and Eisner projective dependency parsing;
- map, value, and entropy observables;
- autograd, selected `torch.func` transforms, `torch.compile`, FP32 autocast,
  detached raw parameter VJPs, and stateful `d2p.nn` layers;
- CPU and NVIDIA CUDA kernels;
- prebuilt Linux x86-64, Linux ARM64, and macOS Apple-silicon artifacts across
  the documented PyTorch/Python matrix.

### Contract notes

- The Python distribution is `py-d2p`; the import package is `d2p`.
- Native binary artifacts are locked to one PyTorch minor and CPU/CUDA lane.
- Raw VJP cotangents and named HVP/full-backward derivative vectors must be
  contiguous FP32 tensors matching the primary map's shape and device;
  invalid public low-level vectors are rejected rather than normalized.
- Native dynamic programs run in FP32 and enforce the documented numerical
  domain.
- Boolean cell masks use `True` to exclude. OSA's separate
  `allowed_transpositions` topology uses `True` to allow a transposition edge.
- Windows, Intel macOS, musllinux, and ROCm binaries are not included.
