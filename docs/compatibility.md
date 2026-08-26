# Planned binary compatibility

**PyPI and Conda packages are coming soon.** This page records the planned
binary grid; it is not a statement that those packages are available today.

Orihime's native extension is compiled against PyTorch's C++ ABI. A future
binary build will therefore match exactly one PyTorch minor version, one
CPU/CUDA/ROCm lane, one CPython ABI, and one platform architecture. Wheel
versions and Conda build strings will make that tuple explicit.

## Version and lane matrix

The planned Linux x86-64 and ARM64 packages cover CPython 3.10 through 3.14
for every row below.

| PyTorch minor | Binary lanes |
| --- | --- |
| 2.9 | `cpu`, `cu126`, `cu128`, `cu129`, `cu130` |
| 2.10 | `cpu`, `cu126`, `cu128`, `cu129`, `cu130` |
| 2.11 | `cpu`, `cu126`, `cu128`, `cu129`, `cu130` |
| 2.12 | `cpu`, `cu126`, `cu129`, `cu130`, `cu132` |
| 2.13 | `cpu`, `cu126`, `cu129`, `cu130`, `cu132` |

The planned macOS Apple-silicon packages are CPU-only. PyTorch 2.9 and 2.10 cover CPython 3.13 and
3.14; PyTorch 2.11 through 2.13 cover CPython 3.10 through 3.14.

The planned grid contains 269 wheel rows and the same 269 Conda rows:
125 per Linux architecture and 19 on macOS Apple silicon.

The initial package grid will not provide Windows, Intel-macOS, musllinux, or ROCm binaries.
Those environments may use a source build when their PyTorch toolchain is
otherwise compatible.

ROCm source builds are available now for the twelve stable operator families.
See [Building from source](source-build.md) for native, multi-family RDNA, and
custom HIP target selection. Saigo–Vert linear and affine remain CPU/CUDA-only
in `0.1.0`.

## Continuous and offline validation

Public GitHub CI builds and runs the complete CPU test suite on Linux x86-64
and Apple Silicon. It does not claim GPU execution. NVIDIA CUDA and AMD HIP are
validated offline on Megure-controlled hardware, and GPU-affecting pull
requests must bind those results into their Kaname-compatible provenance
record. See [Change provenance and merge policy](provenance-policy.md).

## Artifact identity

For PyTorch 2.13/CUDA 13.0 on CPython 3.12, the planned wheel version is
`0.1.0+torch213cu130`. Its metadata will require `torch==2.13.*`. Wheels will
live on a separate index page for each PyTorch/lane pair so incompatible files
are never co-resident.

The matching planned Conda build is named:

```text
orihime-0.1.0-0_torch213_cu130_py312
```

Conda cannot detect a pip-installed PyTorch ABI while solving. Published
installation instructions will therefore require selecting the Torch/CUDA
portion of the build string explicitly.

## Platform floors

- Linux wheels will be tagged `manylinux_2_28` and built against a glibc 2.28
  sysroot.
- Linux x86-64 and ARM64 builds will target NVIDIA Turing (`sm_75`) and newer
  architectures supported by the selected CUDA toolkit.
- macOS ARM64 artifacts will declare macOS 11.0 as their deployment target.
- Orihime will not bundle PyTorch, CUDA, `libtorch`, `libc10`, or an extra OpenMP
  runtime. Those libraries come from the selected PyTorch installation.

Source builds use the broader `torch>=2.5` declaration because they compile
against the active environment. The planned matrix above is the future binary
compatibility target; it does not imply that every historic or future PyTorch
build is ABI-compatible.
