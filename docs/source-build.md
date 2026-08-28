# Source builds

Use these steps for CPU-only, NVIDIA CUDA, AMD HIP, editable, or custom-toolchain
builds. PyPI and Conda packages are coming soon; until then, install Orihime
from source.

## Prerequisites

- Python `3.10` through `3.14`
- `torch>=2.5` installed in the active environment before building `orihime`
- A normal C++ toolchain
- `nvcc` only if you are building the CUDA lane
- `hipcc` and a ROCm PyTorch installation only if you are building the HIP lane
- The build backend when using `--no-build-isolation`:
  `pip install meson-python meson ninja`

`orihime` builds its extension against the active PyTorch installation. Use a
Python version supported by that PyTorch release; Orihime's Python range does
not make an unsupported PyTorch/Python pair buildable.

## CPU-only source build

No CUDA toolkit is required for the CPU-only lane.

```bash
git clone https://github.com/megure-labs/orihime.git
cd orihime
python -m pip install . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=disabled
```

## CUDA source build

For CUDA builds, install a matching PyTorch CUDA lane first. Then make `nvcc`
available from either a system CUDA toolkit or an environment-local package
such as `cuda-nvcc`.

```bash
git clone https://github.com/megure-labs/orihime.git
cd orihime
python -m pip install . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=enabled
```

## CUDA architectures

By default, a CUDA build targets only the attached GPU
(`orihime_cuda_arch_mode=native`), which is the fastest option to compile.
Select another target set with Meson setup arguments:

```bash
python -m pip install . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=enabled \
  --config-settings=setup-args=-Dorihime_cuda_arch_mode=release-cu130
```

- `orihime_cuda_arch_mode`: `native` (default, your GPU only), `release-cu128`,
  `release-cu130` (fat builds spanning `sm_75` through `sm_121`), or `custom`.
- `orihime_cuda_gencode`: with `-Dorihime_cuda_arch_mode=custom`, a
  semicolon-separated NVCC gencode list, for example
  `arch=compute_80,code=sm_80`.
- The `ORIHIME_CUDA_ARCH_LIST` (or `TORCH_CUDA_ARCH_LIST`) environment variable
  overrides the architecture list at build time.

## AMD HIP source build

Install a ROCm PyTorch build and make `hipcc` available from the matching ROCm
toolchain. Orihime builds one GPU lane at a time, so disable CUDA explicitly:

```bash
python -m pip install . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=disabled \
  --config-settings=setup-args=-Dhip=enabled
```

All 14 public algorithms have HIP implementations.

## AMD HIP architectures

`orihime_hip_arch_mode=native` is the default and detects the attached AMD GPU.
The release profile produces a fat binary with one generic LLVM code object for
each supported RDNA family from RDNA2 through RDNA4:

```bash
python -m pip install . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=disabled \
  --config-settings=setup-args=-Dhip=enabled \
  --config-settings=setup-args=-Dorihime_hip_arch_mode=release-rocm7-rdna
```

- `orihime_hip_arch_mode`: `native` (default), `release-rocm7-rdna`, or `custom`.
- `release-rocm7-rdna` embeds `gfx10-3-generic`, `gfx11-generic`, and
  `gfx12-generic`, covering their compatible RDNA2, RDNA3/RDNA3.5, and RDNA4
  processors under a ROCm 7 toolchain.
- With `custom`, pass a semicolon-separated list such as
  `-Dorihime_hip_targets=gfx90a;gfx942`. A custom target being accepted by
  `hipcc` proves compilation, not runtime correctness on unvalidated hardware.
- Reproducible release builds always select a named profile or explicit custom
  list; `native` is intentionally machine-dependent.

The `0.1.0` release profile is build- and runtime-validated from the same
multi-family configuration on both of these machines:

| GPU | Architecture | ROCm | PyTorch |
| --- | --- | --- | --- |
| AMD Radeon 8060S Graphics | `gfx1151` | 7.13 | 2.10 |
| AMD Radeon AI PRO R9700 | `gfx1201` | 7.14 | 2.12 |

The generic code objects cover the listed RDNA families, including both
validation GPUs. Custom CDNA targets such as `gfx90a` and `gfx942` also compile,
but this release does not claim runtime validation for them.

## Editable installs

Editable installs use the same lane selection:

```bash
python -m pip install -e . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=disabled
python -m pip install -e . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=enabled
python -m pip install -e . --no-build-isolation --no-cache-dir \
  --config-settings=setup-args=-Dcuda=disabled \
  --config-settings=setup-args=-Dhip=enabled
```

## Notes

- CPU-only artifacts import without CUDA libraries present, but do not provide
  CUDA tensor dispatch.
- CUDA builds link against the active PyTorch CUDA runtime; `nvcc` is the extra
  build-time requirement.
- HIP builds link against the active PyTorch ROCm runtime; `hipcc` is the extra
  build-time requirement. HIP and CUDA are mutually exclusive build lanes.
- The package distribution and import package are both named `orihime`.

## See also

[Compatibility](compatibility.md) · [Usage guide](usage.md) ·
[Performance](performance.md)
