# Binary compatibility

d2p's native extension is compiled against PyTorch's C++ ABI. A binary build
therefore matches exactly one PyTorch minor version, one CPU/CUDA lane, one
CPython ABI, and one platform architecture. The wheel local version and conda
build string make that tuple explicit.

## Version and lane matrix

Linux x86-64 and ARM64 provide CPython 3.10 through 3.14 for every row below.

| PyTorch minor | Binary lanes |
| --- | --- |
| 2.9 | `cpu`, `cu126`, `cu128`, `cu129`, `cu130` |
| 2.10 | `cpu`, `cu126`, `cu128`, `cu129`, `cu130` |
| 2.11 | `cpu`, `cu126`, `cu128`, `cu129`, `cu130` |
| 2.12 | `cpu`, `cu126`, `cu129`, `cu130`, `cu132` |
| 2.13 | `cpu`, `cu126`, `cu129`, `cu130`, `cu132` |

macOS Apple silicon is CPU-only. PyTorch 2.9 and 2.10 cover CPython 3.13 and
3.14; PyTorch 2.11 through 2.13 cover CPython 3.10 through 3.14.

The complete `0.1.0` grid contains 269 wheel rows and the same 269 conda rows:
125 per Linux architecture and 19 on macOS Apple silicon.

`0.1.0` does not provide Windows, Intel-macOS, musllinux, or ROCm binaries.
Those environments may use a source build when their PyTorch toolchain is
otherwise compatible.

## Artifact identity

For PyTorch 2.13/CUDA 13.0 on CPython 3.12, the wheel has distribution version
`0.1.0+torch213cu130`. Its metadata requires `torch==2.13.*`. Wheels live on a
separate index page for each PyTorch/lane pair so incompatible files are never
co-resident.

The matching conda build is named:

```text
d2p-0.1.0-0_torch213_cu130_py312
```

Conda cannot detect a pip-installed PyTorch ABI while solving. Always select
the Torch/CUDA portion of the build string as shown in the README; the final
`py*` wildcard is safe because each package declares its exact compatible
Python minor. A bare `conda install d2p` is not a supported installation
command.

## Platform floors

- Linux wheels are tagged `manylinux_2_28` and are built against a glibc 2.28
  sysroot.
- Linux x86-64 and ARM64 builds target NVIDIA Turing (`sm_75`) and newer
  architectures supported by the selected CUDA toolkit.
- macOS ARM64 artifacts declare macOS 11.0 as their deployment target.
- d2p does not bundle PyTorch, CUDA, `libtorch`, `libc10`, or an extra OpenMP
  runtime. Those libraries come from the selected PyTorch installation.

Source distributions keep the broader `torch>=2.5` declaration because they
compile against the active environment. The prebuilt matrix above is the
binary compatibility promise; it does not imply that every historic or future
PyTorch build is ABI-compatible.
