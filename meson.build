project('orihime', 'cpp',
  version : '0.1.0',
  # The HIP lane passes custom_target outputs through `objects:`, supported by
  # the Meson floor already declared in pyproject.toml.
  meson_version : '>=1.2.0',
  default_options : [
    'cpp_std=c++17',
    'buildtype=release',
    'warning_level=2',
  ]
)

# Find Python and PyTorch
# pure: false ensures all files are installed to platlib (with the .so)
python = import('python').find_installation('python3', pure: false)
python_dep = python.dependency()

fs = import('fs')

cuda_option = get_option('cuda')
hip_option = get_option('hip')

# All 14 public operator families have committed HIP implementations.
hip_ops = [
  'cky', 'sw', 'sw_affine', 'sv_linear', 'sv_affine', 'nw', 'nw_affine', 'dtw',
  'lev', 'lcs', 'osa', 'damerau', 'mas', 'eisner',
]

# PyTorch exposes both native CUDA and ROCm through the CUDA dispatch key, so
# the two GPU lanes cannot coexist in one extension.
if cuda_option.enabled() and hip_option.enabled()
  error('-Dcuda=enabled and -Dhip=enabled are mutually exclusive: Orihime builds one GPU lane at a time.')
endif

cuda_enabled = false
if not cuda_option.disabled() and not hip_option.enabled()
  cuda_enabled = add_languages('cuda', required : cuda_option.enabled())
endif

hip_enabled = false
hipcc = disabler()
hip_runtime_lib_dirs = []
hip_runtime_library = ''
hip_runtime_include_dirs = []
if not hip_option.disabled() and not cuda_enabled
  torch_rocm_result = run_command(python, '-c',
    'import torch; print(torch.version.hip or "")',
    check : false
  )
  torch_is_rocm = torch_rocm_result.returncode() == 0 and torch_rocm_result.stdout().strip() != ''

  if hip_option.enabled() and not torch_is_rocm
    error('-Dhip=enabled requires a ROCm PyTorch (torch.version.hip is unset for this interpreter).')
  endif

  hip_missing = []
  foreach op : hip_ops
    foreach name : ['kernels_gpu.hip', 'kernels_gpu.hiph', 'torch_hip.cpp']
      if not fs.is_file('src' / op / name)
        hip_missing += 'src' / op / name
      endif
    endforeach
  endforeach
  hip_sources_present = hip_missing.length() == 0

  if hip_option.enabled() and not hip_sources_present
    error('-Dhip=enabled but this tree is missing HIP backend sources: ' + ' '.join(hip_missing))
  endif

  if torch_is_rocm and hip_sources_present
    hipcc = find_program('hipcc', '/opt/rocm/bin/hipcc', required : hip_option.enabled())
    hip_enabled = hipcc.found()
    if hip_enabled
      message('Orihime HIP build: PyTorch ROCm ' + torch_rocm_result.stdout().strip())
      hipconfig = find_program('hipconfig', required : false)
      if hipconfig.found()
        hip_root_result = run_command(hipconfig, '--path', check : false)
        if hip_root_result.returncode() == 0
          hip_root = hip_root_result.stdout().strip()
          if fs.is_file(hip_root / 'include' / 'hip' / 'hip_runtime.h')
            hip_runtime_include_dirs += hip_root / 'include'
          endif
          foreach libdir : [hip_root / 'lib', hip_root / 'lib64']
            if fs.is_dir(libdir)
              hip_runtime_lib_dirs += libdir
              foreach filename : ['libamdhip64.so', 'libamdhip64.so.7', 'libamdhip64.so.6']
                candidate = libdir / filename
                if hip_runtime_library == '' and fs.is_file(candidate)
                  hip_runtime_library = candidate
                endif
              endforeach
            endif
          endforeach
        endif
      endif
    endif
  endif
endif

if cuda_enabled
  message('orihime CUDA build: enabled')
else
  message('orihime CUDA build: disabled')
endif

if hip_enabled
  message('Orihime HIP build: enabled')
else
  message('Orihime HIP build: disabled')
endif

torch_path_kind = 'cpu'
if cuda_enabled or hip_enabled
  torch_path_kind = 'cuda'
endif

# Get PyTorch paths from Python
torch_include_result = run_command(python, '-c',
  '''
from torch.utils.cpp_extension import include_paths

kind = '@0@'
try:
    paths = include_paths(kind)
except Exception:
    if kind != 'cpu':
        raise
    paths = include_paths()

print(';'.join(paths))
'''.format(torch_path_kind),
  check : true
)
torch_lib_result = run_command(python, '-c',
  '''
from torch.utils.cpp_extension import library_paths

kind = '@0@'
try:
    paths = library_paths(kind)
except Exception:
    if kind != 'cpu':
        raise
    paths = library_paths()

print(';'.join(paths))
'''.format(torch_path_kind),
  check : true
)
torch_abi_result = run_command(python, '-c',
  'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))',
  check : true
)

torch_includes = torch_include_result.stdout().strip().split(';')
torch_lib_dirs = torch_lib_result.stdout().strip().split(';')
torch_abi = torch_abi_result.stdout().strip()

message('PyTorch include paths: ' + torch_include_result.stdout().strip())
message('PyTorch library paths: ' + torch_lib_result.stdout().strip())
message('PyTorch CXX11 ABI: ' + torch_abi)

# Create include directories and raw include flags for hipcc custom targets.
# Skip reported include paths that do not actually exist: conda-forge's torch
# advertises a bogus torch/include/torch/csrc/api/include (the real C++ API headers
# live under the env prefix, which it also reports), and meson hard-fails on a
# missing include dir. No-op for pip's torch, where every reported path exists.
torch_inc_dirs = []
torch_inc_flags = []
foreach inc : torch_includes
  if inc != '' and fs.is_dir(inc)
    torch_inc_dirs += include_directories(inc, is_system : true)
    torch_inc_flags += '-I' + inc
  endif
endforeach

# Ignore stale library directories reported by some ROCm PyTorch builds.
torch_lib_dirs_present = []
foreach libdir : torch_lib_dirs
  if libdir != '' and fs.is_dir(libdir)
    torch_lib_dirs_present += libdir
  endif
endforeach
torch_lib_dirs = torch_lib_dirs_present

# Add local src directory for common/ headers
# Allows: #include "common/numerics.cuh"
src_inc = include_directories('src')

# Find torch libraries
cpp = meson.get_compiler('cpp')
torch_lib = cpp.find_library('torch', dirs : torch_lib_dirs, required : true)
torch_cpu_lib = cpp.find_library('torch_cpu', dirs : torch_lib_dirs, required : true)
c10_lib = cpp.find_library('c10', dirs : torch_lib_dirs, required : true)

torch_deps = [torch_lib, torch_cpu_lib, c10_lib]
if cuda_enabled
  torch_cuda_lib = cpp.find_library('torch_cuda', dirs : torch_lib_dirs, required : true)
  c10_cuda_lib = cpp.find_library('c10_cuda', dirs : torch_lib_dirs, required : true)
  torch_deps += [torch_cuda_lib, c10_cuda_lib]
endif
if hip_enabled
  hip_link_dirs = torch_lib_dirs + hip_runtime_lib_dirs
  torch_hip_lib = cpp.find_library('torch_hip', dirs : hip_link_dirs, required : true)
  c10_hip_lib = cpp.find_library('c10_hip', dirs : hip_link_dirs, required : true)
  amdhip_lib = cpp.find_library('amdhip64', dirs : hip_link_dirs, required : false)
  if not amdhip_lib.found()
    if hip_runtime_library == ''
      error('HIP is enabled but libamdhip64 was not found in PyTorch or hipconfig library paths.')
    endif
    amdhip_lib = declare_dependency(link_args : [hip_runtime_library])
  endif
  torch_deps += [torch_hip_lib, c10_hip_lib, amdhip_lib]
endif

# Compiler flags
cxx_abi_flag = '-D_GLIBCXX_USE_CXX11_ABI=' + torch_abi
cxx_args = ['-O3', cxx_abi_flag, '-DUSE_TORCH_LIBRARY']
cuda_args = ['-O3', '-std=c++17', '-DUSE_TORCH_LIBRARY', '--expt-relaxed-constexpr']

# Required by ROCm headers and the host-side HIP binding translation units.
hip_defines = [
  '-D__HIP_PLATFORM_AMD__',
  '-DUSE_ROCM=1',
  '-DHIP_ENABLE_WARP_SYNC_BUILTINS',
]
if hip_enabled
  cxx_args += hip_defines
endif

# Enable OpenMP so ATen's at::parallel_for actually threads the CPU kernels (each
# batch element's DP runs on a torch/ATen worker thread). Without -fopenmp on the
# host C++ compiles, at::parallel_for compiles to its SERIAL fallback (measured:
# ~1x with vs ~7.6x on 8 threads once enabled). Applied to the host C++ sources
# only (cxx_args / link), never to nvcc's cuda_args. Guarded so toolchains that do
# not accept -fopenmp still build (they fall back to the current serial behavior).
openmp_link_args = []
if cpp.has_argument('-fopenmp')
  cxx_args += '-fopenmp'
  openmp_link_args = ['-fopenmp']
endif

cuda_arch_mode = get_option('orihime_cuda_arch_mode')
cuda_gencode_args = []

if cuda_enabled
  message('orihime CUDA arch mode: ' + cuda_arch_mode)
  if cuda_arch_mode == 'native'
    native_cuda_gencode_result = run_command(python, '-c',
      '''
import os
import re

def expand_arch_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []
    wants_ptx = False
    if token.endswith("+PTX"):
        wants_ptx = True
        token = token[:-4]
    token = token.replace(".", "")
    if not token.isdigit():
        raise SystemExit(f"unsupported CUDA arch token: {token!r}")
    flags = [f"-gencode=arch=compute_{token},code=sm_{token}"]
    if wants_ptx:
        flags.append(f"-gencode=arch=compute_{token},code=compute_{token}")
    return flags

arch_list = os.environ.get("ORIHIME_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST") or ""
flags: list[str] = []

if arch_list:
    for token in re.split(r"[;, ]+", arch_list):
        if token:
            flags.extend(expand_arch_token(token))
else:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        token = f"{major}{minor}"
        flags.append(f"-gencode=arch=compute_{token},code=sm_{token}")

print(chr(10).join(flags))
''',
      check : true
    )
    native_cuda_gencode = native_cuda_gencode_result.stdout().strip()
    if native_cuda_gencode != ''
      foreach flag : native_cuda_gencode.split('\n')
        if flag != ''
          cuda_gencode_args += flag
        endif
      endforeach
    endif
  elif cuda_arch_mode == 'release-cu128'
    cuda_gencode_args += [
      '-gencode=arch=compute_75,code=sm_75',
      '-gencode=arch=compute_80,code=sm_80',
      '-gencode=arch=compute_86,code=sm_86',
      '-gencode=arch=compute_89,code=sm_89',
      '-gencode=arch=compute_90,code=sm_90',
      '-gencode=arch=compute_100,code=sm_100',
      '-gencode=arch=compute_101,code=sm_101',
      '-gencode=arch=compute_120,code=sm_120',
      '-gencode=arch=compute_120,code=compute_120',
    ]
  elif cuda_arch_mode == 'release-cu130'
    cuda_gencode_args += [
      '-gencode=arch=compute_75,code=sm_75',
      '-gencode=arch=compute_80,code=sm_80',
      '-gencode=arch=compute_86,code=sm_86',
      '-gencode=arch=compute_89,code=sm_89',
      '-gencode=arch=compute_90,code=sm_90',
      '-gencode=arch=compute_100,code=sm_100',
      '-gencode=arch=compute_103,code=sm_103',
      # CUDA 13.0 renumbered SM101 as SM110.
      '-gencode=arch=compute_110,code=sm_110',
      '-gencode=arch=compute_120,code=sm_120',
      '-gencode=arch=compute_121,code=sm_121',
      '-gencode=arch=compute_121,code=compute_121',
    ]
  elif cuda_arch_mode == 'custom'
    custom_cuda_gencode = get_option('orihime_cuda_gencode')
    if custom_cuda_gencode == ''
      error('orihime_cuda_arch_mode=custom requires -Dorihime_cuda_gencode=arch=compute_XX,code=sm_XX;...')
    endif
    foreach entry : custom_cuda_gencode.split(';')
      if entry != ''
        cuda_gencode_args += '-gencode=' + entry
      endif
    endforeach
  endif
endif

if cuda_enabled and cuda_gencode_args.length() > 0
  cuda_args += cuda_gencode_args
endif

# Meson has no HIP language frontend, so each committed device translation unit
# is compiled explicitly with hipcc and linked into the Python extension as an
# ordinary position-independent object. Host-side torch_hip.cpp files continue
# through Meson's C++ compiler.
hip_objects = []
hip_host_sources = []
hip_inc_dirs = []

if hip_enabled
  hip_runtime_include_dirs += fs.parent(fs.parent(hipcc.full_path())) / 'include'
  foreach rocm_inc : hip_runtime_include_dirs
    if fs.is_file(rocm_inc / 'hip' / 'hip_runtime.h')
      hip_inc_dirs += include_directories(rocm_inc, is_system : true)
      if not torch_inc_flags.contains('-I' + rocm_inc)
        torch_inc_flags += '-I' + rocm_inc
      endif
    endif
  endforeach
  if not cpp.has_header('hip/hip_runtime.h', args : hip_defines + torch_inc_flags)
    error('the HIP lane cannot find <hip/hip_runtime.h>. Put a ROCm install\'s bin/ on PATH so hipcc resolves under its own prefix.')
  endif

  hip_arch_mode = get_option('orihime_hip_arch_mode')
  hip_arch_targets = []
  if hip_arch_mode == 'native'
    hip_native_arch_result = run_command(python, '-c',
      '''
import torch

if not torch.cuda.is_available():
    raise SystemExit("no AMD GPU is visible; select custom mode and pass -Dorihime_hip_targets=gfx... for a cross-build")

properties = torch.cuda.get_device_properties(torch.cuda.current_device())
arch = getattr(properties, "gcnArchName", "").split(":", 1)[0]
if not arch.startswith("gfx"):
    raise SystemExit(f"could not determine the active AMD GPU target from {properties!r}")
print(arch)
''',
      check : false
    )
    if hip_native_arch_result.returncode() != 0
      error('orihime_hip_arch_mode=native could not detect an AMD GPU: ' + hip_native_arch_result.stderr().strip())
    endif
    hip_arch_targets += hip_native_arch_result.stdout().strip()
  elif hip_arch_mode == 'release-rocm7-rdna'
    # One generic code object per supported RDNA subarchitecture family. LLVM's
    # generic processors allow each object to load across every covered member:
    # RDNA2 gfx103x, RDNA3/3.5 gfx110x/gfx115x, and RDNA4 gfx120x.
    hip_arch_targets += [
      'gfx10-3-generic',
      'gfx11-generic',
      'gfx12-generic',
    ]
  else
    custom_hip_targets = get_option('orihime_hip_targets')
    foreach arch : custom_hip_targets.split(';')
      if arch.strip() != ''
        hip_arch_targets += arch.strip()
      endif
    endforeach
  endif

  hip_offload_args = []
  foreach arch : hip_arch_targets
    if arch.strip() != ''
      hip_offload_args += '--offload-arch=' + arch.strip()
    endif
  endforeach
  if hip_offload_args.length() == 0
    error('orihime_hip_arch_mode=custom requires -Dorihime_hip_targets=gfx... with at least one AMD GPU target')
  endif
  message('Orihime HIP targets: ' + ';'.join(hip_arch_targets))

  hip_device_args = hip_offload_args + [
    '-std=c++17',
    '-O3',
    '-fPIC',
    '-DUSE_TORCH_LIBRARY',
    cxx_abi_flag,
  ] + hip_defines + ['-I' + meson.current_source_dir() / 'src'] + torch_inc_flags

  foreach op : hip_ops
    hip_objects += custom_target('orihime-hip-@0@-kernels'.format(op),
      input : files('src' / op / 'kernels_gpu.hip'),
      output : 'orihime_hip_@0@_kernels.o'.format(op),
      depfile : 'orihime_hip_@0@_kernels.d'.format(op),
      command : [hipcc] + hip_device_args + [
        '-MD', '-MF', '@DEPFILE@',
        '-c', '@INPUT@',
        '-o', '@OUTPUT@',
      ],
    )
    hip_host_sources += files('src' / op / 'torch_hip.cpp')
  endforeach
endif

# Source files - Soft Smith-Waterman (regular + affine), Soft CKY, Soft DTW, Soft NW, Soft MAS, Soft Eisner, Soft Levenshtein, Soft LCS, Soft OSA, and Soft Damerau
cpp_sources = files(
  # CKY module (fully self-contained)
  'src/cky/registry.cpp',
  'src/cky/torch_cpu.cpp',
  'src/cky/kernels_cpu.cpp',
  # SW module (fully self-contained)
  'src/sw/registry.cpp',
  'src/sw/torch_cpu.cpp',
  # SW Affine module (fully self-contained)
  'src/sw_affine/registry.cpp',
  'src/sw_affine/torch_cpu.cpp',
  'src/sv_affine/registry.cpp',
  'src/sv_affine/torch_cpu.cpp',
  'src/sv_linear/registry.cpp',
  'src/sv_linear/torch_cpu.cpp',
  'src/sw/kernels_cpu.cpp',
  'src/sw_affine/kernels_cpu.cpp',
  'src/sv_affine/kernels_cpu.cpp',
  'src/sv_linear/kernels_cpu.cpp',
  # NW module (fully self-contained)
  'src/nw/registry.cpp',
  'src/nw/torch_cpu.cpp',
  'src/nw/kernels_cpu.cpp',
  # NW Affine module (fully self-contained)
  'src/nw_affine/registry.cpp',
  'src/nw_affine/torch_cpu.cpp',
  'src/nw_affine/kernels_cpu.cpp',
  # DTW module (fully self-contained)
  'src/dtw/registry.cpp',
  'src/dtw/torch_cpu.cpp',
  'src/dtw/kernels_cpu.cpp',
  # Levenshtein module (fully self-contained)
  'src/lev/registry.cpp',
  'src/lev/torch_cpu.cpp',
  'src/lev/kernels_cpu.cpp',
  # LCS module (fully self-contained)
  'src/lcs/registry.cpp',
  'src/lcs/torch_cpu.cpp',
  'src/lcs/kernels_cpu.cpp',
  # OSA module (fully self-contained)
  'src/osa/registry.cpp',
  'src/osa/torch_cpu.cpp',
  'src/osa/kernels_cpu.cpp',
  # Damerau module (fully self-contained)
  'src/damerau/registry.cpp',
  'src/damerau/torch_cpu.cpp',
  'src/damerau/kernels_cpu.cpp',
  # MAS module (fully self-contained)
  'src/mas/registry.cpp',
  'src/mas/torch_cpu.cpp',
  'src/mas/kernels_cpu.cpp',
  # Eisner module (fully self-contained)
  'src/eisner/registry.cpp',
  'src/eisner/torch_cpu.cpp',
  'src/eisner/kernels_cpu.cpp',
)

extension_sources = cpp_sources

if cuda_enabled
  extension_sources += files(
    'src/cky/torch_cuda.cpp',
    'src/sw/torch_cuda.cpp',
    'src/sw_affine/torch_cuda.cpp',
    'src/sv_affine/torch_cuda.cpp',
    'src/sv_linear/torch_cuda.cpp',
    'src/nw/torch_cuda.cpp',
    'src/nw_affine/torch_cuda.cpp',
    'src/dtw/torch_cuda.cpp',
    'src/lev/torch_cuda.cpp',
    'src/lcs/torch_cuda.cpp',
    'src/osa/torch_cuda.cpp',
    'src/damerau/torch_cuda.cpp',
    'src/mas/torch_cuda.cpp',
    'src/eisner/torch_cuda.cpp',
    'src/cky/kernels_gpu.cu',
    'src/sw/kernels_gpu.cu',
    'src/sw_affine/kernels_gpu.cu',
    'src/sv_affine/kernels_gpu.cu',
    'src/sv_linear/kernels_gpu.cu',
    'src/nw/kernels_gpu.cu',
    'src/nw_affine/kernels_gpu.cu',
    'src/dtw/kernels_gpu.cu',
    'src/lev/kernels_gpu.cu',
    'src/lcs/kernels_gpu.cu',
    'src/osa/kernels_gpu.cu',
    'src/damerau/kernels_gpu.cu',
    'src/mas/kernels_gpu.cu',
    'src/eisner/kernels_gpu.cu',
  )
endif

if hip_enabled
  extension_sources += hip_host_sources
endif

# Build Python extension module
if cuda_enabled
  python.extension_module('_C',
    extension_sources,
    include_directories : [src_inc] + torch_inc_dirs,
    dependencies : [python_dep] + torch_deps,
    cpp_args : cxx_args,
    cuda_args : cuda_args,
    link_args : openmp_link_args,
    link_language : 'cpp',
    install : true,
    subdir : 'orihime',
  )
elif hip_enabled
  python.extension_module('_C',
    extension_sources,
    objects : hip_objects,
    include_directories : [src_inc] + torch_inc_dirs + hip_inc_dirs,
    dependencies : [python_dep] + torch_deps,
    cpp_args : cxx_args,
    link_args : openmp_link_args,
    link_language : 'cpp',
    install : true,
    subdir : 'orihime',
  )
else
  python.extension_module('_C',
    extension_sources,
    include_directories : [src_inc] + torch_inc_dirs,
    dependencies : [python_dep] + torch_deps,
    cpp_args : cxx_args,
    link_args : openmp_link_args,
    link_language : 'cpp',
    install : true,
    subdir : 'orihime',
  )
endif

# Install Python source files
python.install_sources(
  'orihime/__init__.py',
  'orihime/_build_contract.py',
  'orihime/_build_info.py',
  'orihime/_compile.py',
  'orihime/_ops.py',
  'orihime/_pt2_ops.py',
  'orihime/_pt2_utils.py',
  'orihime/_public_ops.py',
  'orihime/cky.py',
  'orihime/damerau.py',
  'orihime/dtw.py',
  'orihime/edit_distance.py',
  'orihime/eisner.py',
  'orihime/lcs.py',
  'orihime/mas.py',
  'orihime/nn.py',
  'orihime/nw.py',
  'orihime/operator.py',
  'orihime/osa.py',
  'orihime/py.typed',
  'orihime/sw.py',
  'orihime/sv.py',
  subdir : 'orihime',
)

# Install orihime.raw subpackage
python.install_sources(
  'orihime/raw/__init__.py',
  'orihime/raw/_base.py',
  'orihime/raw/cky.py',
  'orihime/raw/damerau.py',
  'orihime/raw/dtw.py',
  'orihime/raw/eisner.py',
  'orihime/raw/lcs.py',
  'orihime/raw/lev.py',
  'orihime/raw/mas.py',
  'orihime/raw/nw.py',
  'orihime/raw/nw_affine.py',
  'orihime/raw/osa.py',
  'orihime/raw/sw.py',
  'orihime/raw/sw_affine.py',
  'orihime/raw/sv_affine.py',
  'orihime/raw/sv_linear.py',
  subdir : 'orihime/raw',
)

# Install orihime.ops subpackage
python.install_sources(
  'orihime/ops/__init__.py',
  'orihime/ops/sw.py',
  'orihime/ops/sw_affine.py',
  'orihime/ops/sv_affine.py',
  'orihime/ops/sv_linear.py',
  'orihime/ops/nw.py',
  'orihime/ops/nw_affine.py',
  'orihime/ops/dtw.py',
  'orihime/ops/cky.py',
  'orihime/ops/eisner.py',
  'orihime/ops/mas.py',
  'orihime/ops/damerau.py',
  'orihime/ops/osa.py',
  'orihime/ops/lcs.py',
  'orihime/ops/lev.py',
  subdir : 'orihime/ops',
)
