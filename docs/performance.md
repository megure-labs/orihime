# Performance

Orihime's kernels parallelize across batch items on CPU and across the batch
and dynamic-program frontier on GPU. This page summarizes asymptotic cost and
the practical constraints that matter when choosing shapes and batch sizes.

## Complexity

Per batch element, the costs below use pairwise input lengths `L1` and `L2`,
parser length `N`, or MAS frame and token counts `T` and `S`.

| Function family | Time | Space |
| --- | --- | --- |
| `orihime.sw`, `orihime.sw_affine` | `O(L1·L2)` | `O(L1·L2)` |
| `orihime.nw`, `orihime.nw_affine` | `O(L1·L2)` | `O(L1·L2)` |
| `orihime.dtw` | `O(L1·L2)` | `O(L1·L2)` |
| `orihime.lev`, `orihime.lcs`, `orihime.osa`, `orihime.damerau` | `O(L1·L2)` | `O(L1·L2)` |
| `orihime.mas` | `O(T·S)` | `O(T·S)` |
| `orihime.cky` | `O(N³)` | `O(N³)` |
| `orihime.eisner` | `O(N³)` | `O(N²)` |
| `orihime.sv`, `orihime.sv_affine` | `O(L1·L2)` | `O(L1·L2)` |

Space includes DP working memory and the returned map. CKY returns
`[B, N, N, N]`, so its returned merge map uses `O(N³)` space per batch item.

Batch items run in parallel across CPU threads or GPU blocks. Up to the
available hardware parallelism, wall-clock time is dominated by the per-item
cost above rather than scaling linearly with `B`.

## CPU execution

- Each batch item's dynamic program runs on its own thread through PyTorch's
  `at::parallel_for`. Throughput scales with the intra-op thread count
  (`torch.get_num_threads()` or `OMP_NUM_THREADS`).
- A single sequence (`B = 1`) runs single-threaded. CPU parallelism is across
  the batch, not within one dynamic program.
- Inside a DataLoader worker, or with the thread count set to one, the operators
  run serially.

## GPU execution

- CUDA device code covers NVIDIA architectures from **Turing (`sm_75`)**
  through **Blackwell (`sm_121`)**. A release build includes several targets;
  a native build targets the attached GPU. See
  [source-build.md](source-build.md).
- HIP device code covers all 14 public algorithms and can target generic
  RDNA2, RDNA3/RDNA3.5, and RDNA4 code objects in one ROCm 7 build.
- GPU wrappers validate tensor ownership and select the input tensor's device,
  so multi-GPU dispatch does not silently launch on the wrong device.

## Numerical stability and safety

- The dynamic programs run in **FP32** for stability across smoothed maximum and
  minimum operations. Named primitives register autocast promotion to FP32.
- Each operator validates lengths, shapes, and devices before native dispatch.
  Kernels use `size_t` indexing where needed and avoid unsafe shared-memory
  reductions.

## Tips

- Batch inputs when possible; the batch dimension is the main source of CPU and
  GPU parallelism.
- Use `orihime.dtw(..., bandwidth=w)` to prune transitions and rule out
  implausible warps. Memory use remains `O(L1·L2)`.
- Choose temperature so every finite score or cost remains inside the
  [supported numerical domain](usage.md#numerical-domain-and-masks).
- `orihime.cky` and `orihime.eisner` are cubic in sequence length. Keep `N`
  modest or budget accordingly.

## See also

[Usage guide](usage.md) · [Algorithm guides](algorithms/) ·
[Source builds](source-build.md)
