# Grid benchmarks

`grid.py` measures the public structured attention map and soft Bellman value
functions on the production shape profiles used by the Megure Labs d²p kernel
tuning campaign. The catalog covers balanced, rectangular, odd, batched,
ragged, maximum-size, and structured-mask cases for all 14 algorithms.
Input distributions, deterministic seeds, family parameters, OSA transition
masks, and Damerau transition sources follow the d²p production baseline
runner. The adapter calls Orihime's public `value` and `map` functions
separately and expresses sentinel masks through the public boolean `mask=` API.

Run the full FP32 grid on one CUDA or HIP device:

```bash
python benchmarks/grid.py --output results.jsonl
```

Select algorithms, surfaces, or case identifiers with shell-style patterns:

```bash
python benchmarks/grid.py \
  --operators sw cky eisner \
  --surfaces map \
  --cases '*-b1-*' \
  --output selected.jsonl
```

Use `--resume` to skip successful rows already present in the output file.
Input construction, validation, warmup, and finiteness checks are outside the
timed region. Times are medians of five GPU-event measurements by default.
