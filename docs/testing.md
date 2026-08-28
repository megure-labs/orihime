# Testing and coverage

Orihime's test suite exercises its fourteen public algorithms through the
top-level functions, explicit operations, modules, autograd, supported
`torch.func` transforms, `torch.compile`, input validation, masking, numerical
edge cases, and backend dispatch.

## Running the tests

Build Orihime from source before running the suite. From the repository root:

```bash
python -m compileall -q orihime tests
python -m pytest -q
```

Pytest skips tests whose hardware requirements are not met. In particular,
tests marked `multi_gpu` require at least two real CUDA devices. To run one
algorithm's tests while working on it, pass its test file directly:

```bash
python -m pytest -q tests/test_soft_sw_regular.py
```

## Public coverage snapshot

[`tests/operator_coverage_snapshot.json`](../tests/operator_coverage_snapshot.json)
records logical coverage for the fourteen public algorithms and 48 stable
scenario IDs. It is not a line-coverage report or a record of a particular
test run.

For each scenario, the snapshot assigns every algorithm exactly once to one
of five states: `covered`, `partial`, `missing`, `accepted-representative`, or
`not-applicable`. Entries outside `covered` include a reason when one is
needed. The test matrix checks the schema, algorithm list, scenario list, and
complete partitioning of every scenario.

The public snapshot contains no private evidence locations, workflow records,
or contributor identities. In a Megure-controlled checkout containing the
complete retained evidence, the same test also checks that the public snapshot
matches the evidence-derived projection.

## Continuous and hardware validation

Public GitHub Actions builds Orihime from source and runs the full test suite
on:

- Ubuntu 24.04 across Python 3.10 through 3.14 and PyTorch 2.9 through 2.13;
- Apple Silicon with Python 3.13 and PyTorch 2.12.

Those jobs are CPU-only. CUDA and HIP execution is validated separately on
Megure-controlled NVIDIA and AMD hardware, so a green public workflow must not
be treated as evidence that the GPU kernels ran. A separate CI job checks the
files included in the public release, its metadata, its licensing files, and
its repository policy.

## See also

- [Building from source](source-build.md)
- [Compatibility](compatibility.md)
- [Performance](performance.md)
- [Provenance policy](provenance-policy.md)
