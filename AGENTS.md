# Agent instructions

These instructions apply to the entire repository.

## Project

This is the public repository for the `orihime` distribution and Python
package. Public examples use `import orihime as ohm`. Keep the package and
import name stable unless a maintainer explicitly requests an API migration.

## Public documentation

- Describe Orihime as the re-release of the d²p library. Use d²p for the
  research contribution, paper, and original paper-aligned release.
- Call the unsuffixed top-level result a "structured attention map" when
  explaining what it represents. Keep `map` in API identifiers, field names,
  and mathematical terms such as map VJP and map derivative.
- Describe `value` as the soft Bellman value for the complete problem instance.
  Do not describe it as an expectation or as an unspecified scalar value.
- Describe `entropy` as the Shannon entropy, in nats, of the distribution over
  complete structures counted by the recurrence. State that Orihime computes
  it as `+dV/dT` for score-native operators and `-dV/dT` for cost-native
  operators; it is not elementwise entropy of the attention map.
- The 14 public algorithm names are `sw`, `sw_affine`, `sv`, `sv_affine`,
  `nw`, `nw_affine`, `dtw`, `lcs`, `lev`, `osa`, `damerau`, `mas`, `cky`, and
  `eisner`. Use `sv`, not the implementation name `sv_linear`, in public API
  documentation.
- Describe the public API as plain map/value/entropy functions, stateful
  modules under `ohm.nn`, and explicit `forward`, `backward`, and
  `sensitivity` operations under `ohm.ops`. Native dispatcher bindings and
  compatibility adapters are implementation details.
- Write public documentation like a scientific-computing library: direct,
  restrained, technically exact, and lightly pedagogical. Prefer runnable
  examples and precise rules to promotional adjectives.
- Keep dated hardware results, exact validation transcripts, private Kaname
  traces, and internal release narratives out of public README files.

## Contributions and optional Kaname capture

Public contributions are open. Human-written and agent-assisted changes follow
the same licensing, clean-room, attribution, validation, and maintainer review
requirements. Use focused branches and describe the change, its validation,
external materials, and material AI assistance in the pull request.

Kaname is optional for this public repository. A Kaname task, launcher,
transcript, complete-history proof, enforcement marker, or public change record
is not required to edit, test, submit, review, or merge a contribution. Do not
start capture solely because this repository is being edited. A separately
requested capture or a private development workspace retains its own explicit
scope and requirements.

Preserve existing provenance and never claim that uncaptured work was recorded.

## Clean-room implementation

- Do not retrieve, copy, closely paraphrase, translate, or reconstruct source
  code from third-party implementations. This includes source surfaced by web
  searches, training-memory recall, package caches, unrelated local checkouts,
  decompilers, or other agents.
- Implement from public specifications, standards, papers, mathematical
  definitions, API documentation, test vectors, and independently observed
  behavior. General programming knowledge and documented APIs are allowed.
- You may freely edit, refactor, and reuse code already in this repository.
- You may use code supplied by a maintainer only when the maintainer identifies
  it as Megure-owned or explicitly approves its compatible license and source.
- Do not introduce GPL, AGPL, noncommercial, source-available, or unknown-license
  material. Permissively licensed code still requires maintainer approval and
  preservation of all required notices.
- If you have already inspected an external implementation relevant to the
  requested work, stop and disclose the source before implementing. The
  maintainer will decide whether a separate clean-room implementation is
  required.

This policy governs importing external implementation material. It does not
prevent normal agent-assisted work on this repository or research using
papers, specifications, documentation, and test vectors.

## Licensing and provenance

- New project code is contributed under Apache-2.0. Do not remove or alter
  `LICENSE`, `NOTICE`, attribution, or provenance records.
- Record every approved third-party code, dataset, model, fixture, or generated
  artifact in the relevant notice and provenance files.
- Kaname records are optional. Preserve existing records and describe their
  actual scope; never invent run IDs, evidence, digests, review results, or
  closure state.
- Never commit credentials, private transcripts, internal task graphs,
  unrelated machine paths, or private checkout metadata.
- Do not rewrite public history or force-push unless a maintainer explicitly
  asks for it.

## Validation

For Python-only changes, run the narrow relevant tests. Before handing off a
release-facing change, run:

```bash
python -m compileall -q orihime tests
python -m pytest -q
```

For native changes, also build from source as documented in
`docs/source-build.md`. Public CI validates CPU-only Linux x86-64 and Apple
Silicon builds. CUDA and HIP execution is offline; record the applicable
GPU validation commands, hardware, and results in the pull request. Run
`git diff --check` before submitting.
