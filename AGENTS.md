# Agent instructions

These instructions apply to the entire repository.

## Project

Orihime is the public repository for the `orihime` distribution and Python
package. Public examples conventionally use `import orihime as ori`. Keep the
package and import name stable unless a maintainer explicitly requests an API
migration.

Coding agents are welcome to implement features, fix bugs, refactor, write
tests, review changes, and prepare pull requests. Make focused changes, retain
the public CPU/CUDA/HIP and PyTorch compatibility contracts, and update tests
and documentation with user-visible behavior.

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
- Every proposed change must have a complete Kaname-compatible trace and one
  new public record under `.provenance/changes/`, as specified by
  `docs/provenance-policy.md`. Do not invent run ids, evidence, digests, review
  results, or closure state. If Kaname evidence is unavailable, stop and report
  that the change is not merge-ready.
- Never commit credentials, private Kaname traces, internal task graphs,
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
Megure-controlled GPU validation in the Kaname trace. Run `git diff --check`
before submitting.
