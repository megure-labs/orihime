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

## Scope: actor-neutral

Provenance, clean-room, licensing, and admission requirements attach to a
contribution, not to the kind of actor that produced it. Human-written and
agent-written changes are admitted under identical requirements. After
enforcement, both require a complete Kaname-compatible trace and an entry under
`.provenance/changes/`. Being human is not an exemption; human authors and
agents may carry different legal responsibilities, but the repository
admission boundary is the same. During bootstrap, a maintainer-authorized
change without a trace must disclose its actual authority, validation,
authorship, and material assistance and must not fabricate a trace.

Imperatives below state repository-work and admission conditions for everyone
preparing a change, including maintainers and tools acting on their behalf.

## Contribution admission

External upstream contributions are temporarily closed while Megure Labs builds
and deploys the production Kaname verifier. Forking and downstream modification
remain permitted by Apache-2.0, but an external person or agent must not submit
code, tests, documentation, data, models, generated artifacts, or patches for
upstream admission during this closure.

These admission rules apply equally to human-written and agent-written changes
and to repository policy itself.

Until the repository's base commit contains
`.provenance/KANAME_ENFORCEMENT_BASE`, a Megure Labs maintainer may explicitly
authorize and execute an internal bootstrap change without a Kaname trace. Do
not create a `.provenance/changes/` record or claim Kaname provenance for such a
change. Preserve the actual authority, validation, and authorship disclosures in
the pull request instead. External contributions remain closed during bootstrap.
This bootstrap rule applies equally to human-written and agent-written
maintainer changes. Maintainer authority permits the bootstrap change; it does
not exempt the change from disclosure, clean-room, licensing, or validation
requirements.

Once that marker is present in the base commit, repository changes may be
proposed upstream only when a Megure Labs maintainer initiates and executes the
work through a Megure-controlled Kaname workflow and retains its complete
Kaname-compatible history. Git history, a final diff, a public provenance
record, a retrospective summary, or a contributor's self-attestation is not
that history. Do not replay an externally produced patch and describe the
replay as provenance for its creation.

After enforcement, if the complete history is absent or cannot be verified,
stop and report that the change is not eligible for review or merge.

The complete-history requirement applies equally to human-written and
agent-written changes and to repository policy itself.
`docs/provenance-policy.md` defines the bootstrap boundary, required history,
and admission rules.

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
- After `.provenance/KANAME_ENFORCEMENT_BASE` is present in the base commit,
  every proposed change must have a complete Kaname-compatible trace and one
  new public record under `.provenance/changes/`, as specified by
  `docs/provenance-policy.md`. Before that boundary, maintainer-authorized
  bootstrap changes must disclose that no Kaname trace exists and must not
  fabricate one. Never invent run IDs, evidence, digests, review results, or
  closure state.
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
