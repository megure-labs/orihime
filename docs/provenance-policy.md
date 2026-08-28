# Change provenance and merge policy

Applies to all contributions regardless of who authored them (human or agent).

After the Kaname enforcement marker described below enters the base branch,
every change merged into Orihime must have complete, reviewable provenance.
This applies to code, tests, documentation, dependencies, build configuration,
automation, and repository policy. Passing tests alone does not make a change
eligible for approval.

## Current admission state: internal bootstrap only

External contributions are temporarily closed until Megure Labs deploys and
validates the production Kaname verifier as a required GitHub check. During the
pre-verifier bootstrap period, only changes explicitly authorized and executed
by a Megure Labs maintainer may be proposed or merged.

A bootstrap change must state its actual maintainer authority, validation, and
material human or agent assistance. It must not add a public change record or
claim that Kaname observed work it did not observe. This temporary exception
applies only to work authorized during bootstrap. It neither reconstructs
provenance nor admits an external patch.

An external final diff cannot be made admissible by replaying it, writing a
summary after the fact, or adding a public record. A maintainer may independently
implement an externally reported idea under explicit bootstrap authority or,
after enforcement, in a new Kaname scope. The maintainer must not import the
external patch or claim provenance for work Kaname did not observe.

## Enforcement boundary

The final bootstrap commit will add the tracked marker
`.provenance/KANAME_ENFORCEMENT_BASE`. The marker is deliberately absent while
bootstrap remains open. Because pull-request checks evaluate policy from the
trusted base, the commit that first adds the marker is itself the last bootstrap
commit; every pull request whose base contains the marker is subject to the
strict requirements below.

The marker must be added only when the production Kaname verifier and its
negative controls are ready to be required. It must identify the activation
commit's parent and document the verifier/check identity. Removing or bypassing
the marker after activation does not reopen bootstrap. Reopening or changing
the boundary requires an explicit, independently reviewed policy change; it may
not be inferred from a missing check or administrator bypass.

The trusted-base workflow always rejects external pull requests. Before the
marker exists in the base, it reports bootstrap mode and does not demand a
fictional change record. After the marker exists, it runs the structural record
validator; the production verifier remains a separate required check.

## Required provenance after enforcement

Each post-enforcement pull request must add exactly one immutable public change
record at `.provenance/changes/<change-id>.json`. The record follows
`.provenance/change-provenance.schema.json` and binds the proposed diff, other
than the record itself, to a complete Kaname-compatible trace retained by
Megure Labs.

The complete history must contain:

- the original human authority, goal, decomposed task graph, dependencies,
  constraints, input identities, and scope changes;
- every run's role, agent, provider, requested and reported model, effort,
  harness and launcher identity/version, tool binary digest, host, user,
  workspace, packet, session, and start/end times;
- the exact repository and worktree identity, starting commit, starting and
  ending dirty-state manifests, produced commits, and canonical candidate diff;
- the exact prompt bytes, provider-native event stream, provider stdout and
  stderr as separate byte-preserved artifacts, final response, exit record,
  lifecycle state stream, and capture-completeness/failure accounting;
- commands, tool calls and results, file/process/network observations required
  by the active evidence profile, environment commitments, outputs, and every
  intermediate and final artifact digest;
- validation commands, environments, raw outputs, expected and actual results,
  resource/usage records, and budget outcome;
- all external materials, license and clean-room decisions, custody boundaries,
  transformations, and derived-artifact lineage;
- independent review findings over the exact candidate, a separate adjudication
  decision, human resolution or ratification where required, and terminal scope
  closure; and
- one canonical append-only parent manifest that binds the complete evidence
  graph, its custody receipts, ordered event chain, signatures, and any timestamp
  or transparency anchors required by the packet, clean-room, or release profile.

Provider-native telemetry is evidence of the provider session; it does not
replace Kaname's authoritative operating-system event wall when the active
profile requires one. Raw evidence is never rewritten into a normalized summary
and then labeled as the original record. Unknown schemas, incomplete capture,
unresolved tool calls, missing graph ancestors, and identity or digest mismatches
fail closed.

Empty collections are represented by the digest of their canonical empty
form. A missing category may not be represented by an all-zero or placeholder
digest.

The public record is an index into the retained trace. It contains the Kaname
run and attempt identities, non-sensitive actor metadata, validation summary,
external-material disclosures, and SHA-256 commitments. Private transcripts
and infrastructure metadata remain in Kaname's access-controlled append-only
store and are disclosed under `PROVENANCE.md`.

## Change binding

The record's `change.patch_digest` is the SHA-256 digest of Git's canonical
binary diff from the pull request's exact base commit to its head, excluding
`.provenance/changes/`. Excluding the record avoids an impossible self-hash;
the exact base commit and every other path, mode, deletion, and binary change
remain bound.

Generate the expected value with:

```bash
python tools/validate_change_provenance.py digest \
  --base origin/main --head HEAD
```

Validate a completed record locally with:

```bash
python tools/validate_change_provenance.py check \
  --base origin/main --head HEAD
```

Existing records are append-only. Corrections require a new record in a new
pull request; an earlier record is never edited or deleted.

No external or post-enforcement untraced change may be opened as a draft for
later conversion. The public record and local validator can verify only a
completed history. They cannot create missing history or make a retrospectively
reconstructed change admissible.

## Clean-room rule

Every external paper, standard, API document, dataset, model, fixture, or code
artifact consulted must appear in `external_materials`. Viewing relevant
third-party implementation source taints that implementation attempt. The
record is admissible only after a maintainer orders a clean-room restart, a new
independent implementer works without the tainted material, and the trace binds
the maintainer decision and restarted attempt. Compatible licensing and all
required notices remain independently mandatory.

## Post-enforcement approval gate

A maintainer may approve a pull request only when all of the following are
true:

1. The change was initiated by a Megure Labs maintainer in an authorized Kaname
   scope before implementation began, and its full history is sealed.
2. The required Linux x86-64 and Apple Silicon CPU jobs pass.
3. The required `Provenance policy` job validates the public record and its
   patch binding.
4. The named Kaname scope is closed, its committed history is retrievable, and
   the public digests match the retained ledger and trace bundle.
5. Independent review and adjudication are complete and accept the exact patch
   under review.
6. External-material, clean-room, licensing, and notice obligations are
   resolved.
7. GPU-affecting changes include passing offline NVIDIA and/or AMD validation
   evidence in the Kaname trace. GPU execution is intentionally not performed
   on public GitHub runners.
8. All GitHub review threads are resolved and a current code-owner approval is
   present.

Any push invalidates stale GitHub approval. Squash merge is the only permitted
merge method. A repository-administrator bypass is for repairing the ruleset
itself, not for merging unprovenanced content; any content change made through
an emergency bypass still requires a trace and public change record before the
next release.

## Verifier responsibilities

GitHub can automatically verify that the public record is well formed,
non-placeholder, immutable, and cryptographically bound to the proposed diff.
The repository's trusted-base `pull_request_target` workflow performs that
check without checking out or executing untrusted pull-request code.

The structural validator checks only the public record. It cannot establish
that the private history exists, that capture was complete, that the full
ancestor graph closes, or that the evidence accepts the exact candidate.
During the temporary closure, the trusted-base workflow also rejects pull
requests whose author is not a Megure Labs organization member.

The production verifier will run as a separately trusted service or GitHub App.
It must read the private ledger, verify the canonical evidence graph and
custody receipts, and replay every binding for the exact candidate and every
newly reachable commit or merge parent. It must reject stale or reconstructed
evidence and publish an app-owned required status without exposing ledger
credentials to pull-request workflows. External contributions remain closed
until that gate and its negative controls have been validated and the
repository policy is explicitly reopened.

## Bootstrap record

The initial gate-installation and internal-only policy commits were made during
bootstrap because the production verifier was not yet active. They did not
establish the final Kaname enforcement boundary. Until the marker described
above is merged, Megure-maintainer-authorized bootstrap changes may include
product code, tests, documentation, build inputs, data, models, and release
artifacts when their real authority and validation are disclosed.

Bootstrap history is never rewritten or relabeled as Kaname history. No fake
run ID, evidence bundle, digest, review result, or public change record may be
created for it. The marker-adding commit closes this exception permanently and
all later changes must satisfy the post-enforcement requirements.
