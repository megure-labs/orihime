# Change provenance and merge policy

Every change merged into Orihime must have complete, reviewable provenance.
This requirement applies to code, tests, documentation, dependencies, build
configuration, automation, and repository policy. A change is not eligible for
approval merely because its tests pass.

## Required provenance

Each pull request must add exactly one immutable public change record at
`.provenance/changes/<change-id>.json`. The record follows
`.provenance/change-provenance.schema.json` and binds the proposed diff, other
than the record itself, to a complete Kaname-compatible trace retained by
Megure Labs.

The complete trace must contain:

- the original goal, decomposed task graph, dependencies, and input identities;
- every human, agent, service, provider, model, harness, and effort assignment;
- the machines and reproducibility manifest used for each attempt;
- commands, tool calls, model-call transcripts, outputs, and artifact digests;
- the exact source patch and environment identity;
- validation commands, environments, outputs, and budget outcome;
- independent review findings and a separate adjudication decision; and
- an immutable ledger root and closed-scope certificate.

Empty collections are represented by the digest of their canonical empty
form. A missing category may not be represented by an all-zero or placeholder
digest.

The public record is deliberately an index, not a transcript dump. It contains
the Kaname run and attempt identities, non-sensitive actor metadata, validation
summary, external-material disclosures, and SHA-256 commitments to the complete
trace. Private transcripts and infrastructure metadata remain in Kaname's
access-controlled append-only store and are disclosed under `PROVENANCE.md`.

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

Contributors who do not have access to Kaname may open a draft pull request
without a record. Before the pull request becomes merge-eligible, a maintainer
must import or replay the proposed change through Kaname, preserve the original
author's attribution, complete review and adjudication, and add the resulting
record. Merely copying the contributor's final diff into a record is not a
substitute for tracing the integration and validation work.

## Clean-room rule

Every external paper, standard, API document, dataset, model, fixture, or code
artifact consulted must appear in `external_materials`. Viewing relevant
third-party implementation source taints that implementation attempt. The
record is admissible only after a maintainer orders a clean-room restart, a new
independent implementer works without the tainted material, and the trace binds
the maintainer decision and restarted attempt. Compatible licensing and all
required notices remain independently mandatory.

## Approval gate

A maintainer may approve a pull request only when all of the following are
true:

1. The required Linux x86-64 and Apple Silicon CPU jobs pass.
2. The required `Provenance policy` job validates the public record and its
   patch binding.
3. The named Kaname scope is closed, its committed trace is retrievable, and
   the public digests match the retained ledger and trace bundle.
4. Independent review and adjudication are complete and accept the exact patch
   under review.
5. External-material, clean-room, licensing, and notice obligations are
   resolved.
6. GPU-affecting changes include passing offline NVIDIA and/or AMD validation
   evidence in the Kaname trace. GPU execution is intentionally not performed
   on public GitHub runners.
7. All GitHub review threads are resolved and a current code-owner approval is
   present.

Any push invalidates stale GitHub approval. Squash merge is the only permitted
merge method. A repository-administrator bypass is for repairing the ruleset
itself, not for merging unprovenanced content; any content change made through
an emergency bypass still requires a trace and public change record before the
next release.

## Enforcement boundary

GitHub can automatically prove that the public record is well formed,
non-placeholder, immutable, and cryptographically bound to the proposed diff.
The repository's trusted-base `pull_request_target` workflow performs that
check without checking out or executing untrusted pull-request code.

The final comparison against Kaname's private ledger is currently a code-owner
approval responsibility. Full machine enforcement requires a Kaname verifier
service or GitHub App that can read the private ledger and publish a required
commit status without exposing ledger credentials to pull-request workflows.
Until that verifier is deployed, no maintainer should interpret a green public
record check as proof that the referenced private bundle exists.

## Bootstrap

The commit that first installs this policy, schema, validator, workflow, and
branch requirement is the sole bootstrap exception to the per-pull-request
record rule: the trusted-base workflow cannot enforce a policy that is not yet
present on the base branch. This exception applies only to installing the gate
and creates no exception for subsequent content changes.
