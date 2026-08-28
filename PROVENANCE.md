# Provenance

Applies to all contributions regardless of who authored them (human or agent).

Orihime is a release cut from Megure Labs' private development tree, prepared
for public distribution. Its release Git history intentionally begins with one
commit; the private development commit history and Kaname execution records are
not included.

For release work performed through Kaname, Megure Labs retains complete,
immutable, append-only provenance traces. They include task graphs, agent and
model assignments, review and adjudication records, commands, tool calls,
execution metadata, and the source-to-release mapping.

The repository is currently in an explicitly authorized pre-verifier bootstrap
period. Bootstrap changes are not retrospectively relabeled as Kaname work and
do not receive fabricated trace records. The tracked enforcement marker and the
forward-only boundary are defined in
[Change provenance and merge policy](docs/provenance-policy.md).

To request the full provenance traces, email Megure Labs at
`casey@megure.ai`. Access is reviewed case by case because traces may contain
security-sensitive infrastructure metadata and private development context.

Keeping those traces private does not alter the Apache-2.0 license covering the
released source. Verify release artifacts against the signed or annotated Git
tag and the checksums published with the applicable GitHub release.

## Provenance after enforcement

Every pull request based on a commit containing
`.provenance/KANAME_ENFORCEMENT_BASE` must carry a content-bound public change
record and a complete retained Kaname-compatible trace. The required trace
contents, clean-room treatment, automated checks, and human approval gate are
normative in the change provenance and merge policy.

The public record commits to the private trace with SHA-256 digests; it does
not publish transcripts or sensitive machine metadata. GitHub validates the
record's structure and exact patch binding. A code owner separately verifies
the committed digests against Kaname's access-controlled ledger before
approval.
