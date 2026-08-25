# Provenance

Orihime is a public release cut from Megure Labs' private development tree. Its
public Git history intentionally begins with one release commit; the private
development commit history and Kaname execution records are not included.

Megure Labs retains the complete, immutable append-only Kaname provenance
traces for this release, including task graphs, agent and model assignments,
review and adjudication records, commands, tool calls, execution metadata, and
the source-to-release mapping.

To request the full provenance traces, email Megure Labs at
`casey@megure.ai`. Access is reviewed case by case because traces may contain
security-sensitive infrastructure metadata and private development context.

The absence of the private traces from this repository does not alter the
Apache-2.0 license covering the released source. Release artifacts should be
verified against the signed or annotated Git tag and checksums published with
the applicable GitHub release.

The imported HIP backend has an additional file-by-file source and transformed
checksum record in [HIP_SOURCE_PROVENANCE.md](HIP_SOURCE_PROVENANCE.md).
