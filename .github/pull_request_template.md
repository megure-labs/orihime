> [!IMPORTANT]
> External contributions are temporarily closed while Megure Labs deploys the
> production Kaname verifier. This template is for Megure Labs maintainers
> working under the bootstrap or post-enforcement policy. External pull
> requests are closed automatically; see `CONTRIBUTING.md`.

## Summary

Describe the change and why it is needed.

## Validation

List the commands and environments used to test it.

## Admission mode

Select exactly one:

- [ ] **Bootstrap:** the pull request's base does not contain
      `.provenance/KANAME_ENFORCEMENT_BASE`; a Megure Labs maintainer explicitly
      authorized this work, and no Kaname provenance is claimed.
- [ ] **Post-enforcement:** the base contains the marker and the complete Kaname
      provenance section below applies.

Bootstrap maintainer/authority:

## Kaname provenance (post-enforcement only)

Public change record: `.provenance/changes/<change-id>.json`

Kaname run ID:

- [ ] Work began in an authorized Megure-controlled Kaname scope; this is not a
      retrospective import or replay of an external patch.
- [ ] Exact prompt, provider streams, stderr, final response, lifecycle and exit
      evidence are byte-preserved and bound by the canonical parent manifest.
- [ ] Actor/model/tool/host/worktree/start-commit/produced-commit identities and
      all commands, tool calls, outputs, artifacts, and validations are bound.
- [ ] The Kaname scope is closed and the retained trace is retrievable.
- [ ] The public record binds this pull request's exact base and patch digest.
- [ ] Independent review and adjudication cover the current head.
- [ ] GPU-affecting changes include applicable offline NVIDIA/AMD evidence.

## Provenance and licensing

- [ ] I have the right to submit this contribution under Apache-2.0.
- [ ] I listed every external paper, specification, API document, dataset,
      model, fixture, or code artifact consulted below.
- [ ] I did not copy, closely paraphrase, translate, or reconstruct third-party
      implementation source.
- [ ] I disclosed any material use of coding agents or AI tools below.
- [ ] I preserved all required license, attribution, and provenance notices.
- [ ] I followed `docs/provenance-policy.md` and did not use placeholder ids,
      digests, results, attestations, or fabricated Kaname history.

External references:

AI and coding-agent assistance:
