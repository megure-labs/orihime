# Contributing to Orihime

Public contributions are welcome. Human-written and agent-assisted changes
follow the same requirements. Kaname capture and public Kaname change records
are optional; no private launcher or pre-implementation authorization is needed
to prepare a public pull request.

## Workflow

1. Use a focused branch and explain the problem and resulting behavior.
2. Follow the public API contracts and clean-room rules in `AGENTS.md`.
3. Run checks appropriate to the change and report commands and results.
   Native or GPU changes need applicable build and hardware validation.
4. Disclose relevant external materials and material coding-agent or AI
   assistance. State any missing validation or unavailable hardware plainly.
5. Open a pull request using the repository template. Maintainers review the
   change through the ordinary GitHub process.

Existing Git history, release checksums, notices, model/data provenance, and
optional historical trace records remain intact. Never fabricate capture or
retroactively describe an uncaptured change as Kaname-recorded. See
[`PROVENANCE.md`](PROVENANCE.md) and
[`docs/provenance-policy.md`](docs/provenance-policy.md).

## Licensing and clean-room work

By submitting a contribution, you represent that you have the right to submit
it under Apache-2.0. Preserve required copyright, license, attribution, and
provenance notices, including separate third-party artifact terms.

Implement from specifications, papers, mathematical definitions, documented
APIs, test vectors, and independently observed behavior. Do not copy, closely
paraphrase, translate, or reconstruct third-party implementation source. If
relevant external source has been viewed, disclose it before implementing so a
maintainer can decide whether an isolated clean-room restart is required.

Do not submit GPL, AGPL, noncommercial, source-available, or unknown-license
material. Approved compatible third-party material still needs its required
notices and source/license disclosures.

## Validation and security

Follow the validation commands in `AGENTS.md` and `docs/source-build.md`, scaled
to the affected behavior. Documentation-only changes do not require rebuilding
unchanged native kernels. Report vulnerabilities privately through
`SECURITY.md`.
