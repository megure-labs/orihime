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
4. Credit contributors and relevant external materials and notices. State
   missing validation or unavailable hardware plainly.
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

<!-- BEGIN REPOSITORY-CONTRIBUTION-WORKFLOW-V1 -->
## Public contribution workflow

Use a focused fork branch and open a pull request into `main`. Give a normal
change/test summary and appropriate contributor/source credits. Preserve existing
license notices and project-specific source-use requirements. Kaname capture is
optional for public contributions; never claim capture that did not occur.

A company maintainer approves the exact submitted commit before it enters
privileged internal testing. For an explicitly paired repository, maintainers
port the approved patch into an isolated private development PR. Full human/agent
contribution records, product tests, Greptile and CodeRabbit reviews belong there.
The private record retains the original public PR, exact commit and author credit.

Accepted work returns through a curated release PR with a normal public summary,
validation and credits. Full internal disclosures and private history stay in the
private development record. Public/release repositories do not run Greptile,
CodeRabbit or the full internal disclosure validator. Maintainer approval, branch
protections, release checks and public source/license notices remain in force.

Unpaired repositories have no automatic import destination. Do not invent one.
Cross-repository routing remains inactive until its scoped credentials and hosted
end-to-end validation are complete; it never merges a release automatically.
<!-- END REPOSITORY-CONTRIBUTION-WORKFLOW-V1 -->
