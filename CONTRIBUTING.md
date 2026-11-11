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

<!-- BEGIN REPOSITORY-CONTRIBUTION-WORKFLOW-V1 -->
## Public contribution workflow

Public pull requests are welcome. Use a fork and a focused branch; target `main`.
Submit the required human, agent, source, and validation disclosure below. No
Kaname launcher, recording, transcript, enforcement marker, or provenance task is
required for public contributions. Preserve historical evidence without claiming
that uncaptured work was recorded. This does not change separately scoped private
development requirements.

Greptile and CodeRabbit do not review this public repository. Maintainer review
and applicable CI remain required. A maintainer-approved external PR may be
ported to its explicitly paired private dev repo, tested there, and returned
through a new release PR. The dev PR and release PR must remain linked to the
original public PR and its exact reviewed commit. Conflicts or different release
layouts require an explicit port; never replace a release tree with a dev tree.
Cross-repository automation must be configured and authenticated before it is
described as active. It may open PRs but may not merge release PRs automatically.

## Required contribution disclosure

Every pull request, including maintenance and release PRs, must state:

- **Human contribution:** who directed the work, wrote or changed code, reviewed
  it, made design decisions, and ran validation. Identify the accountable human.
- **Agent contribution:** each agent/tool and model/version when known; what it
  generated, changed, researched, reviewed, or executed; the scope of autonomy;
  and how the human checked its output. Say "None" if no agents were used.
- **External sources:** URLs and exact revisions/versions when available for
  copied, adapted, translated, generated-from, or consulted code; affected files;
  licenses/notices; dependencies, papers, documentation, models, and data. Explain
  what was reused versus independently implemented. Say "None" if applicable.
- **Validation:** exact commands, results, platforms, and meaningful gaps. Do not
  invent test results, captures, authorship, reviews, citations, or permissions.

Do not publish credentials, private chat transcripts, hidden reasoning, customer
data, or private repository details. A clear factual account is required; raw
agent session logs are not required. Preserve existing licenses and notices.
Repository-specific source-use and clean-room requirements still apply.

Only a maintainer can grant organization write access. Outside contributors use
forks and pull requests and must obtain approval before their changes enter an
organization branch or privileged workflow. Approval is tied to the exact head
commit; subsequent changes require approval again. Never push directly to main,
force-push protected branches, or treat a bot review as human approval.
<!-- END REPOSITORY-CONTRIBUTION-WORKFLOW-V1 -->
