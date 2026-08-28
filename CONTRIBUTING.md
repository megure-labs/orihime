# Contributing to Orihime

The requirements in this file attach to each proposed change, not to whether
its author is human or an agent. Human contributors, maintainers, and coding
agents are subject to the same clean-room, licensing, provenance, validation,
and admission rules; no actor is exempt. Authority determines who may initiate
or approve work, but it does not waive requirements for the work itself.

## External contributions are temporarily closed

Megure Labs is not accepting external upstream contributions while the
production Kaname verifier is being built and deployed. Please do not open a
pull request or send a patch by issue, email, or another channel. This closure
applies to code, tests, documentation, dependencies, automation, data, models,
and generated artifacts, regardless of whether humans, coding agents, or both
produced them.

You may still fork and modify Orihime under Apache-2.0. Security vulnerabilities
should continue to be reported privately as described in `SECURITY.md`. A
maintainer may independently act on an externally reported idea, but the
external patch may not be imported or retrospectively described as having a
complete history. After Kaname enforcement is activated, the independent
implementation must begin inside Megure's Kaname workflow.

External contributions will reopen only after the Kaname verifier is deployed
as a required, trusted GitHub check and the maintainers publish an updated
policy here.

## Maintainer development workflow during bootstrap

Bootstrap is active while the pull request's base commit does not contain
`.provenance/KANAME_ENFORCEMENT_BASE`.

Bootstrap is not an exemption for maintainers or human authors. It means only
that the enforcement trace does not yet exist: every authorized bootstrap
change must disclose its actual authority, validation, authorship, external
materials, and material human or agent assistance, and must not fabricate a
trace.

1. Obtain explicit authorization from a Megure Labs maintainer before editing.
2. Work on a focused branch and include tests for changed behavior.
3. Follow the public contracts in the README and `docs/`.
4. Run the relevant test subset, then the full suite when practical.
5. Disclose the actual maintainer authority, validation, external materials,
   and material human or agent assistance in the pull request.
6. Do not add a `.provenance/changes/` record or claim Kaname history when no
   Kaname trace exists.

## Maintainer development workflow after enforcement

After enforcement, every change, whether human-written or agent-written, must
have a complete Kaname-compatible trace and an entry under
`.provenance/changes/`.

1. Initiate the change as an authorized Kaname goal or packet before editing.
2. Work on a focused branch and include tests for changed behavior.
3. Follow the public contracts in the README and `docs/`.
4. Run the relevant test subset, then the full suite when practical.
5. Seal the complete Kaname-compatible history and add its public change record
   at `.provenance/changes/<change-id>.json`.
6. Validate the record against the exact pull-request base with
   `python tools/validate_change_provenance.py check --base origin/main --head HEAD`.
7. Open a pull request using the repository template.

For a CPU source build:

```bash
python -m pip install meson-python meson ninja "pytest>=7"
python -m pip install --no-build-isolation --no-deps -e . \
  --config-settings=setup-args=-Dcuda=disabled
python -m pytest -q
```

See `docs/source-build.md` for CUDA, HIP, architecture-selection, and toolchain
details.

Public CI builds and runs the CPU-only extension on Linux x86-64 and Apple
Silicon. CUDA and HIP validation is performed offline on Megure-controlled
hardware; GPU-affecting changes must include those results in their retained
trace and public validation summary.

## Clean-room and AI-assisted work

The clean-room rule applies to the contribution regardless of whether a
maintainer writes it directly or uses coding agents.

Maintainer-controlled coding agents may implement, edit, test, and review work
during an explicitly authorized bootstrap change or, after enforcement, inside
an authorized Kaname run. All such work must follow `AGENTS.md`, including its
clean-room rule. In particular, do not base a change on third-party
implementation source.
Public papers, specifications, standards, API documentation, mathematical
definitions, test vectors, and black-box behavior are acceptable references.

In the pull request, disclose:

- which coding agents or AI tools materially contributed;
- every external paper, specification, API document, dataset, model, fixture,
  or code artifact consulted; and
- whether any external implementation source was viewed.

If external implementation source was viewed, say so before submitting code so
the maintainers can decide whether a clean-room restart is needed.

The complete history and approval requirements are documented in
[`docs/provenance-policy.md`](docs/provenance-policy.md). A public change record
is only an index into that history; it cannot make an external or
retrospectively reconstructed change eligible to merge.

## License

When external contributions reopen, submitting one will represent that you have
the right to submit it and license it to recipients under the Apache License
2.0. Unless explicitly stated otherwise by the maintainers, the project's
inbound and outbound license is Apache-2.0. Preserve all copyright, license,
attribution, and provenance notices.

Do not submit GPL, AGPL, noncommercial, source-available, or unknown-license
material. Do not add permissively licensed third-party code or data without
maintainer approval and the notices required by its license.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow `SECURITY.md`.
