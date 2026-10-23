# Public contribution and provenance policy

This policy applies to Orihime's public repository. Public pull requests are
open to human contributors and coding-agent-assisted contributors under the
same licensing, clean-room, validation, and maintainer review requirements.

## Optional Kaname capture

Kaname is optional. Editing, testing, submitting, reviewing, and merging public
changes do not require a Kaname workflow, transcript, complete-history proof,
private launcher, enforcement marker, or `.provenance/changes/` record.
The former internal-only bootstrap and mandatory post-enforcement admission
rules are retired for this public repository.

Ordinary Git commits and pull requests should explain the change, actual
validation, relevant external materials, material AI assistance, and known
limitations. Optional trace evidence may supplement those records; it does not
replace code review, tests, licensing, or clean-room obligations. Do not publish
private transcripts, credentials, or sensitive infrastructure details.

## Historical evidence

Preserve existing Git history, notices, release checksums, and provenance
records. Historical capture claims retain their original scope. Uncaptured
history is never rewritten or relabeled as Kaname history, and missing evidence
must not be reconstructed or fabricated.

The existing Kaname v1 schema and `tools/validate_change_provenance.py` remain
available for validating historical or voluntarily supplied records in that
format. They do not impose a contribution gate. Their strict validation rules
apply only when someone elects to supply such a record.

## Review and validation

Maintainers use ordinary GitHub review and applicable CI. Report actual checks,
platforms, and results; GPU-affecting changes need applicable offline hardware
validation. The public contribution workflow does not close pull requests based
on author organization membership and does not require a Kaname status check.

This policy does not enable capture or change separately scoped private
workspace requirements. Follow `CONTRIBUTING.md`, `AGENTS.md`, and the licensing
and artifact notices for this repository.
