# Owner-approved contribution routing

Status: prepared but inactive until the dedicated GitHub App is configured.
There is no shared personal access token, hidden local daemon, or automatic merge.

1. An outside contributor forks the public repository and opens a PR into main,
   giving a normal change/test summary and appropriate source/contributor credits.
2. The owner reviews and approves the exact public head commit, then applies
   `route-to-dev`. Only `caseysm` can trigger the privileged import. New commits
   invalidate approval and require a new approval and label event.
3. The router copies the public change into an explicitly paired private dev
   branch and opens a PR into main. It preserves the source URL, SHA and author.
   It never executes source code while handling the privileged event. Product
   Full human/agent/source/validation disclosures belong in this private PR.
   The generated agent section deliberately remains TODO until a maintainer
   records the actual contribution facts; the disclosure check blocks merging
   incomplete records. Public submission text is quoted as untrusted reference.
   Product CI and exact-head company-maintainer approval happen on the private
   PR; neither is replaced by a bot.
4. Once that dev PR is approved and merged, the router checks the exact approved
   head's real product CI and every required status check. A new public release
   PR is opened using only the original public commit/tree. It includes a normal
   summary, validation, public source URL/SHA and contributor credits; full private
   disclosures are never copied out. Public CI and exact-head owner approval
   still apply. Nothing merges automatically. The original public PR
   remains open as the intake/source record; close it after its release lands.

The importer requires matching public/private base blobs for every touched file.
Conflicts, renames, symlinks, submodules, files over 1 MiB, more than 100 files,
or changes to automation/agent/policy paths require a manual maintainer port.
If dev review changes the public patch, the owner must update the public PR and
approve/validate that new SHA. No private source or private PR URL is published.
Concurrent public-base changes also require revalidation. Idempotent branch names
avoid creating duplicate PRs for the same source SHA.

## Activation

Use a dedicated GitHub App installed only on the explicit public/dev pairs.
Required repository permissions: Contents write, Pull requests write, Commit
statuses write, Checks read, and implicit Metadata read. It needs no repository
administration, Actions write, Workflows write, organization membership access,
or branch-protection bypass. Keep it out of review-vendor app installations.

Set organization/repository variables `CONTRIBUTION_ROUTER_APP_ID` and
`CONTRIBUTION_ROUTER_APP_SLUG`, and the encrypted Actions secret
`CONTRIBUTION_ROUTER_APP_PRIVATE_KEY`. Grant the secret only to the paired repos.
The workflow mints a short-lived installation token restricted to the two repos
and the action revokes it after the run. Never commit or print the private key.

Both repos need branch protection, exact-head company-maintainer approval and
real product CI. Only the private dev PR requires full disclosure validation and
Greptile/CodeRabbit reviews; neither review vendor runs on the public release.
Any repository-specific admission/capture requirements must also be satisfied
before activation. Do not mark the integration active until
an owner-approved harmless fixture has completed import, dev CI, and release-PR
creation in the real hosted workflow. No live fixture has been executed yet.

Public mirrors without an explicitly approved dev pair are excluded. Do not
guess a destination, import an upstream fork into an unrelated product, publish
development history, or bypass existing curated release/export requirements.
