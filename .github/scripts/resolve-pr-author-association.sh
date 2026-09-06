#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: resolve-pr-author-association.sh OWNER/REPO PULL_NUMBER" >&2
  exit 2
fi

repository="$1"
pull_number="$2"

if [[ ! "${repository}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid repository: ${repository}" >&2
  exit 2
fi

if [[ ! "${pull_number}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid pull-request number: ${pull_number}" >&2
  exit 2
fi

association="$({
  gh api --method GET \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2026-03-10" \
    "repos/${repository}/pulls/${pull_number}" \
    --jq '.author_association'
})"

# Validate before a caller writes this value to GITHUB_OUTPUT.
if [[ ! "${association}" =~ ^[A-Z_]+$ ]]; then
  echo "GitHub returned an invalid author association" >&2
  exit 1
fi

printf '%s\n' "${association}"
