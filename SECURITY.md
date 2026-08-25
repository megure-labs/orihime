# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected security or memory-safety
vulnerability. Use GitHub's private vulnerability-reporting flow on the
`megure-labs/orihime` repository. Include the affected version, platform, PyTorch
and CUDA versions, a minimal reproducer, and any sanitizer or crash output.

Do not include credentials, private data, or proprietary inputs in a report.
We will acknowledge a usable report, investigate it privately, and coordinate
disclosure and a fixed release when appropriate.

## Supported versions

Until a newer public release exists, `0.1.0` is the supported release line.
After future releases, only the latest patch release of each explicitly listed
supported line will receive security fixes.

## Scope

Reports involving native CPU/CUDA memory safety, malformed tensor handling,
artifact or update-channel integrity, dependency confusion, and unsafe binary
loading are in scope. General numerical-model quality, unsupported platform
combinations, and untrusted Python code execution are not security boundaries.
