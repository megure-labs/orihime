# Contributing to Orihime

We welcome contributions from people and coding agents. A human contributor is
responsible for every submitted change, including changes produced with an AI
tool.

## Development workflow

1. Open an issue first for a substantial API change or new algorithm.
2. Work on a focused branch and include tests for changed behavior.
3. Follow the public contracts in the README and `docs/`.
4. Run the relevant test subset, then the full suite when practical.
5. Open a pull request using the repository template.

For a CPU source build:

```bash
python -m pip install meson-python meson ninja "pytest>=7"
python -m pip install --no-build-isolation --no-deps -e . \
  --config-settings=setup-args=-Dcuda=disabled
python -m pytest -q
```

See `docs/source-build.md` for CUDA, HIP, architecture-selection, and toolchain
details.

## Clean-room and AI-assisted contributions

Coding agents may implement, edit, test, and review contributions. They must
follow `AGENTS.md`, including its clean-room rule. In particular, do not base a
contribution on third-party implementation source. Public papers,
specifications, standards, API documentation, mathematical definitions, test
vectors, and black-box behavior are acceptable references.

In the pull request, disclose:

- which coding agents or AI tools materially contributed;
- every external paper, specification, API document, dataset, model, fixture,
  or code artifact consulted; and
- whether any external implementation source was viewed.

If external implementation source was viewed, say so before submitting code so
the maintainers can decide whether a clean-room restart is needed.

## License

By submitting a contribution, you represent that you have the right to submit
it and license it to recipients under the Apache License 2.0. Unless explicitly
stated otherwise by the maintainers, the project's inbound and outbound
license is Apache-2.0. Preserve all copyright, license, attribution, and
provenance notices.

Do not submit GPL, AGPL, noncommercial, source-available, or unknown-license
material. Do not add permissively licensed third-party code or data without
maintainer approval and the notices required by its license.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow `SECURITY.md`.
