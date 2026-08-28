# SPDX-License-Identifier: Apache-2.0
"""Execute every end-user Python snippet in the public documentation."""

from __future__ import annotations

import inspect
import json
import re
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import torch

import orihime


PUBLIC_DOCUMENTS = (
    Path("README.md"),
    Path("docs/concepts.md"),
    Path("docs/examples.md"),
    Path("docs/faq.md"),
    Path("docs/usage.md"),
    Path("docs/algorithms/cky.md"),
    Path("docs/algorithms/dtw.md"),
    Path("docs/algorithms/edit-distance.md"),
    Path("docs/algorithms/eisner.md"),
    Path("docs/algorithms/mas.md"),
    Path("docs/algorithms/nw.md"),
    Path("docs/algorithms/sw.md"),
    Path("docs/algorithms/sv.md"),
)
PUBLIC_ALGORITHMS = (
    "sw",
    "sw_affine",
    "sv",
    "sv_affine",
    "nw",
    "nw_affine",
    "dtw",
    "lcs",
    "lev",
    "osa",
    "damerau",
    "mas",
    "cky",
    "eisner",
)
ALGORITHM_GUIDES = (
    Path("docs/algorithms/sw.md"),
    Path("docs/algorithms/sv.md"),
    Path("docs/algorithms/nw.md"),
    Path("docs/algorithms/dtw.md"),
    Path("docs/algorithms/cky.md"),
    Path("docs/algorithms/mas.md"),
    Path("docs/algorithms/eisner.md"),
    Path("docs/algorithms/edit-distance.md"),
)
SOURCE_READMES = tuple(
    Path("src") / algorithm / "README.md" for algorithm in (
        "cky",
        "damerau",
        "dtw",
        "eisner",
        "lcs",
        "lev",
        "mas",
        "nw",
        "nw_affine",
        "osa",
        "sv_affine",
        "sv_linear",
        "sw",
        "sw_affine",
    )
)
DOCUMENTATION_SURFACES = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("CHANGELOG.md"),
    Path("CONTRIBUTING.md"),
    Path("PROVENANCE.md"),
    Path("SECURITY.md"),
    Path("docs/compatibility.md"),
    Path("docs/concepts.md"),
    Path("docs/examples.md"),
    Path("docs/faq.md"),
    Path("docs/performance.md"),
    Path("docs/provenance-policy.md"),
    Path("docs/source-build.md"),
    Path("docs/testing.md"),
    Path("docs/usage.md"),
    Path("src/ARCHITECTURE.md"),
    Path("docs/algorithms/README.md"),
    *ALGORITHM_GUIDES,
    *SOURCE_READMES,
)
PYTHON_FENCE = re.compile(
    r"^```python[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)

PUBLIC_FUNCTIONS = (
    "sw",
    "sw_value",
    "sw_entropy",
    "sw_affine",
    "sw_affine_value",
    "sw_affine_entropy",
    "sv",
    "sv_value",
    "sv_entropy",
    "sv_affine",
    "sv_affine_value",
    "sv_affine_entropy",
    "nw",
    "nw_value",
    "nw_entropy",
    "nw_affine",
    "nw_affine_value",
    "nw_affine_entropy",
    "dtw",
    "dtw_value",
    "dtw_entropy",
    "lcs",
    "lcs_value",
    "lcs_entropy",
    "lev",
    "lev_value",
    "lev_entropy",
    "osa",
    "osa_value",
    "osa_entropy",
    "damerau",
    "damerau_value",
    "damerau_entropy",
    "build_damerau_transposition_sources",
    "mas",
    "mas_value",
    "mas_entropy",
    "cky",
    "cky_leaf_map",
    "cky_value",
    "cky_entropy",
    "eisner",
    "eisner_value",
    "eisner_entropy",
)


def _source_root() -> Path:
    local_root = Path(__file__).resolve().parents[1]
    if (local_root / "README.md").is_file():
        return local_root

    distribution = metadata.distribution("orihime")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is not None:
        source_url = json.loads(direct_url_text).get("url", "")
        parsed = urlparse(source_url)
        if parsed.scheme == "file":
            installed_from = Path(unquote(parsed.path)).resolve()
            if (installed_from / "README.md").is_file():
                return installed_from

    raise AssertionError(
        "could not locate the source README/docs for snippet validation"
    )


def _github_anchor(heading: str) -> str:
    without_markup = re.sub(r"[`*_<>]", "", heading).strip().lower()
    without_punctuation = re.sub(r"[^\w\- ]", "", without_markup)
    return re.sub(r" +", "-", without_punctuation)


def _relative_links(source: str) -> tuple[tuple[str, str], ...]:
    links = []
    for match in MARKDOWN_LINK.finditer(source):
        target = match.group(1).strip().strip("<>")
        parsed = urlparse(target)
        if parsed.scheme or parsed.netloc:
            continue
        links.append((unquote(parsed.path), unquote(parsed.fragment)))
    return tuple(links)


@pytest.mark.parametrize(
    "relative_path",
    PUBLIC_DOCUMENTS,
    ids=lambda path: path.as_posix(),
)
def test_public_python_snippets_execute(relative_path: Path) -> None:
    source_path = _source_root() / relative_path
    source = source_path.read_text(encoding="utf-8")
    snippets = PYTHON_FENCE.findall(source)
    assert snippets, f"{relative_path} has no executable Python snippets"

    torch.manual_seed(9100 + PUBLIC_DOCUMENTS.index(relative_path))
    namespace = {
        "__file__": str(source_path),
        "__name__": "__public_docs__",
    }
    executable = "\n\n".join(snippets)
    exec(compile(executable, str(source_path), "exec"), namespace)


def test_documentation_vocabulary_matches_the_public_api() -> None:
    source_root = _source_root()
    root_readme = (source_root / "README.md").read_text(encoding="utf-8")

    assert tuple(orihime.ops.__all__) == PUBLIC_ALGORITHMS
    assert "import orihime as ohm" in root_readme
    assert "sv_linear" not in root_readme
    for algorithm in PUBLIC_ALGORITHMS:
        assert f"`ohm.{algorithm}`" in root_readme

    for relative_path in DOCUMENTATION_SURFACES:
        source = (source_root / relative_path).read_text(encoding="utf-8")
        assert not re.search(r"import orihime as ori\b", source), relative_path
        assert not re.search(r"\bori\.", source), relative_path
        assert not re.search(r"\b(?:ori|ohm)\.soft_[a-z_]", source), relative_path


def test_every_native_operator_has_a_structured_readme() -> None:
    source_root = _source_root()
    native_directories = tuple(
        sorted(
            registry.parent.relative_to(source_root)
            for registry in (source_root / "src").glob("*/registry.cpp")
        )
    )

    assert native_directories == tuple(path.parent for path in SOURCE_READMES)
    required_sections = (
        "## Recurrence",
        "## State and memory layout",
        "## Native operations",
        "## Files and backends",
        "## See also",
    )
    report_language = re.compile(
        r"## (?:HIP )?Verification|\b(?:passed|skipped)\b|"
        r"\b(?:RTX|Radeon)\b|\bgfx\d+|\b20\d\d-\d\d-\d\d\b"
    )

    for relative_path in SOURCE_READMES:
        source = (source_root / relative_path).read_text(encoding="utf-8")
        positions = tuple(source.index(section) for section in required_sections)
        assert positions == tuple(sorted(positions)), relative_path
        assert source.splitlines()[0].endswith(" implementation"), relative_path
        assert report_language.search(source) is None, relative_path


@pytest.mark.parametrize(
    "relative_path",
    DOCUMENTATION_SURFACES,
    ids=lambda path: path.as_posix(),
)
def test_relative_documentation_links_resolve(relative_path: Path) -> None:
    source_root = _source_root()
    source_path = source_root / relative_path
    source = source_path.read_text(encoding="utf-8")

    for link_path, fragment in _relative_links(source):
        target_path = source_path if not link_path else source_path.parent / link_path
        target_path = target_path.resolve()
        assert target_path.exists(), f"{relative_path}: missing {link_path!r}"
        if not fragment or not target_path.is_file() or target_path.suffix != ".md":
            continue
        target_source = target_path.read_text(encoding="utf-8")
        anchors = {_github_anchor(heading) for heading in MARKDOWN_HEADING.findall(target_source)}
        assert fragment in anchors, (
            f"{relative_path}: missing anchor #{fragment} in "
            f"{target_path.relative_to(source_root)}"
        )


@pytest.mark.parametrize("function_name", PUBLIC_FUNCTIONS)
def test_public_function_docstrings_cover_the_signature(
    function_name: str,
) -> None:
    function = getattr(orihime, function_name)
    docstring = inspect.getdoc(function)
    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring
    assert "Raises:" in docstring
    for parameter_name in inspect.signature(function).parameters:
        assert f"{parameter_name}:" in docstring
