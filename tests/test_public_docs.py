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

import d2p


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
)
PYTHON_FENCE = re.compile(
    r"^```python[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

PUBLIC_FUNCTIONS = (
    "sw",
    "sw_value",
    "sw_entropy",
    "sw_affine",
    "sw_affine_value",
    "sw_affine_entropy",
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

    distribution = metadata.distribution("py-d2p")
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


@pytest.mark.parametrize("function_name", PUBLIC_FUNCTIONS)
def test_public_function_docstrings_cover_the_signature(
    function_name: str,
) -> None:
    function = getattr(d2p, function_name)
    docstring = inspect.getdoc(function)
    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring
    assert "Raises:" in docstring
    for parameter_name in inspect.signature(function).parameters:
        assert f"{parameter_name}:" in docstring
