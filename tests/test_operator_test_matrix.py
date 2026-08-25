# SPDX-License-Identifier: Apache-2.0
"""Executable drift gates for the shared twelve-operator test matrix."""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import orihime
import pytest

from orihime import ops
from operator_cases import OPERATOR_CASES, OPERATOR_NAMES
from operator_test_baseline import (
    EXPECTED_AUTOCAST_DTYPE_IDS,
    EXPECTED_CASES,
    OPERATOR_ORDER,
)


R73_SCENARIO_IDS = (
    "PUB-01",
    "PUB-02",
    "RAW-01",
    "NN-01",
    "AUT-01",
    "TRN-01",
    "TRN-02",
    "TRN-03",
    "TRN-04",
    "CMP-01",
    "CMP-02",
    "VAL-01",
    "VAL-02",
    "RAW-02",
    "VAL-03",
    "MASK-01",
    "MASK-02",
    "MASK-03",
    "ACC-01",
    "ACC-02",
    "ACC-03",
    "ACC-04",
    "ACC-05",
    "NUM-01",
    "NUM-02",
    "NUM-03",
    "NUM-04",
    "NUM-05",
    "LEN-01",
    "LEN-02",
    "EDGE-01",
    "GUARD-01",
    "GUARD-02",
    "THREAD-01",
    "BACK-01",
    "BACK-02",
    "OVER-01",
    "OVER-02",
    "PRUNE-01",
    "DEVICE-01",
    "DEVICE-02",
    "DEVICE-03",
    "FAM-01",
    "FAM-02",
    "FAM-03",
    "FAM-04",
    "FAM-05",
    "FAM-06",
)

STATUS_KEYS = (
    "covered",
    "partial",
    "missing",
    "accepted-representative",
    "not-applicable",
)
MAS_STATUS_KEYS = set(STATUS_KEYS)
OPERATOR_TEST_FILES = tuple(
    f"tests/test_soft_{name}.py"
    for name in (
        "sw_regular",
        "sw_affine",
        "nw",
        "nw_affine",
        "dtw",
        "lcs",
        "levenshtein",
        "osa",
        "damerau",
        "mas",
        "cky",
        "eisner",
    )
)
SNAPSHOT_PATH = Path(__file__).with_name("operator_coverage_snapshot.json")
TEST_LOCATOR_RE = re.compile(
    r"tests/[A-Za-z0-9_./-]+\.py::"
    r"[A-Za-z_][A-Za-z0-9_.-]*(?:::[A-Za-z_][A-Za-z0-9_.-]*)*"
)

# Independent pre-migration values.  This table is intentionally kept in the
# gate rather than the catalog so a catalog transcription error cannot make
# the adapter assertions self-consistent.
EXPECTED_SUITE_INDICES = {
    "sw": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "sw_affine": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "nw": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "nw_affine": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "dtw": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "lcs": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "lev": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "osa": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "damerau": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 2, 2),
    },
    "mas": {
        "off_optimal_index": (0, 1, 0),
        "padded_index": (0, 4, 0),
        "invariance_index": (0, 2, 1),
    },
    "cky": {
        "off_optimal_index": (0, 0, 1, 3),
        "padded_index": (0, 1, 2, 3),
        "invariance_index": (0, 0, 1, 3),
    },
    "eisner": {
        "off_optimal_index": (0, 0, 2),
        "padded_index": (0, 3, 0),
        "invariance_index": (0, 0, 2),
    },
}
KNOWN_CAPABILITIES = {
    "lengths",
    "affine",
    "bandwidth",
    "allowed-transpositions",
    "transposition-sources",
    "score-alignment",
    "cost-alignment",
    "edit-match",
    "edit-distance",
    "chart-parser",
    "projective",
    "leaf-scores",
    "monotonic-alignment",
}


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _control_root() -> Path:
    configured = os.environ.get("ORIHIME_CONTROL_REPO_ROOT", "").strip()
    if not configured:
        return _source_root()
    candidate = Path(configured).expanduser().resolve()
    if not candidate.is_dir() or not (candidate / "pyproject.toml").is_file():
        raise AssertionError(
            "ORIHIME_CONTROL_REPO_ROOT is configured but invalid: "
            f"{candidate}. Set ORIHIME_CONTROL_REPO_ROOT to a valid orihime control "
            "checkout containing pyproject.toml."
        )
    return candidate


def _evidence_root() -> Path:
    return _control_root()


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _status_operators(value: Any) -> set[str]:
    if isinstance(value, list):
        return set(value)
    if isinstance(value, dict):
        return set(value)
    raise AssertionError(f"status group must be a list or mapping, got {value!r}")


def _import_adapter(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _operator_imports_from_helper() -> tuple[str, ...]:
    source_path = Path(importlib.import_module("operator_test_utils").__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "orihime.ops":
            continue
        return tuple(alias.name for alias in node.names)
    raise AssertionError("operator_test_utils.py no longer imports orihime.ops")


def _decorator_is_multi_gpu(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "multi_gpu"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _decorator_is_shared_guard(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "TWO_CUDA_DEVICES_REQUIRED"


def _is_device_count_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "device_count"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
        and not node.args
        and not node.keywords
    )


def _is_two_device_comparison(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Lt)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 2
        and _is_device_count_call(node.left)
    )


def _is_pytest_skipif(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skipif"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "mark"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "pytest"
    )


def _shared_guard_assignment(tree: ast.Module) -> ast.Assign:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "TWO_CUDA_DEVICES_REQUIRED"
            for target in node.targets
        ):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or len(value.args) != 1:
            continue
        condition = value.args[0]
        is_skipif = _is_pytest_skipif(value)
        if is_skipif and _is_two_device_comparison(condition):
            return node
    raise AssertionError(
        "operator_test_prerequisites.TWO_CUDA_DEVICES_REQUIRED must be "
        "pytest.mark.skipif(torch.cuda.device_count() < 2, ...), with the "
        "fail-closed condition owned by the shared helper"
    )


def _source_tree(relative_path: str) -> ast.Module:
    path = _source_root() / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)


def _test_module_paths() -> tuple[str, ...]:
    tests_root = _source_root() / "tests"
    return tuple(
        str(path.relative_to(_source_root()))
        for path in sorted(tests_root.rglob("*.py"))
        if path.name != "operator_test_prerequisites.py"
    )


def _multi_gpu_nodes(tree: ast.Module) -> list[ast.AST]:
    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if any(_decorator_is_multi_gpu(decorator) for decorator in node.decorator_list):
            nodes.append(node)
    return nodes


def _shared_guard_nodes(tree: ast.Module) -> list[ast.AST]:
    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if any(_decorator_is_shared_guard(decorator) for decorator in node.decorator_list):
            nodes.append(node)
    return nodes


def _top_level_shared_guard_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "operator_test_prerequisites":
            continue
        if any(alias.name == "TWO_CUDA_DEVICES_REQUIRED" for alias in node.names):
            return True
    return False


def _inline_two_device_skipifs(tree: ast.Module) -> list[ast.Call]:
    matches = []
    for node in ast.walk(tree):
        if not _is_pytest_skipif(node):
            continue
        if node.args and any(
            _is_two_device_comparison(child) for child in ast.walk(node.args[0])
        ):
            matches.append(node)
    return matches


def _test_node_locators(relative_path: str) -> set[str]:
    tree = _source_tree(relative_path)
    locators: set[str] = set()

    def visit(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                path = prefix + (node.name,)
                locators.add(f"{relative_path}::{'::'.join(path)}")
                visit(node.body, path)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                path = prefix + (node.name,)
                locators.add(f"{relative_path}::{'::'.join(path)}")

    visit(tree.body, ())
    return locators


def _test_locators(value: str, context: str) -> list[str]:
    if "tests/" not in value:
        return []
    locators = TEST_LOCATOR_RE.findall(value)
    if not locators:
        raise AssertionError(
            f"{context} cites a test file without a stable "
            f"path::node_id locator: {value}"
        )
    return locators


FAMILY_LEDGER_SPECS = (
    ("logs/r74-linear-alignment-test-symmetry/coverage.json", {"sw", "nw"}),
    (
        "logs/r75-affine-alignment-test-symmetry/coverage.json",
        {"sw_affine", "nw_affine"},
    ),
    (
        "logs/r76-edit-distance-test-symmetry/coverage.json",
        {"lev", "osa", "damerau"},
    ),
    ("logs/r77-dtw-lcs-test-symmetry/coverage.json", {"dtw", "lcs"}),
    ("logs/r78-chart-parser-test-symmetry/coverage.json", {"cky", "eisner"}),
    ("logs/r79-mas-test-symmetry/coverage.json", {"mas"}),
)
FAMILY_STATUS_KEYS = set(STATUS_KEYS)
SCENARIO_EVIDENCE_LEDGER_PATHS = {
    relative_path for relative_path, _ in FAMILY_LEDGER_SPECS[:4]
}
MUTABLE_TEST_CITATION_RE = re.compile(
    r"tests/[A-Za-z0-9_./-]+\.py:[0-9]"
)
BARE_TEST_PATH_RE = re.compile(
    r"tests/[A-Za-z0-9_./-]+\.py(?!::)"
)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"unable to read {context} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssertionError(f"{context} at {path} must contain a JSON object")
    return value


def _resolve_test_locator(locator: str, context: str) -> None:
    relative_path = locator.split("::", 1)[0]
    if not relative_path.startswith("tests/"):
        raise AssertionError(f"{context} is not a tests/ locator: {locator}")
    path = _source_root() / relative_path
    if not path.is_file():
        raise AssertionError(f"{context} cites missing test file: {locator}")
    if locator not in _test_node_locators(relative_path):
        raise AssertionError(f"{context} cites an unresolved test node: {locator}")


def _stable_test_locators(value: str, context: str) -> list[str]:
    if "tests/" not in value:
        return []
    if MUTABLE_TEST_CITATION_RE.search(value):
        raise AssertionError(
            f"{context} contains a mutable test line citation: {value}"
        )
    if BARE_TEST_PATH_RE.search(value):
        raise AssertionError(
            f"{context} contains a bare test-file citation; use path::node_id: {value}"
        )
    locators = TEST_LOCATOR_RE.findall(value)
    if not locators:
        raise AssertionError(
            f"{context} cites a test file without a stable path::node_id locator: {value}"
        )
    for locator in locators:
        _resolve_test_locator(locator, context)
    return locators


def _load_family_ledgers(evidence_root: Path) -> list[dict[str, Any]]:
    ledgers: list[dict[str, Any]] = []
    seen_operators: set[str] = set()
    for relative_path, expected_operators in FAMILY_LEDGER_SPECS:
        path = evidence_root / relative_path
        if not path.is_file():
            raise AssertionError(
                "private operator-coverage evidence is incomplete under "
                f"{evidence_root}: missing {relative_path}"
            )
        ledger = _read_json(path, "family coverage ledger")
        assert ledger.get("schema_version") == 2, path
        assert ledger.get("ledger_kind") == "current-family-coverage", path
        operators = ledger.get("operators")
        assert isinstance(operators, list) and operators, path
        assert len(operators) == len(set(operators)), path
        assert set(operators) == expected_operators, path
        assert not seen_operators.intersection(operators), (
            "family ledgers duplicate an operator",
            path,
        )
        seen_operators.update(operators)

        scenario_ids = ledger.get("scenario_ids")
        assert scenario_ids == list(R73_SCENARIO_IDS), path
        assert len(scenario_ids) == len(set(scenario_ids)) == 48, path
        default_status = ledger.get("default_status")
        assert default_status in FAMILY_STATUS_KEYS, path
        overrides = ledger.get("status_overrides")
        assert isinstance(overrides, dict)
        assert set(overrides) == set(operators), path

        expanded: dict[str, dict[str, tuple[str, str | None]]] = {}
        for operator in operators:
            expanded[operator] = {
                scenario_id: (default_status, None)
                for scenario_id in R73_SCENARIO_IDS
            }
            operator_overrides = overrides[operator]
            assert isinstance(operator_overrides, dict), (path, operator)
            for status, group in operator_overrides.items():
                assert status in FAMILY_STATUS_KEYS, (path, operator, status)
                assert isinstance(group, dict), (path, operator, status)
                for scenario_id, reason in group.items():
                    assert scenario_id in R73_SCENARIO_IDS, (
                        path,
                        operator,
                        scenario_id,
                    )
                    assert expanded[operator][scenario_id][0] == default_status, (
                        path,
                        operator,
                        scenario_id,
                    )
                    if status != "covered":
                        assert isinstance(reason, str) and reason.strip(), (
                            path,
                            operator,
                            scenario_id,
                            status,
                        )
                    expanded[operator][scenario_id] = (status, reason)

        evidence = ledger.get("evidence")
        assert isinstance(evidence, dict)
        assert set(evidence) == set(operators), path
        for operator, values in evidence.items():
            assert isinstance(values, list) and values, (path, operator)
            for index, value in enumerate(values):
                assert isinstance(value, str) and value.strip(), (path, operator)
                _stable_test_locators(value, f"{path} {operator} evidence {index}")
        ledger["_relative_path"] = relative_path
        ledger["_expanded_status"] = expanded
        ledgers.append(ledger)

    assert seen_operators == set(OPERATOR_NAMES)
    return ledgers


def _project_private_ledgers(
    r73: dict[str, Any],
    family_ledgers: list[dict[str, Any]],
) -> dict[str, Any]:
    assert r73.get("inventory_phase") == "pre-symmetry inventory"
    assert r73.get("operators") == list(OPERATOR_NAMES)
    r73_scenario_ids = [scenario["id"] for scenario in r73.get("scenarios", [])]
    assert r73_scenario_ids == list(R73_SCENARIO_IDS)

    scenarios = []
    for scenario_id in R73_SCENARIO_IDS:
        coverage: dict[str, Any] = {}
        for ledger in family_ledgers:
            for operator, statuses in ledger["_expanded_status"].items():
                status, reason = statuses[scenario_id]
                if status == "covered":
                    coverage.setdefault(status, []).append(operator)
                else:
                    coverage.setdefault(status, {})[operator] = reason
        if "covered" in coverage:
            coverage["covered"].sort(key=OPERATOR_NAMES.index)
        scenarios.append({"id": scenario_id, "coverage": coverage})
    return {
        "schema_version": 2,
        "operators": list(OPERATOR_NAMES),
        "scenario_ids": list(R73_SCENARIO_IDS),
        "scenarios": scenarios,
    }


def _r73_statuses(r73: dict[str, Any]) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    for scenario in r73["scenarios"]:
        scenario_id = scenario["id"]
        coverage = scenario["coverage"]
        scenario_statuses: dict[str, str] = {}
        for status in STATUS_KEYS:
            if status not in coverage:
                continue
            for operator in _status_operators(coverage[status]):
                assert operator not in scenario_statuses, (scenario_id, operator)
                scenario_statuses[operator] = status
        assert set(scenario_statuses) == set(OPERATOR_NAMES), scenario_id
        statuses[scenario_id] = scenario_statuses
    return statuses


def _validate_upgrade_evidence(
    r73: dict[str, Any],
    family_ledgers: list[dict[str, Any]],
) -> None:
    r73_statuses = _r73_statuses(r73)
    for ledger in family_ledgers:
        if ledger["_relative_path"] not in SCENARIO_EVIDENCE_LEDGER_PATHS:
            continue
        path = ledger["_relative_path"]
        scenario_evidence = ledger.get("scenario_evidence")
        assert isinstance(scenario_evidence, dict), path
        assert set(scenario_evidence) == set(ledger["operators"]), path
        for operator in ledger["operators"]:
            operator_evidence = scenario_evidence[operator]
            assert isinstance(operator_evidence, dict), (path, operator)
            expected_upgrades = {
                scenario_id
                for scenario_id in R73_SCENARIO_IDS
                if r73_statuses[scenario_id][operator] in {"missing", "partial"}
                and ledger["_expanded_status"][operator][scenario_id][0]
                == "covered"
            }
            assert set(operator_evidence) == expected_upgrades, (
                path,
                operator,
                sorted(expected_upgrades),
                sorted(operator_evidence),
            )
            for scenario_id, values in operator_evidence.items():
                assert isinstance(values, list) and values, (
                    path,
                    operator,
                    scenario_id,
                )
                for index, value in enumerate(values):
                    assert isinstance(value, str) and value.strip(), (
                        path,
                        operator,
                        scenario_id,
                    )
                    _stable_test_locators(
                        value,
                        f"{path} {operator} {scenario_id} upgrade evidence {index}",
                    )


def _private_projection() -> dict[str, Any] | None:
    # A public/staged tree has no private ledgers and validates its checked-in
    # projection without attempting private-control reads.  A private checkout
    # compares automatically whenever the complete evidence set is present.
    evidence_root = _evidence_root()
    r73_path = evidence_root / "logs/r73-operator-test-symmetry-inventory/coverage.json"
    family_paths = [evidence_root / relative for relative, _ in FAMILY_LEDGER_SPECS]
    present = [r73_path.is_file(), *(path.is_file() for path in family_paths)]
    if not any(present):
        return None
    if not all(present):
        raise AssertionError(
            "private operator-coverage evidence is incomplete under "
            f"{evidence_root}: R73 and all six R74-R79 family ledgers are required"
        )
    r73 = _read_json(r73_path, "R73 pre-symmetry inventory")
    family_ledgers = _load_family_ledgers(evidence_root)
    _validate_upgrade_evidence(r73, family_ledgers)
    return _project_private_ledgers(r73, family_ledgers)


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    assert set(snapshot) == {"schema_version", "operators", "scenario_ids", "scenarios"}
    assert snapshot.get("schema_version") == 2
    assert snapshot.get("operators") == list(OPERATOR_NAMES)
    assert snapshot.get("scenario_ids") == list(R73_SCENARIO_IDS)

    scenarios = snapshot.get("scenarios")
    assert isinstance(scenarios, list)
    assert [scenario.get("id") for scenario in scenarios] == list(R73_SCENARIO_IDS)
    for scenario in scenarios:
        coverage = scenario.get("coverage")
        assert isinstance(coverage, dict)
        assert set(coverage) <= FAMILY_STATUS_KEYS
        placed: list[str] = []
        for status in STATUS_KEYS:
            if status not in coverage:
                continue
            group = coverage[status]
            operators = _status_operators(group)
            placed.extend(operators)
            if status == "covered":
                assert isinstance(group, list)
            else:
                assert isinstance(group, dict)
                for operator, reason in group.items():
                    assert operator in OPERATOR_NAMES
                    assert isinstance(reason, str) and reason.strip(), (
                        scenario["id"],
                        status,
                        operator,
                    )
        assert sorted(placed) == sorted(OPERATOR_NAMES), scenario["id"]
        assert len(placed) == len(set(placed)) == len(OPERATOR_NAMES)

    for value in _all_strings(snapshot):
        assert not any(
            forbidden in value
            for forbidden in (
                "tests/",
                "logs/",
                "docs/work",
                "packet",
                "owner",
                "evidence",
            )
        ), value


def _coverage_snapshot() -> dict[str, Any]:
    try:
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(
            f"unable to load public operator coverage snapshot {SNAPSHOT_PATH}: "
            f"{error}"
        ) from error
    if not isinstance(snapshot, dict):
        raise AssertionError("operator coverage snapshot must contain a JSON object")
    _validate_snapshot(snapshot)
    private = _private_projection()
    if private is not None and private != snapshot:
        raise AssertionError(
            "tests/operator_coverage_snapshot.json does not exactly match the "
            "R73/R74-R79 private-ledger projection"
        )
    return snapshot


def _assert_capabilities_are_meaningful(spec: Any) -> None:
    capabilities = set(spec.capabilities)
    assert capabilities
    assert capabilities <= KNOWN_CAPABILITIES
    parameter_names = {parameter.name for parameter in spec.parameters}
    structural_names = set(spec.structural_names)
    expected = {
        "lengths": "lengths" in structural_names,
        "affine": "gap_open_score" in parameter_names,
        "bandwidth": "bandwidth" in structural_names,
        "allowed-transpositions": "allowed_transpositions" in structural_names,
        "transposition-sources": "transposition_sources" in structural_names,
        "score-alignment": spec.name in {"sw", "sw_affine", "nw", "nw_affine"},
        "cost-alignment": spec.name == "dtw",
        "edit-match": spec.name == "lcs",
        "edit-distance": spec.name in {"lev", "osa", "damerau"},
        "chart-parser": spec.name in {"cky", "eisner"},
        "projective": spec.name == "eisner",
        "leaf-scores": "leaf_scores" in spec.input_names,
        "monotonic-alignment": spec.name == "mas",
    }
    assert spec.cost_native == (spec.orientation == "cost-native")
    assert {name for name, enabled in expected.items() if enabled} == capabilities


def _baseline_case_by_name() -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in EXPECTED_CASES}


def _assert_catalog_case_matches_baseline(
    spec: Any,
    expected: dict[str, Any],
) -> None:
    assert spec.name == expected["name"]
    assert spec.input_names == expected["input_names"]
    assert spec.profiles == expected["profiles"]
    assert tuple(
        (
            parameter.name,
            parameter.contract_default,
            parameter.matrix_value,
            parameter.mask_value,
            parameter.raw_vmap_name,
        )
        for parameter in spec.parameters
    ) == expected["parameters"]
    assert spec.structural_defaults == expected["structural_defaults"]
    assert spec.vjp_fields == expected["vjp_fields"]
    assert spec.orientation == expected["orientation"]
    assert spec.nn_class == expected["nn_class"]
    assert spec.operator_module == expected["operator_module"]
    assert spec.operator_name == expected["operator_name"]
    assert spec.nonnegative == expected["nonnegative"]
    for index_name in (
        "off_optimal_index",
        "padded_index",
        "invariance_index",
    ):
        assert getattr(spec, index_name) == expected[index_name]


def _assert_catalog_and_adapters_match_independent_pre_migration_baseline() -> None:
    expected_cases = _baseline_case_by_name()
    assert OPERATOR_ORDER == tuple(case["name"] for case in EXPECTED_CASES)
    assert OPERATOR_NAMES == OPERATOR_ORDER
    assert tuple(case.name for case in OPERATOR_CASES) == OPERATOR_ORDER

    for spec in OPERATOR_CASES:
        _assert_catalog_case_matches_baseline(spec, expected_cases[spec.name])

    contract = _import_adapter("test_v3_contract")
    validation = _import_adapter("test_validation")
    func = _import_adapter("test_func_compat")
    compile_tests = _import_adapter("test_compile")
    autocast = _import_adapter("test_autocast")
    masking = _import_adapter("test_masking")
    mask_kwarg = _import_adapter("test_mask_kwarg")
    invariance = _import_adapter("test_masking_invariance")

    assert autocast.AUTOCAST_DTYPE_IDS == EXPECTED_AUTOCAST_DTYPE_IDS
    for expected in EXPECTED_CASES:
        name = expected["name"]
        contract_case = next(case for case in contract.CASES if case.name == name)
        assert contract_case.input_names == expected["input_names"]
        assert contract_case.input_shapes == dict(expected["profiles"])["contract"]
        assert contract_case.param_defaults == tuple(
            (parameter[0], parameter[1]) for parameter in expected["parameters"]
        )
        assert contract_case.structural_defaults == expected["structural_defaults"]
        assert contract_case.vjp_fields == expected["vjp_fields"]
        assert contract_case.orientation == expected["orientation"]
        assert contract_case.nn_class == expected["nn_class"]
        assert contract_case.operator_module == expected["operator_module"]
        assert contract_case.operator_name == expected["operator_name"]
        assert contract_case.nonnegative == expected["nonnegative"]

        validation_case = next(case for case in validation.CASES if case.name == name)
        assert validation_case.input_shapes == dict(expected["profiles"])["validation"]
        assert validation_case.params == tuple(
            (parameter[0], parameter[2]) for parameter in expected["parameters"]
        )

        for module, profile in (
            (func, "func"),
            (compile_tests, "compile"),
            (autocast, "autocast"),
        ):
            case_name = "_AUTOCAST_CASES" if module is autocast else "CASES"
            adapted = next(
                case for case in getattr(module, case_name) if case.name == name
            )
            assert adapted.input_shapes == dict(expected["profiles"])[profile]
            assert adapted.params == tuple(
                (parameter[0], parameter[2]) for parameter in expected["parameters"]
            )

        assert func.PARAMETER_VMAP_ARGUMENT_NAMES[name] == tuple(
            parameter[4] for parameter in expected["parameters"]
        )

        masking_case = next(case for case in masking.CASES if case.name == name)
        assert masking_case.input_shapes == dict(expected["profiles"])["masking"]
        assert masking_case.cost_native == (expected["orientation"] == "cost-native")
        assert masking_case.params == tuple(
            (parameter[0], parameter[3]) for parameter in expected["parameters"]
        )
        assert masking_case.off_optimal_index == expected["off_optimal_index"]
        assert masking_case.padded_index == expected["padded_index"]

        if name == "cky":
            assert name not in mask_kwarg._CASES
        else:
            assert mask_kwarg._CASES[name] == (
                expected["orientation"] == "cost-native",
                dict(expected["profiles"])["mask_kwarg"][0],
            )
        assert invariance._CASES[name] == (
            expected["orientation"] == "cost-native",
            dict(expected["profiles"])["invariance"][0],
            expected["invariance_index"],
        )


def test_operator_coverage_snapshot_is_valid_and_private_projection_matches() -> None:
    _coverage_snapshot()


def test_catalog_matches_the_authoritative_twelve_operator_surface() -> None:
    _assert_catalog_and_adapters_match_independent_pre_migration_baseline()
    snapshot = _coverage_snapshot()
    expected = tuple(snapshot["operators"])
    engine_only = {"sv_affine", "sv_linear"}

    assert expected == OPERATOR_NAMES
    assert tuple(name for name in ops.__all__ if name not in engine_only) == expected
    assert engine_only.issubset(ops.__all__)
    assert set(_operator_imports_from_helper()) == set(expected)
    assert "hamming" not in expected

    for case in OPERATOR_CASES:
        assert case.public_entrypoints == (
            case.name,
            f"{case.name}_value",
            f"{case.name}_entropy",
        )
        for entrypoint in case.public_entrypoints:
            assert hasattr(orihime, entrypoint), entrypoint
        assert hasattr(orihime.raw, case.name), case.name
        assert hasattr(orihime.nn, case.nn_class), case.nn_class
        assert case.vjp_fields
        assert set(case.vjp_fields).isdisjoint(case.structural_names)
        _assert_capabilities_are_meaningful(case)


def test_suite_adapters_are_lossless_views_of_the_catalog() -> None:
    snapshot = _coverage_snapshot()
    expected = set(snapshot["operators"])
    contract = _import_adapter("test_v3_contract")
    validation = _import_adapter("test_validation")
    func = _import_adapter("test_func_compat")
    compile_tests = _import_adapter("test_compile")
    autocast = _import_adapter("test_autocast")
    masking = _import_adapter("test_masking")
    mask_kwarg = _import_adapter("test_mask_kwarg")
    invariance = _import_adapter("test_masking_invariance")

    assert {case.name for case in contract.CASES} == expected
    assert {case.name for case in validation.CASES} == expected
    assert {case.name for case in func.CASES} == expected
    assert {case.name for case in compile_tests.CASES} == expected
    assert {case.name for case in autocast._AUTOCAST_CASES} == expected
    assert {case.name for case in masking.CASES} == expected
    assert set(mask_kwarg._CASES) == expected - {"cky"}
    assert set(invariance._CASES) == expected

    for spec in OPERATOR_CASES:
        expected_indices = EXPECTED_SUITE_INDICES[spec.name]
        assert {
            "off_optimal_index": spec.off_optimal_index,
            "padded_index": spec.padded_index,
            "invariance_index": spec.invariance_index,
        } == expected_indices

        contract_case = next(case for case in contract.CASES if case.name == spec.name)
        assert contract_case.input_names == spec.input_names
        assert contract_case.input_shapes == spec.shapes("contract")
        assert contract_case.param_defaults == spec.contract_params
        assert contract_case.structural_defaults == spec.structural_defaults
        assert contract_case.vjp_fields == spec.vjp_fields
        assert contract_case.orientation == spec.orientation
        assert contract_case.nn_class == spec.nn_class

        validation_case = next(case for case in validation.CASES if case.name == spec.name)
        assert validation_case.input_shapes == spec.shapes("validation")
        assert validation_case.params == spec.matrix_params

        for module, profile in (
            (func, "func"),
            (compile_tests, "compile"),
            (autocast, "autocast"),
        ):
            case_name = "_AUTOCAST_CASES" if module is autocast else "CASES"
            adapted = next(
                case for case in getattr(module, case_name)
                if case.name == spec.name
            )
            assert adapted.input_shapes == spec.shapes(profile)
            assert adapted.params == spec.matrix_params

        masking_case = next(case for case in masking.CASES if case.name == spec.name)
        assert masking_case.input_shapes == spec.shapes("masking")
        assert masking_case.params == spec.mask_params
        assert masking_case.off_optimal_index == expected_indices["off_optimal_index"]
        assert masking_case.padded_index == expected_indices["padded_index"]

        if spec.name != "cky":
            assert mask_kwarg._CASES[spec.name] == (
                spec.cost_native,
                spec.shapes("mask_kwarg")[0],
            )
        else:
            assert spec.name not in mask_kwarg._CASES
        assert invariance._CASES[spec.name] == (
            spec.cost_native,
            spec.shapes("invariance")[0],
            expected_indices["invariance_index"],
        )


def test_r73_logical_scenario_partition_is_complete_and_reasoned() -> None:
    snapshot = _coverage_snapshot()
    scenarios = snapshot["scenarios"]
    assert tuple(scenario["id"] for scenario in scenarios) == R73_SCENARIO_IDS
    assert len(scenarios) == len(R73_SCENARIO_IDS) == 48

    for scenario in scenarios:
        coverage = scenario["coverage"]
        placed: list[str] = []
        for status in STATUS_KEYS:
            group = coverage.get(status, [])
            operators = _status_operators(group)
            placed.extend(operators)
            if isinstance(group, dict):
                for operator, reason in group.items():
                    assert operator in OPERATOR_NAMES
                    assert isinstance(reason, str) and reason.strip(), (
                        scenario["id"],
                        status,
                        operator,
                    )
        assert sorted(placed) == sorted(OPERATOR_NAMES)
        assert len(placed) == len(set(placed)) == len(OPERATOR_NAMES)


def test_current_family_ledgers_use_resolvable_stable_node_locators() -> None:
    _coverage_snapshot()
    evidence_root = _evidence_root()
    required_paths = [
        evidence_root / "logs/r73-operator-test-symmetry-inventory/coverage.json",
        *(evidence_root / relative for relative, _ in FAMILY_LEDGER_SPECS),
    ]
    present = [path.is_file() for path in required_paths]
    if not any(present):
        assert not any(path.exists() for path in required_paths)
        return
    assert all(present)
    ledgers = _load_family_ledgers(evidence_root)
    locators = [
        value
        for ledger in ledgers
        for values in ledger["evidence"].values()
        for value in values
        if "tests/" in value
    ]
    assert locators


def test_multi_gpu_marker_and_guards_are_normalized() -> None:
    _coverage_snapshot()
    pyproject = tomllib.loads(
        (_source_root() / "pyproject.toml").read_text(encoding="utf-8")
    )
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]
    assert pytest_options["strict_markers"] is True
    assert any(
        str(marker).startswith("multi_gpu:")
        for marker in pytest_options["markers"]
    )

    helper_tree = _source_tree("tests/operator_test_prerequisites.py")
    _shared_guard_assignment(helper_tree)

    all_test_modules = _test_module_paths()
    seen_files: set[str] = set()
    seen_nodes = 0
    for relative_path in all_test_modules:
        tree = _source_tree(relative_path)
        nodes = _multi_gpu_nodes(tree)
        guards = _shared_guard_nodes(tree)
        if nodes:
            seen_files.add(relative_path)
            seen_nodes += len(nodes)
        for node in nodes:
            assert any(
                _decorator_is_shared_guard(decorator)
                for decorator in node.decorator_list
            ), f"{relative_path}:{node.lineno} lacks shared CUDA prerequisite"
        for node in guards:
            assert any(
                _decorator_is_multi_gpu(decorator)
                for decorator in node.decorator_list
            ), f"{relative_path}:{node.lineno} uses shared guard without multi_gpu marker"
        assert not _inline_two_device_skipifs(tree), (
            f"{relative_path} contains an inline two-device skipif; use "
            "operator_test_prerequisites.TWO_CUDA_DEVICES_REQUIRED"
        )
        assert not any(
            isinstance(node, ast.Name) and node.id == "MULTI_CUDA_REQUIRED"
            for node in ast.walk(tree)
        ), f"{relative_path} retains a local two-device guard name"

        if relative_path in OPERATOR_TEST_FILES:
            assert _top_level_shared_guard_import(tree), (
                f"{relative_path} must import TWO_CUDA_DEVICES_REQUIRED from "
                "operator_test_prerequisites at module scope"
            )

    assert set(OPERATOR_TEST_FILES) <= seen_files
    assert seen_nodes > 0
    # DEVICE-02's validation.py cases compare one CUDA device with CPU. They
    # are not DEVICE-03 two-CUDA cases and must not be forced into this marker.
    assert not _multi_gpu_nodes(_source_tree("tests/test_validation.py"))
