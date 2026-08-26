import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_change_provenance.py"
SCHEMA_PATH = ROOT / ".provenance" / "change-provenance.schema.json"

SPEC = importlib.util.spec_from_file_location("validate_change_provenance", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def digest(byte: str) -> str:
    return "sha256:" + byte * 64


def valid_manifest() -> dict:
    return {
        "schema": "kaname.change-provenance/v1",
        "change": {
            "id": "change-1234",
            "repository": "megure-labs/orihime",
            "base_commit": "a" * 40,
            "summary": "Exercise the provenance contract.",
            "patch_digest": digest("b"),
        },
        "kaname": {
            "run_id": "run-1",
            "attempt_ids": ["attempt-1"],
            "host_nodes": ["node-1"],
            "repro_manifest_digest": digest("1"),
            "dispatch_proof_digest": digest("2"),
            "trace_bundle_digest": digest("3"),
            "ledger_revision": 1,
            "ledger_root_digest": digest("4"),
            "closure_certificate_digest": digest("5"),
            "status": "closed",
        },
        "evidence": {
            "task_graph_digest": digest("6"),
            "inputs_digest": digest("7"),
            "commands_digest": digest("8"),
            "tool_calls_digest": digest("9"),
            "transcripts_digest": digest("a"),
            "artifacts_digest": digest("b"),
            "environment_digest": digest("c"),
            "budget_outcome_digest": digest("d"),
            "validation_digest": digest("e"),
            "review_digest": digest("f"),
            "adjudication_digest": digest("2"),
        },
        "actors": [
            {
                "id": "caseysm",
                "kind": "human",
                "roles": ["accountable", "author"],
                "provider": None,
                "model": None,
                "harness": None,
                "effort": None,
            },
            {
                "id": "review-agent",
                "kind": "agent",
                "roles": ["reviewer"],
                "provider": "provider-a",
                "model": "review-model",
                "harness": "kaname",
                "effort": "high",
            },
            {
                "id": "adjudication-agent",
                "kind": "agent",
                "roles": ["adjudicator"],
                "provider": "provider-b",
                "model": "adjudication-model",
                "harness": "kaname",
                "effort": "high",
            },
        ],
        "external_materials": [],
        "validation": [
            {
                "command": "python -m pytest -q",
                "environment": "Linux x86-64 CPU",
                "outcome": "passed",
                "evidence_digest": digest("1"),
            }
        ],
        "clean_room": {
            "external_implementation_source_viewed": False,
            "clean_room_restart_performed": False,
            "maintainer_decision_digest": None,
        },
    }


def validate(manifest: dict) -> None:
    validator.validate_manifest(
        manifest,
        path=".provenance/changes/change-1234.json",
        base="a" * 40,
        expected_patch_digest=digest("b"),
    )


def test_schema_and_validator_agree_on_identity():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["properties"]["schema"]["const"] == validator.SCHEMA
    assert schema["properties"]["change"]["properties"]["repository"]["const"] == validator.REPOSITORY
    validate(valid_manifest())


def test_patch_digest_is_exactly_bound():
    manifest = valid_manifest()
    manifest["change"]["patch_digest"] = digest("c")
    with pytest.raises(validator.PolicyError, match="does not bind"):
        validate(manifest)


def test_placeholder_digest_is_rejected():
    manifest = valid_manifest()
    manifest["kaname"]["trace_bundle_digest"] = "sha256:" + "0" * 64
    with pytest.raises(validator.PolicyError, match="placeholder"):
        validate(manifest)


def test_review_and_adjudication_must_be_independent():
    manifest = valid_manifest()
    manifest["actors"][1]["roles"].append("adjudicator")
    manifest["actors"] = manifest["actors"][:2]
    with pytest.raises(validator.PolicyError, match="independent"):
        validate(manifest)


def test_viewed_implementation_requires_clean_room_restart():
    manifest = valid_manifest()
    manifest["external_materials"].append(
        {
            "kind": "source",
            "locator": "https://example.invalid/source",
            "purpose": "disclosed contamination",
            "license": "unknown",
            "digest": None,
            "implementation_source_viewed": True,
        }
    )
    manifest["clean_room"]["external_implementation_source_viewed"] = True
    with pytest.raises(validator.PolicyError, match="clean-room restart"):
        validate(manifest)
