#!/usr/bin/env python3
"""Validate a proposed change's public Kaname-compatible provenance record.

The ``check`` command reads the proposed manifest directly from Git objects. It
never checks out or executes pull-request code, which makes it suitable for the
repository's ``pull_request_target`` policy workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Any


SCHEMA = "kaname.change-provenance/v1"
REPOSITORY = "megure-labs/orihime"
PROVENANCE_PREFIX = ".provenance/changes/"
MAX_MANIFEST_BYTES = 256 * 1024
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
CHANGE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}\Z")


class PolicyError(RuntimeError):
    """A deterministic provenance-policy violation."""


def git(*args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PolicyError(f"git {' '.join(args)} failed: {detail}")
    if text:
        return result.stdout.decode("utf-8", "strict")
    return result.stdout


def revision(value: str) -> str:
    resolved = str(git("rev-parse", "--verify", f"{value}^{{commit}}", text=True)).strip()
    if not COMMIT_RE.fullmatch(resolved):
        raise PolicyError(f"{value!r} did not resolve to a full commit id")
    return resolved


def changed_paths(base: str, head: str, *, diff_filter: str | None = None) -> list[str]:
    args = ["diff", "--name-only", "-z"]
    if diff_filter is not None:
        args.append(f"--diff-filter={diff_filter}")
    args.extend([base, head, "--", PROVENANCE_PREFIX])
    raw = bytes(git(*args))
    return [part.decode("utf-8", "strict") for part in raw.split(b"\0") if part]


def patch_digest(base: str, head: str) -> str:
    patch = bytes(
        git(
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-renames",
            base,
            head,
            "--",
            ".",
            ":(exclude).provenance/changes/**",
        )
    )
    return f"sha256:{hashlib.sha256(patch).hexdigest()}"


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PolicyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be a JSON object")
    return value


def exact_keys(value: dict[str, Any], label: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PolicyError(f"{label} keys differ; missing={missing}, extra={extra}")


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"{label} must be a non-empty string")
    return value


def digest_value(value: Any, label: str) -> str:
    text = nonempty_string(value, label)
    if not DIGEST_RE.fullmatch(text):
        raise PolicyError(f"{label} must be sha256:<64 lowercase hex characters>")
    if text == "sha256:" + "0" * 64:
        raise PolicyError(f"{label} may not be the all-zero placeholder digest")
    return text


def string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PolicyError(f"{label} must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        result.append(nonempty_string(item, f"{label}[{index}]"))
    if len(result) != len(set(result)):
        raise PolicyError(f"{label} entries must be unique")
    return result


def validate_manifest(
    manifest: dict[str, Any], *, path: str, base: str, expected_patch_digest: str
) -> None:
    exact_keys(
        manifest,
        "manifest",
        {
            "schema",
            "change",
            "kaname",
            "evidence",
            "actors",
            "external_materials",
            "validation",
            "clean_room",
        },
    )
    if manifest["schema"] != SCHEMA:
        raise PolicyError(f"schema must be {SCHEMA!r}")

    change = object_value(manifest["change"], "change")
    exact_keys(
        change,
        "change",
        {"id", "repository", "base_commit", "summary", "patch_digest"},
    )
    change_id = nonempty_string(change["id"], "change.id")
    if not CHANGE_ID_RE.fullmatch(change_id):
        raise PolicyError("change.id must be 8-128 lowercase letters, digits, '.', '_' or '-'")
    expected_name = f"{change_id}.json"
    if PurePosixPath(path).name != expected_name:
        raise PolicyError(f"manifest filename must be {expected_name!r}")
    if change["repository"] != REPOSITORY:
        raise PolicyError(f"change.repository must be {REPOSITORY!r}")
    if change["base_commit"] != base:
        raise PolicyError(f"change.base_commit must equal the pull request base {base}")
    nonempty_string(change["summary"], "change.summary")
    if change["patch_digest"] != expected_patch_digest:
        raise PolicyError(
            "change.patch_digest does not bind the proposed non-provenance diff; "
            f"expected {expected_patch_digest}"
        )

    kaname = object_value(manifest["kaname"], "kaname")
    exact_keys(
        kaname,
        "kaname",
        {
            "run_id",
            "attempt_ids",
            "host_nodes",
            "repro_manifest_digest",
            "dispatch_proof_digest",
            "trace_bundle_digest",
            "ledger_revision",
            "ledger_root_digest",
            "closure_certificate_digest",
            "status",
        },
    )
    nonempty_string(kaname["run_id"], "kaname.run_id")
    string_list(kaname["attempt_ids"], "kaname.attempt_ids")
    string_list(kaname["host_nodes"], "kaname.host_nodes")
    for field in (
        "repro_manifest_digest",
        "dispatch_proof_digest",
        "trace_bundle_digest",
        "ledger_root_digest",
        "closure_certificate_digest",
    ):
        digest_value(kaname[field], f"kaname.{field}")
    if type(kaname["ledger_revision"]) is not int or kaname["ledger_revision"] < 1:
        raise PolicyError("kaname.ledger_revision must be a positive integer")
    if kaname["status"] != "closed":
        raise PolicyError("kaname.status must be 'closed'")

    evidence = object_value(manifest["evidence"], "evidence")
    evidence_fields = {
        "task_graph_digest",
        "inputs_digest",
        "commands_digest",
        "tool_calls_digest",
        "transcripts_digest",
        "artifacts_digest",
        "environment_digest",
        "budget_outcome_digest",
        "validation_digest",
        "review_digest",
        "adjudication_digest",
    }
    exact_keys(evidence, "evidence", evidence_fields)
    for field in sorted(evidence_fields):
        digest_value(evidence[field], f"evidence.{field}")

    actors = manifest["actors"]
    if not isinstance(actors, list) or not actors:
        raise PolicyError("actors must be a non-empty array")
    actor_ids: set[str] = set()
    roles: dict[str, set[str]] = {}
    for index, raw_actor in enumerate(actors):
        actor = object_value(raw_actor, f"actors[{index}]")
        exact_keys(
            actor,
            f"actors[{index}]",
            {"id", "kind", "roles", "provider", "model", "harness", "effort"},
        )
        actor_id = nonempty_string(actor["id"], f"actors[{index}].id")
        if actor_id in actor_ids:
            raise PolicyError(f"duplicate actor id: {actor_id}")
        actor_ids.add(actor_id)
        kind = actor["kind"]
        if kind not in {"human", "agent", "service"}:
            raise PolicyError(f"actors[{index}].kind is invalid")
        actor_roles = string_list(actor["roles"], f"actors[{index}].roles")
        allowed_roles = {"accountable", "author", "implementer", "reviewer", "adjudicator"}
        if not set(actor_roles) <= allowed_roles:
            raise PolicyError(f"actors[{index}].roles contains an invalid role")
        for role in actor_roles:
            roles.setdefault(role, set()).add(actor_id)
        if kind == "agent":
            for field in ("provider", "model", "harness", "effort"):
                nonempty_string(actor[field], f"actors[{index}].{field}")
        else:
            for field in ("provider", "model", "harness", "effort"):
                if actor[field] is not None:
                    raise PolicyError(f"actors[{index}].{field} must be null for {kind}")

    if not any(
        actor.get("kind") == "human" and "accountable" in actor.get("roles", [])
        for actor in actors
        if isinstance(actor, dict)
    ):
        raise PolicyError("actors must include a human accountable for the change")
    producer_ids = roles.get("author", set()) | roles.get("implementer", set())
    reviewer_ids = roles.get("reviewer", set())
    adjudicator_ids = roles.get("adjudicator", set())
    if not producer_ids or not reviewer_ids or not adjudicator_ids:
        raise PolicyError("actors must include production, independent review, and adjudication roles")
    if producer_ids & reviewer_ids or producer_ids & adjudicator_ids or reviewer_ids & adjudicator_ids:
        raise PolicyError("producer, reviewer, and adjudicator identities must be independent")

    materials = manifest["external_materials"]
    if not isinstance(materials, list):
        raise PolicyError("external_materials must be an array, which may be empty")
    implementation_source_viewed = False
    for index, raw_material in enumerate(materials):
        material = object_value(raw_material, f"external_materials[{index}]")
        exact_keys(
            material,
            f"external_materials[{index}]",
            {"kind", "locator", "purpose", "license", "digest", "implementation_source_viewed"},
        )
        for field in ("kind", "locator", "purpose", "license"):
            nonempty_string(material[field], f"external_materials[{index}].{field}")
        if material["digest"] is not None:
            digest_value(material["digest"], f"external_materials[{index}].digest")
        if not isinstance(material["implementation_source_viewed"], bool):
            raise PolicyError(
                f"external_materials[{index}].implementation_source_viewed must be boolean"
            )
        implementation_source_viewed |= material["implementation_source_viewed"]

    validations = manifest["validation"]
    if not isinstance(validations, list) or not validations:
        raise PolicyError("validation must contain at least one passed validation record")
    for index, raw_validation in enumerate(validations):
        validation = object_value(raw_validation, f"validation[{index}]")
        exact_keys(
            validation,
            f"validation[{index}]",
            {"command", "environment", "outcome", "evidence_digest"},
        )
        nonempty_string(validation["command"], f"validation[{index}].command")
        nonempty_string(validation["environment"], f"validation[{index}].environment")
        if validation["outcome"] != "passed":
            raise PolicyError(f"validation[{index}].outcome must be 'passed'")
        digest_value(validation["evidence_digest"], f"validation[{index}].evidence_digest")

    clean_room = object_value(manifest["clean_room"], "clean_room")
    exact_keys(
        clean_room,
        "clean_room",
        {
            "external_implementation_source_viewed",
            "clean_room_restart_performed",
            "maintainer_decision_digest",
        },
    )
    if not isinstance(clean_room["external_implementation_source_viewed"], bool):
        raise PolicyError("clean_room.external_implementation_source_viewed must be boolean")
    if not isinstance(clean_room["clean_room_restart_performed"], bool):
        raise PolicyError("clean_room.clean_room_restart_performed must be boolean")
    if clean_room["external_implementation_source_viewed"] != implementation_source_viewed:
        raise PolicyError("clean_room source-view declaration must match external_materials")
    if implementation_source_viewed:
        if not clean_room["clean_room_restart_performed"]:
            raise PolicyError("viewed implementation source requires a clean-room restart")
        digest_value(
            clean_room["maintainer_decision_digest"],
            "clean_room.maintainer_decision_digest",
        )
    elif clean_room["maintainer_decision_digest"] is not None:
        raise PolicyError("maintainer_decision_digest must be null when no implementation source was viewed")


def check(base_arg: str, head_arg: str) -> None:
    base = revision(base_arg)
    head = revision(head_arg)
    touched = changed_paths(base, head)
    added = changed_paths(base, head, diff_filter="A")
    if len(added) != 1:
        raise PolicyError(
            "each pull request must add exactly one .provenance/changes/<change-id>.json record"
        )
    if touched != added:
        raise PolicyError("existing provenance records are immutable and may not be modified or deleted")
    path = added[0]
    if not path.startswith(PROVENANCE_PREFIX) or not path.endswith(".json"):
        raise PolicyError("the added provenance record must be a JSON file in .provenance/changes/")
    raw = bytes(git("show", f"{head}:{path}"))
    if len(raw) > MAX_MANIFEST_BYTES:
        raise PolicyError(f"provenance record exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyError(f"invalid provenance JSON: {error}") from error
    validate_manifest(
        object_value(manifest, "manifest"),
        path=path,
        base=base,
        expected_patch_digest=patch_digest(base, head),
    )
    print(f"validated {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="validate one proposed change")
    check_parser.add_argument("--base", required=True)
    check_parser.add_argument("--head", required=True)
    digest_parser = subparsers.add_parser(
        "digest", help="print the canonical non-provenance patch digest"
    )
    digest_parser.add_argument("--base", required=True)
    digest_parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        base = revision(args.base)
        head = revision(args.head)
        if args.command == "check":
            check(base, head)
        else:
            print(patch_digest(base, head))
    except PolicyError as error:
        print(f"provenance policy: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
