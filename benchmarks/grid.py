#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Orihime's public map and value APIs on the d²p production grid."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch

import orihime as ohm


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "grid_cases.json"
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

PARAMETERS: dict[str, dict[str, float]] = {
    "sw": {"gap_score": -0.75, "temperature": 0.73},
    "sw_affine": {
        "gap_open_score": -1.25,
        "gap_extend_score": -0.25,
        "temperature": 0.73,
    },
    "sv": {"gap_score": -0.75, "temperature": 0.73},
    "sv_affine": {
        "gap_open_score": -2.0,
        "gap_extend_score": -0.5,
        "temperature": 1.0,
    },
    "nw": {"gap_score": -0.75, "temperature": 0.73},
    "nw_affine": {
        "gap_open_score": -1.25,
        "gap_extend_score": -0.25,
        "temperature": 0.73,
    },
    "dtw": {"temperature": 0.73},
    "lcs": {"temperature": 0.73},
    "lev": {
        "insertion_cost": 0.61,
        "deletion_cost": 0.67,
        "temperature": 0.73,
    },
    "osa": {
        "insertion_cost": 0.61,
        "deletion_cost": 0.67,
        "transposition_cost": 0.42,
        "temperature": 0.73,
    },
    "damerau": {
        "insertion_cost": 0.61,
        "deletion_cost": 0.67,
        "transposition_cost": 0.42,
        "temperature": 0.73,
    },
    "mas": {"temperature": 0.73},
    "cky": {"temperature": 1.0},
    "eisner": {"temperature": 0.73},
}


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _random(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    value = torch.rand(shape, generator=generator, dtype=torch.float32) - 0.5
    return value.cuda(non_blocking=False)


def _lengths(case: dict[str, Any], profile_id: str) -> torch.Tensor | None:
    policy = case["length_policy"]
    mode = policy["mode"]
    if mode == "not-supported":
        return None
    batch = int(case["batch"])
    dimensions = case["dimensions"]
    if mode == "explicit":
        values = policy["values"]
    elif profile_id == "eisner-production-v1":
        values = [dimensions["length"]] * batch
    elif profile_id == "mas-production-v1":
        values = [[dimensions["time"], dimensions["source"]]] * batch
    else:
        values = [[dimensions["length1"], dimensions["length2"]]] * batch
    return torch.tensor(values, dtype=torch.int32, device="cuda").contiguous()


def _structured_mask(
    case: dict[str, Any], profile_id: str
) -> torch.Tensor | None:
    mode = case["mask_policy"]["mode"]
    if mode == "none":
        return None
    batch = int(case["batch"])
    dimensions = case["dimensions"]
    if profile_id == "cky-production-v1":
        n = int(dimensions["length"])
        mask = torch.zeros((batch, n, n, n), dtype=torch.bool, device="cuda")
        mask[..., ::4] = True
        return mask
    if profile_id == "eisner-production-v1":
        n = int(dimensions["length"])
        mask = torch.zeros((batch, n, n), dtype=torch.bool, device="cuda")
        diagonal = torch.arange(n, device="cuda")
        mask[:, diagonal, diagonal] = True
        mask[:, 1::4, ::3] = True
        return mask
    raise ValueError(f"unsupported mask policy {mode!r} for {profile_id}")


def _eisner_diagonal_mask(case: dict[str, Any]) -> torch.Tensor:
    batch = int(case["batch"])
    n = int(case["dimensions"]["length"])
    mask = torch.zeros((batch, n, n), dtype=torch.bool, device="cuda")
    diagonal = torch.arange(n, device="cuda")
    mask[:, diagonal, diagonal] = True
    return mask


def _damerau_sources(
    primary: torch.Tensor, lengths: torch.Tensor | None
) -> torch.Tensor:
    batch, length1, length2 = primary.shape
    sources = torch.full(
        (batch, length1, length2, 2),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    if length1 > 1 and length2 > 1:
        rows = torch.arange(length1 - 1, dtype=torch.int32, device="cuda")
        columns = torch.arange(length2 - 1, dtype=torch.int32, device="cuda")
        sources[:, 1:, 1:, 0] = rows.view(1, -1, 1)
        sources[:, 1:, 1:, 1] = columns.view(1, 1, -1)
    if lengths is not None:
        rows = torch.arange(length1, device="cuda").view(1, length1, 1)
        columns = torch.arange(length2, device="cuda").view(1, 1, length2)
        active = (rows < lengths[:, 0].view(-1, 1, 1)) & (
            columns < lengths[:, 1].view(-1, 1, 1)
        )
        sources.masked_fill_(~active.unsqueeze(-1), -1)
    return sources.contiguous()


def make_call(
    operator: str,
    profile_id: str,
    case_index: int,
    case: dict[str, Any],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    batch = int(case["batch"])
    dimensions = case["dimensions"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5A17 + case_index)
    kwargs = dict(PARAMETERS[operator])

    if profile_id == "cky-production-v1":
        n = int(dimensions["length"])
        merge = _random((batch, n, n, n), generator=generator)
        leaf = _random((batch, n), generator=generator)
        mask = _structured_mask(case, profile_id)
        if mask is not None:
            kwargs["mask"] = mask
        return (merge, leaf), kwargs

    if profile_id == "eisner-production-v1":
        n = int(dimensions["length"])
        primary = _random((batch, n, n), generator=generator)
    elif profile_id == "mas-production-v1":
        primary = _random(
            (batch, int(dimensions["time"]), int(dimensions["source"])),
            generator=generator,
        )
    else:
        primary = _random(
            (batch, int(dimensions["length1"]), int(dimensions["length2"])),
            generator=generator,
        )

    lengths = _lengths(case, profile_id)
    kwargs["lengths"] = lengths
    mask = _structured_mask(case, profile_id)
    if profile_id == "eisner-production-v1" and mask is None:
        mask = _eisner_diagonal_mask(case)
    if mask is not None:
        kwargs["mask"] = mask
    if operator == "osa":
        allowed = torch.zeros_like(primary, dtype=torch.bool)
        if primary.shape[1] > 1 and primary.shape[2] > 1:
            allowed[:, 1::4, 1::4] = True
        if lengths is not None:
            rows = torch.arange(primary.shape[1], device="cuda").view(1, -1, 1)
            columns = torch.arange(primary.shape[2], device="cuda").view(1, 1, -1)
            active = (rows < lengths[:, 0].view(-1, 1, 1)) & (
                columns < lengths[:, 1].view(-1, 1, 1)
            )
            allowed &= active
        kwargs["allowed_transpositions"] = allowed.contiguous()
    elif operator == "damerau":
        kwargs["transposition_sources"] = _damerau_sources(primary, lengths)
    return (primary,), kwargs


def event_time_ms(fn: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(iterations):
        output = fn()
    end.record()
    end.synchronize()
    del output
    return float(start.elapsed_time(end)) / iterations


def benchmark(
    fn: Callable[[], torch.Tensor], *, repeats: int, target_ms: float
) -> dict[str, Any]:
    output = None
    for _ in range(5):
        output = fn()
    torch.cuda.synchronize()
    if output is None or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("benchmark output is non-finite")
    del output

    probe_ms = max(event_time_ms(fn, 1), 0.001)
    iterations = max(3, min(200, math.ceil(target_ms / probe_ms)))
    samples = [event_time_ms(fn, iterations) for _ in range(repeats)]
    return {
        "iterations": iterations,
        "max_ms": max(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "repeats": repeats,
    }


def _matches(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def selected_jobs(
    catalog: dict[str, Any],
    *,
    operators: list[str],
    case_patterns: list[str],
    surfaces: list[str],
):
    for operator in PUBLIC_ALGORITHMS:
        if operator not in operators:
            continue
        profile_id = catalog["family_shape_profiles"][operator]
        for case_index, case in enumerate(
            catalog["shape_profiles"][profile_id]["cases"]
        ):
            if not _matches(case["id"], case_patterns):
                continue
            for surface in surfaces:
                yield operator, profile_id, case_index, case, surface


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=PUBLIC_ALGORITHMS,
        default=list(PUBLIC_ALGORITHMS),
    )
    parser.add_argument("--cases", nargs="+", default=["*"])
    parser.add_argument(
        "--surfaces",
        nargs="+",
        choices=("value", "map"),
        default=["value", "map"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog()
    jobs = list(
        selected_jobs(
            catalog,
            operators=args.operators,
            case_patterns=args.cases,
            surfaces=args.surfaces,
        )
    )
    if args.list_cases:
        for operator, profile_id, _, case, surface in jobs:
            print(f"{operator}\t{surface}\t{profile_id}\t{case['id']}")
        return
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA or HIP device is required")
    if args.repeats < 1 or args.target_ms <= 0:
        raise SystemExit("--repeats and --target-ms must be positive")

    completed: set[tuple[str, str, str]] = set()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and args.output.exists():
            for line in args.output.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("record_type") == "result" and row.get("status") == "passed":
                    completed.add((row["operator"], row["case"], row["surface"]))
        elif args.output.exists():
            args.output.unlink()

    properties = torch.cuda.get_device_properties(0)
    header = {
        "record_type": "metadata",
        "schema_version": catalog["schema_version"],
        "catalog_origin": catalog["origin"],
        "commit": os.environ.get("ORIHIME_COMMIT", "unknown"),
        "orihime": ohm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "total_memory_bytes": properties.total_memory,
        "python": sys.version.split()[0],
        "job_count": len(jobs),
        "timestamp_unix": time.time(),
        "timing": "median GPU-event milliseconds per public API call; 5 warmups",
    }
    print(json.dumps(header, sort_keys=True), flush=True)
    if args.output and not (args.resume and args.output.exists()):
        _write_jsonl(args.output, header)

    for index, (operator, profile_id, case_index, case, surface) in enumerate(
        jobs, 1
    ):
        identity = (operator, case["id"], surface)
        if identity in completed:
            continue
        call_args, call_kwargs = make_call(
            operator, profile_id, case_index, case
        )
        function_name = operator if surface == "map" else f"{operator}_value"
        function = getattr(ohm, function_name)

        def invoke(
            function=function,
            call_args=call_args,
            call_kwargs=call_kwargs,
        ):
            return function(*call_args, **call_kwargs)

        row = {
            "record_type": "result",
            "operator": operator,
            "profile": profile_id,
            "case": case["id"],
            "surface": surface,
            "batch": case["batch"],
            "dimensions": case["dimensions"],
            "length_policy": case["length_policy"],
            "mask_policy": case["mask_policy"],
            "parameters": PARAMETERS[operator],
            "tags": case["tags"],
        }
        try:
            with torch.no_grad():
                timing = benchmark(
                    invoke, repeats=args.repeats, target_ms=args.target_ms
                )
            row.update(timing)
            row["items_per_second"] = 1000.0 * case["batch"] / timing["median_ms"]
            row["status"] = "passed"
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        print(
            json.dumps({"progress": f"{index}/{len(jobs)}", **row}, sort_keys=True),
            flush=True,
        )
        if args.output:
            _write_jsonl(args.output, row)
        del call_args, call_kwargs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
