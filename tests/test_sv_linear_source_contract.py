"""Build-free structural and semantic checks for the sv_linear source family."""

import itertools
import math
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "src" / "sv_linear" / "kernels.cu"
CPU = ROOT / "src" / "sv_linear" / "kernels_cpu.cpp"
CUDA_HEADER = ROOT / "src" / "sv_linear" / "kernels.cuh"
CPU_HEADER = ROOT / "src" / "sv_linear" / "kernels_cpu.h"
REGISTRY = ROOT / "src" / "sv_linear" / "registry.cpp"


def _lse(values, temp):
    finite = [value for value in values if value > -1e20]
    if not finite:
        return -1e30
    maximum = max(finite)
    return maximum + temp * math.log(
        sum(math.exp((value - maximum) / temp) for value in finite)
    )


def _sv_linear_reference(scores, gap, temp):
    length1 = len(scores)
    length2 = len(scores[0])
    ninf = -1e30
    match = [[ninf] * (length2 + 1) for _ in range(length1 + 1)]
    insert = [[ninf] * (length2 + 1) for _ in range(length1 + 1)]
    delete = [[ninf] * (length2 + 1) for _ in range(length1 + 1)]
    match[0][0] = 0.0
    for diagonal in range(2, length1 + length2 + 1):
        for i in range(max(1, diagonal - length2), min(length1, diagonal - 1) + 1):
            j = diagonal - i
            score = scores[i - 1][j - 1]
            match[i][j] = _lse(
                [(match[i - 1][j - 1] + score) if i > 1 and j > 1 else ninf,
                 (insert[i - 1][j - 1] + score) if i > 1 and j > 1 else ninf,
                 (delete[i - 1][j - 1] + score) if i > 1 and j > 1 else ninf,
                 score],
                temp,
            )
            insert[i][j] = _lse(
                [match[i - 1][j] + gap, insert[i - 1][j] + gap], temp
            )
            # Exactly one I->D cross; no symmetric D->I edge.
            delete[i][j] = _lse(
                [match[i][j - 1] + gap,
                 insert[i][j - 1] + gap,
                 delete[i][j - 1] + gap],
                temp,
            )
    return _lse(
        [0.0]
        + [match[i][j] for i in range(1, length1 + 1) for j in range(1, length2 + 1)],
        temp,
    )


def _exhaustive_value(scores, gap, temp):
    total = 1.0
    length1 = len(scores)
    length2 = len(scores[0])
    for matches in range(1, min(length1, length2) + 1):
        for indices1 in itertools.combinations(range(length1), matches):
            for indices2 in itertools.combinations(range(length2), matches):
                gaps = sum(
                    indices1[k + 1] - indices1[k] - 1
                    + indices2[k + 1] - indices2[k] - 1
                    for k in range(matches - 1)
                )
                score = sum(scores[indices1[k]][indices2[k]] for k in range(matches))
                total += math.exp((score + gap * gaps) / temp)
    return temp * math.log(total)


def test_reference_recurrence_matches_exhaustive_monotone_skeletons():
    scores = [[1.2, -0.4, 0.3], [-0.7, 0.9, 0.1], [0.2, -0.5, 1.1]]
    actual = _sv_linear_reference(scores, gap=-0.8, temp=0.7)
    expected = _exhaustive_value(scores, gap=-0.8, temp=0.7)
    assert actual == pytest.approx(expected, abs=1e-11)


def test_source_has_four_launch_units_and_no_dormant_cuda_symbols():
    cuda = CUDA.read_text()
    gpu_hosts = re.findall(r'void\s+(sv_linear_\w+)\s*\(', CUDA_HEADER.read_text())
    cpu_hosts = re.findall(r'void\s+(sv_linear_\w+_cpu)\s*\(', CPU_HEADER.read_text())
    device_symbols = re.findall(r'__global__\s+void\s+(sv_linear_\w+)\s*\(', cuda)
    assert gpu_hosts == [
        "sv_linear_forward",
        "sv_linear_backward",
        "sv_linear_hvp",
        "sv_linear_param_grad",
    ]
    assert cpu_hosts == [
        "sv_linear_forward_cpu",
        "sv_linear_backward_cpu",
        "sv_linear_hvp_cpu",
        "sv_linear_param_grad_cpu",
    ]
    assert "sv_linear_param_grad_init_U_kernel" not in cuda
    assert all(cuda.count(f"{symbol}<<<") >= 1 for symbol in device_symbols)
    assert "cudaMemsetAsync(d_U" in cuda


def test_registry_exposes_gap_temperature_api_only():
    registry = REGISTRY.read_text()
    apis = re.findall(r'm\.def\("([a-zA-Z0-9_]+)\(', registry)
    assert len(apis) == 13
    assert "sv_linear_marginals_grad_gap" in apis
    assert "sv_linear_marginals_grad_temp" in apis
    assert "gap_open" not in registry
    assert "gap_ext" not in registry
