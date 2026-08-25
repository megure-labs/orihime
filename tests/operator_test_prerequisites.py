# SPDX-License-Identifier: Apache-2.0
"""Dependency-light prerequisites shared by operator-local test suites."""

from __future__ import annotations

import pytest
import torch


TWO_CUDA_DEVICES_REQUIRED = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires multiple CUDA devices",
)
