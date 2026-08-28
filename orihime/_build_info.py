# SPDX-License-Identifier: Apache-2.0
"""Artifact compatibility identity, filled by release packaging.

Source builds leave these values unset because they compile against the active
environment. Final wheel and conda packaging replaces this module with the
exact ABI tuple used for that artifact.
"""

EXPECTED_TORCH_MINOR: str | None = None
EXPECTED_LANE: str | None = None
SOURCE_COMMIT: str | None = None
