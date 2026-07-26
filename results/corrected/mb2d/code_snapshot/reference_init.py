"""Auditable equilibrium-reference generators.

The legacy :mod:`adj_thermo.langevin` helpers are intentionally kept for
backwards compatibility.  New reference datasets should use a problem-specific
generator from this package whenever an independent equilibrium construction is
available.
"""

from adj_thermo.reference.mb2d_analytic import (
    MB2DDomain,
    generate_mb2d_reference_v2,
    validate_mb2d_reference_v2,
)

__all__ = [
    "MB2DDomain",
    "generate_mb2d_reference_v2",
    "validate_mb2d_reference_v2",
]
