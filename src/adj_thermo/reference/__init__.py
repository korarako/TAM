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
from adj_thermo.reference.dw4_smc import (
    DW4SMCConfig,
    compare_dw4_samples,
    run_dw4_smc,
    summarize_dw4,
)
from adj_thermo.reference.lj13_rehmc import (
    LJ13REHMCConfig,
    helmert_basis_np as lj13_helmert_basis_np,
    run_lj13_rehmc,
)

__all__ = [
    "MB2DDomain",
    "DW4SMCConfig",
    "LJ13REHMCConfig",
    "compare_dw4_samples",
    "generate_mb2d_reference_v2",
    "lj13_helmert_basis_np",
    "run_dw4_smc",
    "run_lj13_rehmc",
    "summarize_dw4",
    "validate_mb2d_reference_v2",
]
