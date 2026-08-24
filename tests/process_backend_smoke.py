from __future__ import annotations

import numpy as np

from adj_thermo.geometric_alignment import joint_alignment_cost_matrix


def main() -> None:
    rng = np.random.default_rng(20260824)
    left = rng.normal(size=(2, 13, 3))
    right = rng.normal(size=(2, 13, 3))
    serial = joint_alignment_cost_matrix(left, right, workers=1)
    parallel = joint_alignment_cost_matrix(
        left,
        right,
        workers=2,
        parallel_backend="process",
    )
    np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=1.0e-12)


if __name__ == "__main__":
    main()
