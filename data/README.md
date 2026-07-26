# Data manifests

The v0.2 source release does not store NumPy sample arrays. Corrected
reference metadata is recorded in:

- `mb2d_reference_v2/dataset_manifest.yaml`
- `dw4_reference_v2/dataset_manifest.yaml`
- `lj13_reference_v2/dataset_manifest.yaml`

The full audit records and source hashes are under the corresponding
`results/corrected/<system>` directories.

DW1D and DW2D are retained as legacy evidence only. Ala2 trajectories remain
external and are described by `ala2/dataset_manifest.yaml`.

The v0.1 MB2D, DW4, and LJ13 finite-time ULA manifests are intentionally absent
from the current branch. They remain available for historical audit at the
immutable `v0.1.0` tag.
