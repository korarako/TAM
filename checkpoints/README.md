# Released checkpoints

The checkpoint paths, SHA-256 digests, and original ADTM source runs are indexed
in `checkpoints/index.yaml`. Model and training hyperparameters are recorded in
`configs/final_runs.yaml` and `configs/reproduction_profiles.yaml`.

- DW1D and DW2D each include the final base FM and AM checkpoints.
- MB2D uses one shared multi-anchor FM checkpoint and one AM checkpoint per
  transfer.
- DW4 includes three formal AM replicates. `am_seed2.pkl` is the representative
  formal checkpoint for visualization, while aggregate claims use all three
  seeds. The exploratory scan-selected checkpoint is intentionally excluded
  from the v0.1.0 public release tree.
- The three reported LJ13 evaluation seeds used the same FM and AM checkpoint,
  so the release stores each binary only once. LJ13 is a mixed result:
  pairwise/radial geometry improves consistently, while energy W2 does not.
- Ala2 checkpoints are historical experimental failed-case artifacts. They are
  indexed separately under `experimental_checkpoints` and must not be included
  in the successful benchmark aggregate. The release retains representative
  `grid35 -> 400 K`, `grid3579 -> 400/600 K`, and updated
  `grid3579 -> 800 K` attempts together with their original metric JSON files.

The Ala2 metric files use the legacy key `ew2_2k`, but these runs record
`energy_w2_samples: 1000`; report them as fixed-sample energy W2 with
`n=1000`, not as a 2000-sample estimate.

All currently indexed historical checkpoints use the `target-refinement` AM
parameterization. New `anchor-residual` runs use a self-contained composite
checkpoint with `model_type: tam_anchor_residual`; no anchor-residual result is
claimed as a released best run until it has been trained and evaluated.

Use `scripts/inspect_checkpoint.py` to inspect package metadata and compute a
checkpoint SHA-256 digest without modifying it.

The v0.1.0 release contains 15 formal benchmark checkpoint files and seven
Ala2 failed-case checkpoint files. See `RELEASE_MANIFEST.yaml` for the exact
release boundary.
