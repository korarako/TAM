# TAM

TAM is the release implementation of **Thermodynamic Adjoint Matching** for
Boltzmann generators. It first trains an inverse-temperature-conditioned Flow
Matching (FM) velocity field, then refines a selected anchor-to-target
temperature transfer using an energy-derived terminal adjoint.

This repository is a research release. Its benchmark outcomes are deliberately
classified as positive, mixed, or failed-case rather than presented as a
uniform improvement over FM.

## Release status

| System | Velocity model | Released transfer | Outcome in v0.1.0 |
| --- | --- | --- | --- |
| DW1D | MLP | 0.25 to 0.70 | **Positive:** AM improves the released energy and geometric metrics |
| DW2D | MLP | 0.25 to 0.70 | **Positive:** AM improves the released energy and geometric metrics |
| MB2D | MLP | four multi-anchor transfers | **Mixed:** high-beta transfers improve, while low-beta energy metrics worsen |
| DW4 | EGNN | anchors 0.8/1.2, target 1.0 | **Positive:** all three formal AM training seeds improve energy W2 |
| LJ13 | PaiNN | anchors 0.8/1.2, target 1.0 | **Mixed:** geometry improves modestly, but energy W2 is not stable across evaluation seeds |
| Ala2 | chiral PaiNN-JAX | experimental temperature transfers | **Failed case:** partial metric improvements do not resolve the mismatched high-energy tail |

The Ala2 checkpoints are retained to make the unsuccessful route inspectable
and reproducible. They are separated under `experimental_checkpoints` in
[`checkpoints/index.yaml`](checkpoints/index.yaml) and are not included in the
successful benchmark aggregate.

See [`RELEASE.md`](RELEASE.md) for the v0.1.0 artifact inventory and
[`LIMITATIONS.md`](LIMITATIONS.md) for the precise interpretation of these
results.

## Installation

TAM requires Python 3.11 or newer:

```bash
python -m pip install -e .
```

Optional dependencies:

```bash
python -m pip install -e ".[plots,test]"
python -m pip install -e ".[ala2]"  # adds OpenMM
```

The small MIT-licensed `painn-jax` implementation used by the LJ13 checkpoint
is vendored under `src/painn_jax`; its license is included in
`THIRD_PARTY_LICENSES`.

## Released checkpoints and data

The checkpoint registry is
[`checkpoints/index.yaml`](checkpoints/index.yaml). Checkpoint files are tracked
with Git LFS and are also intended to be distributed in the
`tam-checkpoints-v0.1.0` GitHub Release asset. Verify a downloaded release
against [`SHA256SUMS`](SHA256SUMS).

Reference NumPy arrays are intentionally not stored in ordinary Git history.
Their shapes, thermodynamic settings, provenance, and hashes are recorded in
the `data/**/dataset_manifest.yaml` files:

- DW1D, DW2D, MB2D, DW4, and LJ13 reference arrays are distributed as the
  `tam-reference-data-v0.1.0` GitHub Release asset.
- Ala2 MD trajectories are distributed separately. The repository includes
  only the small topology file, manifest, and loading instructions.

For Ala2, point TAM to the separately downloaded MD archive:

```bash
export TAM_ALA2_DATA_ROOT=/path/to/ala2
tam reproduce --profile ala2_fm_baseline --phase fm
```

## Reproduction profiles

The commands in this README are usage examples; the released run settings live
in [`configs/reproduction_profiles.yaml`](configs/reproduction_profiles.yaml)
and [`configs/final_runs.yaml`](configs/final_runs.yaml).

Print a complete profile without starting a run:

```bash
tam reproduce --profile lj13_best --phase all
```

Add `--execute` to run it. Profiles marked `exact` have saved run metadata.
Some legacy toy profiles are marked `reconstructed` because the original AM
runs did not save a separate configuration. A profile reproduces the training
setup, not necessarily a byte-identical checkpoint on nondeterministic
accelerator kernels.

## Basic workflow

Generate reference data with Langevin dynamics:

```bash
tam generate-data \
  --problem dw1d \
  --beta-grid 0.25 1.50 \
  --n-per-beta 100000
```

Train Flow Matching and target-refinement Adjoint Matching:

```bash
tam train-fm \
  --problem dw1d --model mlp \
  --data-dir data/dw1d --beta-grid 0.25 1.50 \
  --hidden-dim 512 --n-layers 3 \
  --run-dir outputs/dw1d

tam train-am \
  --problem dw1d --model mlp \
  --run-dir outputs/dw1d \
  --beta0 0.25 --beta1 0.70 \
  --hidden-dim 512 --n-layers 3 \
  --am-parameterization target-refinement
```

Generate samples in bounded-memory chunks:

```bash
tam sample \
  --problem lj13 \
  --checkpoint checkpoints/lj13/0p8_1p2_to_1p0/am_params.pkl \
  --beta 1.0 \
  --num-samples 100000 --batch-size 1000 \
  --ode-steps 150 --method euler --seed 101 \
  --output samples/lj13_am_beta1_seed101.npy
```

Evaluate repeated fixed-size subsets:

```bash
tam evaluate \
  --problem lj13 \
  --samples samples/lj13_am_beta1_seed101.npy \
  --reference data/lj13_paper/samples_beta_1.00.npy \
  --metric-samples 2000 --repeats 10 --seed 101 \
  --output results/lj13_am_beta1_seed101.json
```

The number of generated samples and the metric subset size are distinct.
Release-facing metrics use 2,000-sample subsets and should be reported with
their repeat statistics.

## AM parameterizations

TAM exposes two AM parameterizations:

- `target-refinement` initializes the trainable velocity at the pretrained
  target-conditioned slice, `v_new(beta1) = v_base(beta1)`. Every v0.1.0
  released result uses this parameterization.
- `anchor-residual` initializes a zero correction around the frozen anchor,
  `v_total = v_base(beta0) + v_trainable(beta1) - v_base(beta1)`. This route is
  implemented for further research but is not part of the v0.1.0 benchmark
  claims.

An anchor-residual checkpoint is stored as a self-contained composite package
containing the frozen FM package, correction package, anchor beta, and target
beta. The normal `tam sample` command loads it and rejects a sampling beta that
does not match the stored target.

## Reproducibility and provenance

- `configs/final_runs.yaml` maps each released checkpoint to its curated run
  configuration and sampling protocol.
- `checkpoints/index.yaml` records checkpoint roles, hashes, and the separation
  between benchmarks and experimental Ala2 failed cases.
- `data/**/dataset_manifest.yaml` records the externally distributed reference
  data.
- `SHA256SUMS` verifies curated artifacts.

The original exploratory ADTM workspace is not required to use this release
and is not modified by the release extraction.

## Citation

If you use TAM, cite the software metadata in [`CITATION.cff`](CITATION.cff).
GitHub exposes the same information through **Cite this repository**.

## License

TAM is released under the [MIT License](LICENSE). Vendored third-party code is
covered by the notices in `THIRD_PARTY_LICENSES`.
