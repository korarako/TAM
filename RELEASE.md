# TAM v0.1.0

TAM v0.1.0 is the first curated research release of Thermodynamic Adjoint
Matching for Boltzmann generators. It extracts the reproducible implementation,
selected checkpoints, run metadata, and evaluation records from the original
exploratory workspace.

## Scientific scope

The release contains six system families with the following fixed
interpretation:

| Category | Systems | Release claim |
| --- | --- | --- |
| Positive | DW1D, DW2D | AM improves the released energy and geometric metrics over the corresponding FM baseline |
| Positive | DW4 | The three formal AM training seeds improve energy W2; seed 2 is representative, not the sole evidence |
| Mixed | MB2D | The two high-beta transfers improve, while the two low-beta transfers worsen in energy W2 despite small geometric improvements |
| Mixed | LJ13 | Pairwise/radial geometry improves modestly, while energy W2 is inconsistent across evaluation seeds |
| Experimental failed case | Ala2 | Some individual metrics improve, but the proposal retains a substantially mismatched high-energy tail |

These categories are part of the release contract. In particular, MB2D and
LJ13 are not labeled as uniformly successful, and Ala2 is not part of the
successful benchmark aggregate.

## Included in the source repository

- the `tam` command-line package;
- FM and AM training, sampling, and evaluation code;
- exact and reconstructed reproduction profiles;
- checkpoint and dataset manifests;
- curated machine-readable metric records;
- tests;
- the small Ala2 topology file;
- third-party license notices;
- SHA-256 provenance records.

The checkpoint layout and individual hashes are defined by
`checkpoints/index.yaml`. The released result protocol is defined by
`configs/final_runs.yaml`.

## GitHub Release assets

The v0.1.0 release is designed to carry these binary assets outside ordinary
Git history:

| Asset | Contents |
| --- | --- |
| `tam-checkpoints-v0.1.0.tar.gz` | Curated FM/AM checkpoints, including the separately labeled Ala2 failed-case checkpoints |
| `tam-checkpoints-v0.1.0.tar.gz.sha256` | Checksum for the checkpoint archive |
| `tam-reference-data-v0.1.0.tar.gz` | DW1D, DW2D, MB2D, DW4, and LJ13 reference arrays |
| `tam-reference-data-v0.1.0.tar.gz.sha256` | Checksum for the reference-data archive |

The `.pkl` checkpoint files in a full Git checkout are tracked with Git LFS.
The release archives provide a transport-independent alternative.

Large Ala2 MD trajectories are not part of either source history or the common
reference-data archive. They are distributed separately and are identified by
`data/ala2/dataset_manifest.yaml`.

## Checkpoint policy

The benchmark checkpoint registry contains:

- DW1D FM and AM;
- DW2D FM and AM;
- one shared MB2D multi-anchor FM and four transfer-specific AM checkpoints;
- DW4 FM and three formal AM training seeds;
- LJ13 FM and AM.

The representative Ala2 checkpoints are placed under
`experimental_checkpoints` with `status: experimental_failed_case`.
Exploratory or scan-selected checkpoints must not be used as evidence for the
formal multi-seed claims.

## Reproduction policy

Release-facing comparisons use the sampling and metric protocol recorded in
`configs/final_runs.yaml`. Generated sample count and metric sample count are
separate quantities. The primary release metrics are computed on fixed-size
2,000-sample subsets with repeated evaluations where recorded.

Profiles marked `reconstructed` reproduce the inferred legacy setup but are not
claimed to reproduce a checkpoint byte for byte. Accelerator nondeterminism can
also prevent byte-identical retraining.

## Verification

After downloading a full release checkout and assets, verify curated files:

```bash
sha256sum -c SHA256SUMS
```

Run the test suite:

```bash
python -m pip install -e ".[test]"
pytest
```

Print a reproduction profile without executing it:

```bash
tam reproduce --profile lj13_best --phase all
```

## Known limitations

The main scientific and reproducibility limitations are documented in
[`LIMITATIONS.md`](LIMITATIONS.md). The most important one is that the release
does not establish Adjoint Matching as a uniformly improving correction:
performance depends on the system, transfer region, observable, and sampling
seed.
