# TAM

TAM is the research implementation of **Thermodynamic Adjoint Matching** for
Boltzmann generators. It first trains an inverse-temperature-conditioned Flow
Matching (FM) velocity field and then adapts a selected anchor-to-target
temperature transfer using an energy-derived terminal adjoint.

Version **0.2.0** is a corrected-reference release. It preserves both positive
and negative evidence and does not claim that Adjoint Matching uniformly
improves Flow Matching.

## Corrected result status

| System | Evidence unit | v0.2 outcome |
|---|---|---|
| DW1D | one hash-bound legacy run | Positive legacy evidence, with incomplete provenance |
| DW2D | one hash-bound legacy run | Positive legacy evidence, with incomplete provenance |
| MB2D | corrected reference, model seed 0 | Strong positive single-seed pilot |
| DW4 | one shared FM seed, AM seeds 1/2/3 | Conditional positive: all three AM seeds improve energy and invariant geometry |
| LJ13 | one shared FM seed, AM seeds 1/2/3 | Conditional negative: neither primary metric improves in any AM seed |
| Ala2 | fair fixed-10k single-seed comparisons | Mixed diagnostic; some observables improve and others worsen |

The corrected numerical records, training configurations, standalone figures,
and provenance are under [`results/corrected`](results/corrected). The compact
machine-readable summary is [`results/summary.json`](results/summary.json).
Detailed interpretation and claim boundaries are in
[`RESULTS.md`](RESULTS.md) and [`LIMITATIONS.md`](LIMITATIONS.md).

### Headline corrected metrics

- **MB2D, beta 1.00 -> 1.20, seed 0:** energy W2 (20k) falls from
  `0.549178` to `0.065675`; analytic-grid JS falls from `0.190643` to
  `0.008814`.
- **DW4, beta 0.80 -> 1.00:** energy W2 (20k) falls from `0.748119` to
  `0.359301 +/- 0.000981` across three AM seeds conditional on one FM seed;
  geometric W2 (2k) falls from `0.155135` to
  `0.137861 +/- 0.000054`.
- **LJ13, beta 0.80 -> 1.00:** the corrected preregistered result is negative.
  Energy W2 (2k) changes from `1.343121` to
  `1.508057 +/- 0.115961`, and geometric W2 (2k) changes from `3.005161`
  to `3.013798 +/- 0.003287`.
- **Ala2:** the best available 500 K -> 400 K single-seed case is mixed.
  Phi and Ramachandran JS improve, while psi and pooled pair-distance W2
  worsen. Rare high-energy samples dominate the unclipped energy metric.

## What changed from v0.1.0

The finite-time ULA reference arrays used for the original MB2D, DW4, and LJ13
claims were audited and rejected as equilibrium references. Version 0.2.0:

- replaces MB2D with an analytic float64 quadrature reference;
- replaces DW4 with an intrinsic-coordinate SMC/MALA reference;
- replaces LJ13 with a replica-exchange HMC reference;
- uses independent train/eval pools and protocol-matched metrics;
- quarantines the old MB2D/DW4/LJ13 conclusions at the immutable
  [`v0.1.0`](https://github.com/korarako/TAM/tree/v0.1.0) tag;
- publishes figures, metrics, configs, and hashes without model checkpoints or
  sample arrays.

The `v0.1.0` tag is historical and has not been moved.

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

The small MIT-licensed `painn-jax` implementation is vendored under
`src/painn_jax`; its license is in `THIRD_PARTY_LICENSES`.

## Basic workflow

Train a temperature-conditioned FM model and an AM refinement:

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

The exact corrected-run metadata is recorded under each system directory in
`results/corrected`. The source release deliberately contains no `.pkl`,
`.npy`, or `.npz` payloads. Checkpoint SHA-256 values are retained as
provenance, but checkpoint binaries remain outside this release.

## Result integrity

Verify the corrected result export:

```bash
cd results/corrected
sha256sum -c SHA256SUMS
```

`results/corrected/PROVENANCE.json` binds the public export to the local
research freeze. `results/corrected/MANIFEST.json` records every exported file,
size, and digest. Robust plotting windows affect display only; no finite sample
is silently dropped from the reported metrics. The clean-tree test record is
in [`VALIDATION.md`](VALIDATION.md).

## Citation

If you use TAM, cite the metadata in [`CITATION.cff`](CITATION.cff).

## License

TAM is released under the [MIT License](LICENSE). Vendored third-party code is
covered by the notices in `THIRD_PARTY_LICENSES`.
