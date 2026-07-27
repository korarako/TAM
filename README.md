# TAM

TAM is the research implementation of **Thermodynamic Adjoint Matching** for
Boltzmann generators. It first trains an inverse-temperature-conditioned Flow
Matching (FM) velocity field and then adapts a selected anchor-to-target
temperature transfer using an energy-derived terminal adjoint.


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


## Citation

If you use TAM, cite the metadata in [`CITATION.cff`](CITATION.cff).

## License

TAM is released under the [MIT License](LICENSE). Vendored third-party code is
covered by the notices in `THIRD_PARTY_LICENSES`.
