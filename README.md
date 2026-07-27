# TAM

## Introduction

TAM is a JAX implementation of Thermodynamic Adjoint Matching for Boltzmann
generators. It first trains an inverse-temperature-conditioned Flow Matching
velocity field, then refines an anchor-to-target temperature transfer with an
energy-derived terminal adjoint. The package includes low-dimensional,
many-particle, and optional all-atom molecular systems.

## Install

Python 3.11 or newer is required.

```bash
git clone https://github.com/korarako/TAM.git
cd TAM
python -m pip install -e .
```

Optional plotting and all-atom dependencies can be installed with:

```bash
python -m pip install -e ".[plots]"
python -m pip install -e ".[ala2]"
```

## Quickstart

This self-contained DW1D example generates its own training data, trains Flow
Matching and Adjoint Matching models, and samples the adapted generator:

```bash
tam generate-data \
  --problem dw1d --beta-grid 1.25 1.30 \
  --data-dir outputs/quickstart/data \
  --n-per-beta 256 --langevin-steps 200 --seed 0

tam train-fm \
  --problem dw1d --model mlp \
  --data-dir outputs/quickstart/data --beta-grid 1.25 1.30 \
  --run-dir outputs/quickstart/dw1d \
  --samples-per-beta 256 --steps 200 --batch-size 64 \
  --hidden-dim 32 --n-layers 2 --log-every 20

tam train-am \
  --problem dw1d --model mlp \
  --run-dir outputs/quickstart/dw1d \
  --base-checkpoint outputs/quickstart/dw1d/fm_params.pkl \
  --beta0 1.25 --beta1 1.30 \
  --steps 50 --batch-size 64 --sde-steps 8 \
  --loss-steps 4 --keep-last-steps 2 \
  --am-parameterization target-refinement --log-every 10

tam sample \
  --problem dw1d \
  --checkpoint outputs/quickstart/dw1d/am_params.pkl \
  --beta 1.30 --num-samples 1000 --batch-size 250 \
  --ode-steps 50 --method heun \
  --output outputs/quickstart/dw1d/am_beta_1.30.npy
```

## Cite

Please cite the software metadata in [CITATION.cff](CITATION.cff) when using
TAM in academic work.

## License

TAM is released under the [MIT License](LICENSE). The vendored PaiNN
implementation is covered by the notice in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
