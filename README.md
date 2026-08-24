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

## Evaluation

Geometric Wasserstein-2 evaluation exposes the particle-alignment convention
through `--geometric-protocol`:

- `auto` selects exact permutation enumeration for small particle systems and
  symmetry-consistent iterative joint alignment for larger systems such as
  LJ-13. This is the recommended default.
- `exact` enumerates particle permutations and is intended only for systems
  small enough for the factorial cost to be tractable.
- `joint` computes the complete all-pairs ground-cost matrix using iterative
  joint particle and rigid alignment. It is symmetry-consistent but its
  particle alignment is approximate.
- `sequential` first performs raw-coordinate Hungarian particle matching and
  then rigid alignment. It is provided for comparison with that convention and
  is not invariant to independent rotations of the inputs.
- `legacy-topk` reproduces the mixed proxy/top-k refinement estimator used in
  TAM 0.2.1 and earlier. Use it only when auditing historical output.

For example, evaluate 2,000 LJ-13 generated and reference configurations with
the recommended protocol using:

```bash
tam evaluate \
  --problem lj13 \
  --samples outputs/lj13/generated.npy \
  --reference data/lj13/reference.npy \
  --metric-samples 2000 \
  --geometric-protocol auto \
  --geometric-workers 8 \
  --geometric-parallel-backend process \
  --output outputs/lj13/evaluation.json
```

The JSON output records `geometric_protocol_requested` and per-run
`geometric_w2_metadata`, including the resolved protocol, ground-cost and outer
transport definitions, subsampling rule and seed, sample count, candidate
truncation, parallel backend, worker count, and whether particle alignment is
approximate. Preserve this metadata when reporting results.

Geometric W2 values produced by different protocols are different estimators.
In particular, results from `auto`/`joint` in TAM 0.3.0 are not directly
comparable to historical TAM 0.2.1-and-earlier `legacy-topk` values; recompute
all compared methods under the same protocol and sampling setup.

The full LJ-13 cost matrix contains `N^2` pair alignments (four million at the
default 2,000 samples). When using multiple workers, set the BLAS thread-count
variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS`) to
`1` to avoid nested oversubscription.

The process backend is fastest for formal command-line runs. When calling the
Python API with multiple process workers, protect the entry point with
`if __name__ == "__main__":`, as required by Python multiprocessing. Use the
thread backend in notebooks or scripts where that guard is unavailable.

## Cite

Please cite the software metadata in [CITATION.cff](CITATION.cff) when using
TAM in academic work.

## License

TAM is released under the [MIT License](LICENSE). The vendored PaiNN
implementation is covered by the notice in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
