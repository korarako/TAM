# Corrected DW4 fixed-FM AM-seed aggregate

This directory aggregates one shared corrected-reference FM checkpoint
(`train seed 0`) and three independent AM refinements (`train seeds 1, 2, 3`).
It is not a three-seed end-to-end FM+AM experiment.

## Primary result

- Energy W2 (20k): shared FM `0.748119`; AM
  `0.359301 ± 0.000981`.
- Pair-distance W2 (20k): shared FM `0.055866`; AM
  `0.035973 ± 0.000096`.
- Geometric W2 (2k): shared FM `0.155135`; AM
  `0.137861 ± 0.000054`.

Standard deviations use `ddof=1` across the three AM training seeds.

## Scientific scope

The corrected reference was generated with annealed SMC in the intrinsic
six-dimensional COM-free space, systematic resampling and Metropolis-adjusted
Langevin rejuvenation. It replaces the legacy short fixed-step ULA endpoint
dataset. The SMC particles are finite and genealogically dependent, so the
reference and reference-floor values are empirical approximations rather than
analytic truth or iid Monte Carlo standard errors.

The independent Gaussian-mixture importance calculation and exact virial /
cross-beta identities audit the reference construction, but the importance
calculation has limited effective sample size and is not a second exact ground
truth.

All four controllers were evaluated with the same seed-0 chunked Gaussian
initialization: base key `7000`, chunk size
`5000`, `N=100000`. The exact array and SHA-256 are
stored as `initial_X.npy` and in `experiment_bindings.json`.

## Reference floors

- Energy W2: 0.040157 ±
  0.010158
  (30 repeats, 20k rows per split).
- Pair-distance W2: 0.003481 ±
  0.000979
  (30 repeats, 20k rows per split).
- Geometric W2: 0.128256 ±
  0.006028
  (10 repeats, 2k rows per split).

The permitted conclusion is that AM refinement is or is not consistently
better than the one shared FM baseline. Full pipeline robustness would require
independent FM training seeds.
