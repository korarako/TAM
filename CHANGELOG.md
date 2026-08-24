# Changelog

## 0.3.0 - 2026-08-24

### Changed

- Corrected geometric Wasserstein-2 evaluation by making the particle-alignment
  protocol explicit and using full symmetry-consistent joint alignment for
  larger particle systems under the recommended `auto` protocol.
- Added `exact`, `joint`, `sequential`, and deprecated `legacy-topk` evaluation
  modes, together with machine-readable protocol metadata in evaluation output.
- Retained `legacy-topk` solely for reproducing or auditing TAM 0.2.1-and-earlier
  metric output. Historical geometric W2 values must not be compared directly
  with 0.3.0 `auto`/`joint` results.
- Preserved the 0.2.1 single-generator subsampling order so DW-4 and
  non-particle evaluations do not change solely because of sample selection.
- Added regression tests and CI coverage for symmetry invariance, repeated
  distance signatures, exact DW-4 alignment, input validation, and parallel
  cost-matrix execution.

This release changes evaluation only. Training algorithms, model architectures,
sampling behavior, checkpoint formats, and existing trained checkpoints are
unchanged; retraining is not required. Existing checkpoints can be reevaluated
with the corrected metric.
