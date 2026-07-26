# Limitations

TAM v0.2.0 is a research release. Its evidence does not support a claim that
Adjoint Matching always improves a pretrained Flow Matching proposal.

## Evidence units differ

- MB2D is one predeclared model seed.
- DW4 uses one shared FM seed and three independent AM optimization seeds.
- LJ13 uses one shared FM seed and three independent AM optimization seeds.
- Ala2 is a fair fixed-10k single-seed diagnostic.
- DW1D and DW2D are retained hash-bound legacy runs.

These repetitions must not be pooled or described as the same form of
multi-seed uncertainty. DW4 establishes robustness of the AM stage conditional
on one FM controller, not robustness of the full FM+AM pipeline.

## Corrected references changed the conclusions

The v0.1 MB2D, DW4, and LJ13 values were tied to finite-time ULA/biased
references that failed later equilibrium audits. Those values remain
available at the immutable `v0.1.0` tag for history only.

Version 0.2 uses:

- analytic float64 quadrature for MB2D;
- intrinsic-coordinate SMC/MALA for DW4;
- replica-exchange HMC for LJ13.

Reference quality is therefore part of the result, not a cosmetic evaluation
choice.

## The method is not uniformly improving

- Corrected MB2D is a strong positive single-seed pilot.
- Corrected DW4 improves all three registered metrics for every AM seed,
  conditional on one FM model.
- Corrected LJ13 improves neither preregistered primary metric in any AM seed.
- Ala2 improves some energy, contact, and torsion diagnostics while worsening
  other structural observables.

Energy and geometry can move in opposite directions. Conclusions must be
reported by system, observable, and evidence unit.

## Ala2 is mixed and tail dominated

The canonical Ala2 comparison uses identical deterministic 10,000-row subsets
for MD, FM, and AM without filtering. Rare high-energy samples dominate the
unclipped energy W2. Robust energy-axis limits affect display only.

Pooled pair-distance W2 is not a symmetry-aware geometric W2. No Ala2
geometric-W2 claim is made. The released Ala2 evidence is single-seed and must
not be described as a validated general temperature-transfer result.

## Legacy DW1D and DW2D

DW1D and DW2D retain positive hash-bound results, but their generation
provenance is incomplete. DW1D anchor arrays contain only 16 samples. These
rows support retained artifact-level evidence, not a corrected multi-seed
benchmark.

## Metric protocols are fixed

Metric sample counts are part of the protocol:

- MB2D primary energy uses 20k rows.
- DW4 energy/pair distance use 20k rows and geometric W2 uses 2k.
- LJ13 uses the same 2k rows for every reported observable without filtering.
- Ala2 uses the same fixed 10k indices for MD/FM/AM.

Changing the reference, subset, seed, sample count, clipping, filtering, or
integration settings creates a different result.

## No checkpoint or sample payloads in v0.2

The v0.2 source release contains metrics, figures, configurations, scripts,
and provenance hashes, but no model parameters or sample arrays. The binaries
remain in the private research archive and are bound by SHA-256 values. The
public release therefore supports evidence inspection and method-level
reproduction, not byte-identical checkpoint recovery.

## Numerical and software scope

TAM uses finite-step ODE/SDE and Monte Carlo procedures. The release records
the tested settings but does not certify arbitrary step sizes, hardware,
topologies, or distributed configurations. Accelerator kernels may prevent
bitwise-identical retraining.
