# TAM v0.2.0 corrected results

This document freezes the corrected-reference result claims. Lower
Wasserstein distances are better. The machine-readable counterpart is
[`results/summary.json`](results/summary.json); all public evidence is under
[`results/corrected`](results/corrected).

## Evidence classes

| Class | Systems | Meaning |
|---|---|---|
| Hash-bound legacy | DW1D, DW2D | Retained positive artifacts with explicit provenance limitations |
| Corrected single-seed positive pilot | MB2D | One predeclared model seed against an analytic equilibrium reference |
| Corrected conditional positive | DW4 | Three AM seeds conditional on one shared FM seed |
| Corrected conditional negative | LJ13 | Three AM seeds conditional on one shared FM seed |
| Single-seed mixed diagnostic | Ala2 | Fair fixed-10k comparison with observable-dependent outcomes |

These classes are not interchangeable. In particular, DW4 is not a
three-seed end-to-end FM+AM result, and MB2D is not a multi-seed claim.

## MB2D reference-v2

The target density is `p_beta(x) proportional to exp[-beta U(x)]`. The
reference uses float64 midpoint quadrature on a 1024x1024 grid and independent
100,000-sample train/eval pools. The released pilot uses model seed 0 and the
transfer beta 1.00 -> 1.20.

| Metric | FM | AM |
|---|---:|---:|
| Energy W2, 20k | 0.549178 | **0.065675** |
| x1 marginal W2 | 0.119442 | **0.012751** |
| x2 marginal W2 | 0.126021 | **0.016734** |
| Analytic-grid JS | 0.190643 | **0.008814** |
| Analytic-grid TV | 0.480938 | **0.085321** |
| Basin-probability L1 | 0.142364 | **0.018293** |

The independent reference-vs-reference energy-W2 floor is
`0.028405 +/- 0.003355`. AM removes most of the FM target-temperature
discrepancy, although its remaining energy W2 is above that floor.

Training settings: MLP hidden size 512 with 3 layers; FM anchors
`[0.25, 0.50, 0.75, 1.00, 1.50]`; FM 30k updates; AM 10k updates;
batch 512; AM `K=40`, 20 loss steps, last 10 retained; Euler sampling with
300 steps.

## DW4 reference-v2

The corrected reference uses intrinsic 6D SMC/MALA sampling with independent
train/eval pools. The experiment fixes one FM seed-0 controller and trains AM
with seeds 1, 2, and 3.

| Metric | Shared FM | AM mean +/- sample SD | AM wins | Reference floor |
|---|---:|---:|---:|---:|
| Energy W2, 20k | 0.748119 | **0.359301 +/- 0.000981** | 3/3 | 0.040157 +/- 0.010158 |
| Pair-distance W2, 20k | 0.055866 | **0.035973 +/- 0.000096** | 3/3 | 0.003481 +/- 0.000979 |
| Geometric W2, 2k | 0.155135 | **0.137861 +/- 0.000054** | 3/3 | 0.128256 +/- 0.006028 |

The mean reductions relative to the shared FM controller are 51.97% for
energy, 35.61% for pair distance, and 11.13% for geometric W2.

Training settings: EGNN hidden size 128 with 5 layers; FM anchors
`{0.8, 1.2}`; FM 100k updates with batch 512 and constant LR `3e-4`;
AM 1k updates per seed with batch 512 and LR `5e-7 -> 5e-9`; Euler sampling
with 150 steps.

The supported conclusion is AM-stage robustness conditional on one FM model.
Coordinate marginals are auxiliary because they depend on rotations,
permutations, and label conventions.

## LJ13 reference-v2

The corrected reference implements the BMS Eq. 234 target with
`epsilon=r_m=tau=c_osc=1` in the 36D COM-free space. It uses replica-exchange
HMC with Metropolis-corrected trajectories and swaps, four independent
ensembles, and independent 100,000-sample train/eval pools at beta
`{0.8, 1.0, 1.2}`. All reference audits pass with zero production
divergences.

The canonical scoring protocol is
`adtm.lj13_reference_v2.score2k.v1`: energy, pair distance, radius of
gyration, minimum-pair distance, and geometric W2 all use the same 2,000 rows
without filtering.

| Metric | Shared FM | AM mean +/- sample SD | AM wins |
|---|---:|---:|---:|
| Energy W2, 2k | **1.343121** | 1.508057 +/- 0.115961 | 0/3 |
| Geometric W2, 2k | **3.005161** | 3.013798 +/- 0.003287 | 0/3 |
| Pair-distance W2, 2k | **0.040800** | 0.042236 +/- 0.003034 | 1/3 |
| Radius-of-gyration W2, 2k | **0.027494** | 0.028268 +/- 0.002072 | 1/3 |
| Minimum-pair W2, 2k | 0.012508 | **0.010450 +/- 0.000576** | 3/3 |

None of the three AM seeds improves either preregistered primary metric.
Minimum-pair distance improves in all three seeds, but this auxiliary result
does not change the registered negative verdict. This is a conditional
negative result, not a general impossibility claim.

## Ala2

Ala2 uses the same canonical MD array and the same deterministic 10,000 index
positions for MD, FM, and AM. No sample is filtered, and OpenMM energy is
evaluated in double precision. Robust plotting limits affect visualization
only. No symmetry-aware geometric W2 is reported.

### Best available mixed case: 500 K -> 400 K, seed 2

| Metric | FM | AM | Direction |
|---|---:|---:|---|
| Energy W2, kJ/mol | 4.69713e8 | 1359.06 | Improves; tail dominated |
| Minimum-pair W2, nm | 0.00169310 | 0.00147757 | Improves |
| Pair-distance W2, nm | 0.000854543 | 0.001100629 | Worsens |
| Phi W2, rad | 0.142898 | 0.109137 | Improves |
| Psi W2, rad | 0.140834 | 0.252668 | Worsens |
| Ramachandran JS | 0.0806832 | 0.0784371 | Slightly improves |

### Negative/mixed diagnostic: 900 K -> 800 K

| Metric | FM | AM | Direction |
|---|---:|---:|---|
| Energy W2, kJ/mol | 925.349 | 439.490 | Improves |
| Minimum-pair W2, nm | 0.00267833 | 0.00224099 | Improves |
| Pair-distance W2, nm | 0.00134370 | 0.00156465 | Worsens |
| Phi W2, rad | 0.153965 | 0.168174 | Worsens |
| Psi W2, rad | 0.0620231 | 0.176903 | Worsens |

The supported conclusion is mixed single-seed evidence. Energy and local
contact diagnostics can improve while structural observables worsen.

## Retained DW1D/DW2D legacy evidence

| System | Energy W2 FM -> AM | Geometric W2 FM -> AM |
|---|---:|---:|
| DW1D | 0.604278 -> 0.184344 | 0.246554 -> 0.078003 |
| DW2D | 1.940773 -> 0.151375 | 0.715530 -> 0.156857 |

DW1D anchor arrays contain only 16 samples and have incomplete generation
provenance. DW2D also has incomplete generation provenance. These rows are
retained as hash-bound legacy evidence, not newly corrected multi-seed
benchmarks.

## Superseded v0.1 results

The original MB2D, DW4, and LJ13 values were computed against rejected
finite-time ULA/biased references. They are preserved only at the immutable
[`v0.1.0`](https://github.com/korarako/TAM/tree/v0.1.0) tag and must not be
quoted as corrected equilibrium results.
