> **Public export note:** this directory was derived from a larger local research freeze. Model parameters, optimizer state, and all `.npy`/`.npz` sample or reference payloads are intentionally omitted here. Their identities remain available through hashes and manifests.

# DW4 corrected-reference freeze

Freeze date: 2026-07-26
Status: complete, visually inspected, and checksum-verified
Experimental unit: one shared FM seed-0 model with independent AM training seeds 1, 2, and 3; evaluation seed 0
Representative publication realization: AM seed 2

## Main outcome

The former `data/dw4_paper` arrays are not valid equilibrium references. They
were endpoint ensembles from a short fixed-step unadjusted Langevin process:

```text
N(0,I) initialization
→ 5,000 ULA steps at h=5e-5
→ total diffusion time 0.25
→ no Metropolis correction
→ coordinate clipping at ±10
→ retain only the final endpoint
```

At beta 1.00, the old reference differs materially from the corrected held-out
reference:

| Quantity | Legacy ULA | Corrected eval | Legacy-vs-corrected W2 |
|---|---:|---:|---:|
| Mean energy | -20.478638 | -21.699141 | 1.445167 |
| Mean pair distance | 1.901885 | 2.074493 | 0.386956 |
| Radius of gyration | — | — | 0.140035 |

The legacy beta-1.00 virial identity has a naive z-score of 10.48. The old
DW4 result is retained only as `legacy_ula_evidence`; it must not be used as
an equilibrium target or mixed into corrected tables.

## Target and corrected reference

DW4 contains four identical particles in two dimensions:

```text
U(R) = Σ_{i<j} [-4(d_ij - 1)^2 + 0.9(d_ij - 1)^4]
p_beta(R) ∝ exp[-beta U(R)]
```

The ambient representation is 8D. Translation is removed with a fixed
orthonormal Helmert basis, so density and reference construction are defined
on the intrinsic 6D COM-free Lebesgue space.

`reference_bundle` was produced independently for train and evaluation with:

```text
exact 6D Gaussian source
→ adaptive CESS annealing
→ systematic resampling
→ Metropolis-adjusted Langevin rejuvenation
→ exact random O(2) transformations and particle permutations
```

Configuration:

- beta values: 0.80, 1.00, 1.20;
- 100,000 samples per beta per split;
- train seed base 37001 and evaluation seed base 47001;
- target CESS 0.8;
- eight MALA steps per annealing stage and 64 final MALA steps;
- base MALA step size 0.02 with beta-dependent decay 0.75;
- reference manifest SHA256:
  `3ae2b927765c37354e591808f90b86f4eb3078a7f9fac8ded2618157b025c932`.

Independent validation:

| beta | Train/eval energy W2 | Train/eval pair W2 | Eval mean U | 5M-IS mean U |
|---:|---:|---:|---:|---:|
| 0.80 | 0.023603 | 0.001707 | -20.726019 | -20.722305 |
| 1.00 | 0.015647 | 0.000939 | -21.699141 | -21.719805 |
| 1.20 | 0.010727 | 0.001666 | -22.387299 | -22.410431 |

The exact-identity audit passed. In particular,
`beta E[x · grad U] = 6` is satisfied within the configured statistical
tolerances. The fitted cross-beta energy-density slopes are:

- beta 0.80 to 1.00: -0.196041, theory -0.2;
- beta 1.00 to 1.20: -0.199419, theory -0.2.

The corrected reference is a finite-particle Monte Carlo construction, not an
analytic iid oracle. Resampling introduces genealogical dependence. The
independent 5M importance calculation is a low-order cross-check, and its ESS
falls to about 1,523 at beta 1.20.

## Corrected formal experiment

The corrected comparison is:

```text
corrected train anchors beta={0.8, 1.2}
→ Flow Matching, seed 0, 100k updates
→ independent Adjoint Matching refinements, seeds {1,2,3}, 1k updates each
→ held-out beta=1.0 evaluation with one identical Gaussian stream
```

FM configuration:

- EGNN, hidden dimension 128, five layers;
- time embedding 16 and beta embedding 8;
- batch size 512;
- constant learning rate 3e-4;
- gradient clip 1.0;
- standard Gaussian source with scale 1.0.

AM configuration:

- batch size 512;
- learning rate 5e-7 to 5e-9;
- K=40, 20 loss steps, retain the last 10;
- max sigma 50, energy-gradient scale 1.0;
- gradient clip 1.0.

All controllers were evaluated from the same deterministic seed-0 Gaussian
initialization: base key 7000, 100,000 samples, chunks of 5,000, and Euler
integration with 150 steps. The formal energy and pair-distance protocols use
20,000 configurations. The geometric protocol uses 2,000 configurations.
Only beta 1.00 was evaluated; beta 0.80 and 1.20 are FM training anchors.

## Multi-seed result at beta 1.00

| Metric | Shared FM | AM seed1 | AM seed2 | AM seed3 | AM mean ± sample SD | Wins | Reference floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Energy W2 (20k) | 0.748119 | 0.360077 | 0.358199 | 0.359625 | 0.359301 ± 0.000981 | 3/3 | 0.040157 ± 0.010158 |
| Pair-distance W2 (20k) | 0.055866 | 0.036065 | 0.035873 | 0.035981 | 0.035973 ± 0.000096 | 3/3 | 0.003481 ± 0.000979 |
| Geometric W2 (2k) | 0.155135 | 0.137920 | 0.137813 | 0.137852 | 0.137861 ± 0.000054 | 3/3 | 0.128256 ± 0.006028 |

Mean relative reductions against the shared FM model are:

- Energy W2: 51.97%;
- Pair-distance W2: 35.61%;
- Geometric W2: 11.13%.

The reference floors compare independent corrected-SMC train and evaluation
splits under the same metric-specific sample protocol. They include
finite-particle and independent-pool discrepancies and are not iid Monte Carlo
standard errors.

The defensible conclusion is:

> Conditioned on one fixed FM seed-0 checkpoint, all three independent AM
> optimization seeds improve energy and invariant geometry under the same
> paired 100k evaluation. AM-stage variation is small, but the final energy and
> pair-distance errors remain above the corrected-reference floor.

This establishes robustness of the AM refinement stage conditional on one FM
model. It is not an end-to-end three-seed FM+AM uncertainty estimate and is
not a multi-seed significance claim over the full training pipeline.
Coordinate marginals are auxiliary because they depend on rotations,
permutations, and label conventions; energy and invariant geometric
observables are primary.

## Frozen contents

- `reference_bundle/`: all six independent corrected arrays, diagnostics,
  manifest, convergence audit, exact-identity audit, and importance audit.
- `formal_seed2_run/`: the representative complete seed-2 run, checkpoint
  hashes/bindings, 100k FM/AM samples, raw metrics, audit copies, and plots.
  Model parameter payloads are excluded.
- `am_seed_replicates/seed1/` and `seed3/`: the two additional independent AM
  refinements. Seed 2 remains in `formal_seed2_run/`.
- `aggregate/`: strict cross-seed bindings, deterministic `initial_X`, unified
  metrics, reference-floor repeats, CSV/JSON/Markdown tables, figures, and its
  own exhaustive SHA256 manifest.
- `figures/`: standalone representative and aggregate PNG/PDF panels using
  green equilibrium reference, orange shared FM, and blue AM.
- `legacy_ula_evidence/`: rejected old arrays and explicit comparisons.
- `code_snapshot/`: corrected source, run scripts, aggregation script, tests,
  requirements, and entry point.
- `provenance/`: formal and replicate logs plus reference audit logs.
- `SUMMARY.json`: machine-readable freeze summary.
- `FREEZE_SHA256SUMS.txt`: exhaustive SHA256 list for every frozen artifact
  except the manifest itself.
- `verify_freeze.ps1`: verifies hashes and rejects missing or unlisted files.

The exact seed-2 script that executed the representative realization is
`formal_seed2_run/exact_run_script.sh`. The seed-1/3 script is
`code_snapshot/scripts/run_dw4_reference_v2_am_replicate.sh`. The exact
aggregation script has SHA256
`129b81f0263378ee467d15c3bbde1e5b363dbd7655be17f1c4c1cc4fecc29425`.

Key bindings:

- corrected-reference manifest:
  `3ae2b927765c37354e591808f90b86f4eb3078a7f9fac8ded2618157b025c932`;
- shared FM checkpoint:
  `4746a1df5057301da085d20155b2e740d3d08126d1ea705863ed90cf9a61fbde`;
- shared FM samples:
  `9d65eff935bfa0311664fbdfe3286b4454fafdbd1fb79c7fead9180511dd381d`;
- deterministic evaluation initialization:
  `8c10ced112f11b6f321da61da18972250c3c68c0db1334e5b6f42c72cfec2f07`;
- aggregate manifest:
  `a37c9f01a01b953730ac8f023b4297facda142927cd8879942677aabf6c26e09`.
