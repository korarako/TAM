> **Public export note:** this directory was derived from a larger local research freeze. Model parameters, optimizer state, and all `.npy`/`.npz` sample or reference payloads are intentionally omitted here. Their identities remain available through hashes and manifests.

# MB2D reference-v2 freeze

Status: **completed single-seed pilot** (`seed=0`). This freeze replaces the
legacy fixed-step ULA datasets for MB2D evaluation. It is not yet a multi-seed
publication aggregate.

## Correct equilibrium reference

The target is

```text
p_beta(x) = exp(-beta * U(x)) / Z_beta
U(x) = 0.02 * scaled Mueller--Brown potential
```

The reference bundle uses float64 midpoint quadrature, stable log-sum-exp
normalization, categorical cell sampling and uniform within-cell jitter. It
does not use the legacy finite-time ULA endpoint sampler.

- betas: `0.25, 0.50, 0.75, 1.00, 1.20, 1.50`
- domain: `x1=[-4.0,2.5]`, `x2=[-2.5,4.5]`
- nested grids: `256 -> 512 -> 1024`
- train split: 100,000 samples/beta, seed base 17001
- eval split: 100,000 samples/beta, seed base 27001
- worst domain-outside mass: `2.5457e-9`
- worst adjacent-grid TV: `7.2488e-4`
- strict audit: PASS
- bundle manifest SHA256:
  `8f8d27311420b47833fe7ccbc7a1d831b851cb327731b93a99479c285069fbfc`

All 33 files listed by `reference_bundle/SHA256SUMS` were rechecked locally:
zero failures.

## Frozen training setup

- FM anchors: `[0.25, 0.50, 0.75, 1.00, 1.50]`
- transfer: `beta0=1.00 -> beta1=1.20`
- model: MLP, hidden 512, 3 layers, t embedding 16, beta embedding 8
- FM: 30,000 updates, batch 512, cosine LR `1e-3 -> 1e-5`
- AM: 10,000 updates, batch 512, LR `2e-5 -> 2e-7`
- AM rollout: `K=40`, loss steps 20, keep last 10, `max_sigma=50`
- prior scale 1, energy-gradient scale 1, gradient clip 1
- sampling: 100,000 samples, Euler ODE, 300 steps
- model seed: 0
- training data: `mb2d_reference_v2/train`
- evaluation data: independent `mb2d_reference_v2/eval`

Final training log:

```text
FM step 30000: loss=1.108674, lr=1e-5
AM step 10000: loss=0.435274, lr=2e-7
```

Checkpoint hashes:

```text
FM cb18f47fb2106612b0cde0611eebc6090f8d0524b71982a416a4ddf22375ebf5
AM 650d073f61d93fc0234d6c3593b5e9075f3db47ee6da77ce53f005396cbe6fef
```

The exact commands are frozen in
`seed0_run/exact_run_script.sh`. The complete FM configuration is in
`seed0_run/config.yaml`.

## Main result at beta=1.20

Against the independent 100k eval split:

| metric | FM | AM |
|---|---:|---:|
| energy W2, 20k | 0.549178 | 0.065675 |
| x1 marginal W2 | 0.119442 | 0.012751 |
| x2 marginal W2 | 0.126021 | 0.016734 |
| analytic-grid JS | 0.190643 | 0.008814 |
| analytic-grid TV | 0.480938 | 0.085321 |
| basin-probability L1 | 0.142364 | 0.018293 |
| analytic-grid energy W2 | 0.555328 | 0.056256 |

The independent reference-vs-reference **20k** energy-W2 baseline is
`0.028405 +/- 0.003355`. AM removes about 88% of the FM energy-W2 error, but
remains measurably above this finite-sample reference floor.

At the anchor beta=1.00, FM already matches the corrected equilibrium
distribution well: 20k energy W2 `0.045778`, analytic-grid JS `0.008710`.

## Legacy biased-data audit

The old finite-time ULA reference is not equilibrium. At beta=1.20:

| legacy object | analytic-grid JS | energy W2, 20k |
|---|---:|---:|
| old FM | 0.174402 | 0.405519 |
| old AM | 0.053583 | 0.229982 |
| old ULA “reference” | 0.051714 | 0.224319 (grid quantiles) |

The old AM improved over the old FM, but it learned a distribution very close
to the biased ULA reference. These values are retained only as a documented
failure/audit baseline.

## Directory contents

- `reference_bundle/`: equilibrium grids, independent samples, manifest,
  convergence audit and SHA256SUMS.
- `seed0_run/`: checkpoint hashes/bindings, 100k samples, metrics, exact
  command, and publication figures. Model parameter payloads are excluded.
- `legacy_audit/`: old sample re-evaluation against the analytic target.
- `logs/`: reference generation and complete pipeline logs.
- `code_snapshot/`: exact source files used for this correction.
- `SUMMARY.json`: machine-readable result summary.

The publication heat maps are standalone panels with one distribution per
image. One-dimensional overlays use:

```text
exact target: black
equilibrium reference: green
FM: orange
AM: blue
```

## Interpretation and next gate

This seed is a successful correctness pilot: correcting the dataset does not
destroy the FM+AM result; instead, AM closes most of the cross-temperature gap.
It is not exact, as the fair 20k reference baseline shows. For a final paper
claim, rerun the identical frozen protocol for two additional predeclared model
seeds and report mean plus uncertainty. Do not select the best seed.
