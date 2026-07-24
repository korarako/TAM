# TAM v0.1.0 results

This document freezes the result claims for the first public TAM release. It
reports both improvements and regressions. Lower Wasserstein distances are
better.

The machine-readable counterpart is [`results/summary.json`](results/summary.json).
Path-sanitized public records are under [`results/public/`](results/public/).
Machine-local extraction records are retained outside the public Git tree for
provenance; the released aggregates preserve their scientific values and run
semantics without exposing workstation paths.

## Result status

| Status | Systems | Release claim |
| --- | --- | --- |
| **Positive** | DW1D, DW2D, DW4 | Released AM result improves the primary energy and geometry metrics. |
| **Mixed** | MB2D, LJ13 | Improvement depends on transfer region or metric; no blanket improvement claim is made. |
| **Experimental failed case** | Ala2 | Representative checkpoints are released for diagnosis, not as a successful benchmark. |

## Metric protocol

The primary benchmark metrics are energy W2 and geometric W2 on fixed
2,000-sample comparisons. All toy and MB2D values below are single released
runs and therefore have no between-training-run uncertainty estimate.

DW4 reports three independent formal AM training runs as mean ± sample standard
deviation (`ddof=1`). LJ13 instead reports three sampling/evaluation seeds from
one shared FM/AM checkpoint; its variation must not be interpreted as
between-training-run uncertainty.

## Positive results

### DW1D and DW2D

| System | Transfer β0→β1 | FM energy W2 | AM energy W2 | FM geometry W2 | AM geometry W2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DW1D | 0.25→0.70 | 0.604278 | **0.184344** | 0.246554 | **0.078003** |
| DW2D | 0.25→0.70 | 1.940773 | **0.151375** | 0.715530 | **0.156857** |

Both released transfers improve energy and geometry. Because each row is one
released training run, these results establish reproducible artifacts rather
than a multi-seed estimate of method-level variability.

### DW4

The DW4 comparison uses the shared FM checkpoint and three independently
trained formal AM checkpoints.

| Metric | FM mean ± std | AM mean ± std | AM better in runs |
| --- | ---: | ---: | ---: |
| Energy W2 (2k) | 0.525119 ± 0.000000003 | **0.180243 ± 0.042294** | 3/3 |
| Geometric W2 (2k) | 0.183385 ± 0.002630 | **0.173944 ± 0.011311** | 3/3 |
| Pairwise-distance W2 | 0.038414 ± 0.00000000004 | **0.019552 ± 0.003127** | 3/3 |

The representative release checkpoint is seed 2, but the claim is based on all
three formal runs. Full values and replicate semantics are in
[`results/dw4/aggregate.json`](results/dw4/aggregate.json).

## Mixed results

### MB2D

| Transfer β0→β1 | FM energy W2 | AM energy W2 | FM geometry W2 | AM geometry W2 | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.25→0.30 | **0.473743** | 0.749420 | 0.111394 | **0.097560** | Energy worse; geometry better |
| 0.50→0.60 | **0.137678** | 0.314678 | 0.101828 | **0.094731** | Energy worse; geometry better |
| 0.75→0.90 | 0.213760 | **0.140192** | 0.103759 | **0.089678** | Both better |
| 1.00→1.20 | 0.210264 | **0.143947** | 0.208112 | **0.077177** | Both better |

AM improves geometry in all four released transfers, but energy W2 improves
only in the two higher-β transfers. The release therefore makes a
transfer-dependent, mixed claim for MB2D.

### LJ13

All three rows below use the same FM and AM checkpoint. They differ only in
sampling/evaluation seed.

| Eval seed | FM energy W2 | AM energy W2 | FM pairwise W2 | AM pairwise W2 | FM radial W2 | AM radial W2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.365277 | **0.295730** | 0.008771 | **0.006270** | 0.007414 | **0.006105** |
| 202 | **0.173496** | 0.561391 | 0.005846 | **0.004525** | 0.006305 | **0.005899** |
| 303 | **0.584218** | 0.585974 | 0.008593 | **0.006872** | 0.007232 | **0.006597** |

Pairwise and radial W2 improve in all three evaluations. Energy W2 improves for
seed 101, worsens strongly for seed 202, and is slightly worse for seed 303.
The filtered energy mean is `0.374330 ± 0.205511` for FM and
`0.481031 ± 0.160946` for AM. Seed 303 removes one extreme sample from both FM
and AM under the declared `max pair distance < 10 nm` filter.

The appropriate conclusion is consistent small geometric improvement but
unstable energy performance. See
[`results/lj13/aggregate.json`](results/lj13/aggregate.json).

## Experimental failed case: Ala2

Ala2 is intentionally excluded from the successful aggregate. Its legacy
fixed-sample energy W2 values use **1,000 samples**, despite the historical
field name `ew2_2k`.

| Attempt | Transfer | Rama JS FM→AM | Energy W2 FM→AM | Energy q99 FM→AM | Reference q99 | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| grid35_t400 | 500 K→400 K | 0.02822→0.02512 | 53.43→29.71 | 161.42→114.17 | 3.44 | Some central metrics improve; ψ W2 and the energy tail remain problematic. |
| grid3579_t400 | 500 K→400 K | 0.08436→0.07996 | 36.60→30.35 | 128.83→45.83 | 3.54 | Balanced central improvement, but the energy tail remains far too heavy. |
| grid3579_t600 | 700 K→600 K | 0.09902→0.09900 | 41.10→46.16 | 257.59→202.04 | 74.29 | Structural result is effectively null and energy W2 worsens. |
| grid3579_125k_t800 | 900 K→800 K | 0.11047→0.11050 | 22850.72→66.79 | 515.62→372.60 | 146.49 | Large energy/torsion correction, but no Rama JS improvement and a heavy tail remains. |

The frozen conclusion is:

> The released Ala2 experiments do not demonstrate a successful AM transfer.
> Some individual energy or torsion metrics improve, but no attempt resolves
> the high-energy tail while consistently improving the structural
> distribution.

Representative path-sanitized values are in
[`results/public/ala2/representative_attempts.json`](results/public/ala2/representative_attempts.json).
The full historical inventory remains an archival diagnostic record and must
not be used for best-run selection without accounting for exploration bias.

## Release boundary

The v0.1.0 claims are therefore:

- **Positive:** DW1D, DW2D, and the three-run DW4 result.
- **Mixed:** MB2D and LJ13.
- **Experimental failed case:** Ala2.

No result from an exploratory best-checkpoint scan is included in the formal
aggregate, and no clipped, filtered, or selected result is silently presented
as an unfiltered multi-seed benchmark.
