# TAM corrected result freeze — 2026-07-26

This directory is the checkpoint-free scientific result freeze assembled after
the MB2D, DW4, and LJ13 reference audits.

It is **not** a relabeling of the existing public `v0.1.0` release. The public
source identity is recorded separately:

```text
repository: https://github.com/korarako/TAM.git
branch: main
tag: v0.1.0
HEAD/origin/main/tag commit:
57f6a2c17c00276e5604ecad2933c4a7c06683ac
```

The local source checkout was clean and synchronized with `origin/main` when
this freeze was assembled. A future public corrected release should use a new
commit and tag; it must not overwrite `v0.1.0`.

## Result status

| System | Status in this freeze | Allowed interpretation |
|---|---|---|
| MB2D reference-v2 | Validated corrected single-seed pilot | Correct analytic/quadrature equilibrium reference; FM-to-AM improvement at beta 1.20. Not yet an end-to-end multi-seed claim. |
| DW4 reference-v2 | Validated corrected aggregate | Independent corrected train/eval reference; one shared FM seed and three independent AM seeds, all improving energy and invariant pair geometry. |
| DW1D | Accepted legacy, hash-bound | Released result retained with explicit provenance/sample-size limitations. |
| DW2D | Accepted legacy, hash-bound | Released result retained with explicit provenance limitations. |
| Ala2 | Single-seed mixed/negative diagnostics | Fair fixed-10k figures and metrics use the same MD and indices without filtering. The best case is mixed/partial; it is not a general positive transfer claim. |
| LJ13 reference-v2 + ADTM | Validated reference; conditional negative result | Exact BMS Eq. 234 target in the 36D COM-free space. Three AM seeds sharing one FM seed fail to improve both primary metrics. |

## Key corrected results

### MB2D, beta 1.20

One predeclared seed (`seed=0`), evaluated against an independent 100k
reference-v2 split:

| Metric | FM | AM |
|---|---:|---:|
| Energy W2, 20k | 0.549178 | 0.065675 |
| x1 marginal W2 | 0.119442 | 0.012751 |
| x2 marginal W2 | 0.126021 | 0.016734 |
| Analytic-grid JS | 0.190643 | 0.008814 |
| Analytic-grid TV | 0.480938 | 0.085321 |

The independent 20k reference-vs-reference energy-W2 floor is
`0.028405 ± 0.003355`.

### DW4, beta 1.00

One shared FM seed-0 controller and independent AM seeds 1, 2, and 3:

| Metric | Shared FM | AM mean ± sample SD |
|---|---:|---:|
| Energy W2, 20k | 0.748119 | 0.359301 ± 0.000981 |
| Pair-distance W2, 20k | 0.055866 | 0.035973 ± 0.000096 |
| Geometric W2, 2k | 0.155135 | 0.137861 ± 0.000054 |

All three AM seeds improve the shared FM result. This supports robustness of
the AM stage conditional on one FM model, not full end-to-end seed uncertainty.

### Released DW1D/DW2D legacy numbers

| System | Energy W2 FM -> AM | Geometric W2 FM -> AM |
|---|---:|---:|
| DW1D | 0.604278 -> 0.184344 | 0.246554 -> 0.078003 |
| DW2D | 1.940773 -> 0.151375 | 0.715530 -> 0.156857 |

Read the `dw1d` and `dw2d` `claim_limit` fields in
`TAM_RESULT_STATUS.json` before citing these values.

### LJ13 corrected-reference ADTM result

The corrected reference uses replica-exchange HMC with Metropolis-corrected
trajectories and swaps, four independent ensembles, and separate train/eval
pools. At beta 0.80, 1.00, and 1.20 it stores 100,000 samples per beta and
split. Both split audits and the train-vs-eval audit pass with zero production
divergences. The largest train-vs-eval absolute z-score is 1.214682.

The corrected downstream experiment uses one shared FM seed 0 and independent
AM seeds 1, 2, and 3. All primary observables use the same canonical 2k subset
without filtering:

| Metric | FM | AM mean +/- sample SD | AM wins |
|---|---:|---:|---:|
| Energy W2 | 1.343121 | 1.508057 +/- 0.115961 | 0/3 |
| Geometric W2 | 3.005161 | 3.013798 +/- 0.003287 | 0/3 |
| Pair-distance W2 | 0.040800 | 0.042236 +/- 0.003034 | 1/3 |
| Radius-of-gyration W2 | 0.027494 | 0.028268 +/- 0.002072 | 1/3 |
| Minimum-pair W2 | 0.012508 | 0.010450 +/- 0.000576 | 3/3 |

The registered verdict is conditional negative: none of the AM seeds improves
either primary metric. The auxiliary minimum-pair improvement does not change
that verdict. Earlier experiments using quarantined ULA arrays remain invalid.

### Ala2 existing-result audit

The authoritative evaluation is a fixed fair 10k comparison using one
canonical MD array, identical deterministic indices for MD/FM/AM, no
filtering, and double-precision OpenMM energies. At 500 K -> 400 K, AM improves
energy, minimum-pair distance, phi, and Rama JS but worsens pooled
pair-distance and psi; the energy W2 remains dominated by rare high-energy
values. The 900 K -> 800 K case is retained as a negative/mixed diagnostic.
Neither case supports a multi-seed or uniformly positive Ala2 claim.

## Directory guide

```text
mb2d_reference_v2/      Corrected reference, run, figures, audits, code snapshot
dw4_reference_v2/       Corrected reference, runs, aggregate, figures, audits
ala2_adtm_extended/     Fair fixed-10k mixed/negative Ala2 evidence
lj13_reference_v2/      Validated RE-HMC reference, audits, traces, source snapshot
lj13_adtm_reference_v2/ Corrected-reference 3-AM-seed conditional negative result
publication_tables/     Standalone corrected tables, metadata, PNG, and PDF
TAM_RESULT_STATUS.json  Cross-system status, including hash-bound DW1D/DW2D
LJ13_REFERENCE_AUDIT.md Audit and quarantine decision for superseded LJ13 arrays
MANIFEST.json           Definitive package file manifest
SHA256SUMS.txt           Definitive package checksum list
VALIDATION.json         Machine-readable package validation report
```

DW1D and DW2D are retained only as compact, hash-bound legacy result records in
`TAM_RESULT_STATUS.json` and the low-dimensional publication table. Their
original sample/checkpoint payloads are intentionally not duplicated here.

## Checkpoint exclusion

This package intentionally contains no model parameters or optimizer state:

```text
*.pkl
*.ckpt
*.pt
*.pth
*.safetensors
```

Paths representing checkpoint, optimizer, parameter, or weight payloads were
also excluded. Some nested README files and source-freeze manifests document
hashes or filenames of omitted checkpoints; those textual references are
provenance only. The only authoritative integrity records for this filtered
package are the top-level `MANIFEST.json`, `SHA256SUMS.txt`, and
`VALIDATION.json`.

## Scientific exclusions

- Do not use the old MB2D finite-time ULA dataset as equilibrium reference.
- Do not use the old DW4 finite-time ULA dataset as equilibrium reference.
  Copies under `legacy_*_evidence` are audit evidence only.
- Do not use the superseded LJ13 ULA arrays or their derived metrics.
- The new LJ13 reference-v2 and corrected downstream rerun are validated; old
  FM/AM experiments on quarantined ULA arrays remain invalid.
- Do not promote the Ala2 mixed/negative single-seed cases to a uniform or
  statistically general positive result.

## Visual style

Standalone panels are preserved rather than preassembled paper composites.
Where applicable:

```text
equilibrium/MD reference: green
FM or proposal:           orange
AM or reweighted output:  blue
analytic exact target:    black
```
