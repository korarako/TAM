# TAM v0.2.0

TAM v0.2.0 is the corrected-reference research release of Thermodynamic
Adjoint Matching.

## Release boundary

| System | Status |
|---|---|
| DW1D, DW2D | Accepted hash-bound legacy positive evidence |
| MB2D | Corrected single-seed positive pilot |
| DW4 | Corrected conditional positive result |
| LJ13 | Corrected conditional negative result |
| Ala2 | Fair fixed-10k single-seed mixed diagnostic |

The old MB2D, DW4, and LJ13 values are preserved only at
[`v0.1.0`](https://github.com/korarako/TAM/tree/v0.1.0). That tag has not been
moved or overwritten.

## Included

- the TAM training, sampling, and evaluation implementation;
- corrected-reference result summaries and standalone figures;
- exact frozen training/evaluation configurations and scripts;
- corrected reference-generation code snapshots and audits;
- source checkpoint/sample SHA-256 bindings;
- publication-ready tables in PNG, PDF, TeX, and CSV;
- a release-level manifest and checksum list;
- tests and third-party license notices.

## Excluded

The v0.2 source tree intentionally excludes:

- model or optimizer state (`.pkl`, `.ckpt`, `.pt`, `.pth`,
  `.safetensors`);
- generated sample/reference arrays (`.npy`, `.npz`);
- workstation-specific absolute paths;
- the rejected v0.1 MB2D/DW4/LJ13 result records from the current branch.

The immutable v0.1 tag preserves the historical release exactly.

## Verification

Verify the public corrected export:

```bash
cd results/corrected
sha256sum -c SHA256SUMS
```

Run the source tests:

```bash
python -m pip install -e ".[test]"
pytest
```

The corrected export is bound to the research freeze by
`results/corrected/PROVENANCE.json`.

## Scientific interpretation

The corrected evidence demonstrates that energy-based AM refinement can
substantially improve an FM proposal, but the gain is system- and
observable-dependent. MB2D and DW4 provide positive evidence at different
strengths; LJ13 is a conditional negative result; Ala2 is mixed.

See [`RESULTS.md`](RESULTS.md) for values and [`LIMITATIONS.md`](LIMITATIONS.md)
for the exact claim boundary.
