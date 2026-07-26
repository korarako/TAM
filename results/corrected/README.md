# TAM v0.2 corrected result export

This directory is the checkpoint- and sample-free public export of the
2026-07-26 corrected research freeze.

- `mb2d/`: corrected analytic-reference single-seed positive pilot.
- `dw4/`: corrected conditional positive result (one FM seed, three AM seeds).
- `lj13/`: corrected conditional negative result (one FM seed, three AM seeds).
- `ala2/`: fair fixed-10k single-seed mixed diagnostics.
- `publication_tables/`: standalone CSV/TeX/PDF/PNG tables.

`MANIFEST.json` and `SHA256SUMS` verify this exported tree.
`PROVENANCE.json` binds it to the full local research freeze. Model
checkpoints, optimizer states, and sample/reference arrays are intentionally
excluded. Machine-local paths in text records are replaced by symbolic roots.

Use `../summary.json`, `../../RESULTS.md`, and `../../LIMITATIONS.md` for the
release-level scientific interpretation.
