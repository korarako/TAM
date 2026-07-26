# Corrected ADTM result tables

This directory contains publication-ready corrected result tables in source
(`.tex`), data (`.csv`/`.json`), and rendered (`.pdf`/`.png`) formats.

## Corrections relative to the former tables

- The old MB-2 `0.75 -> 0.90` row is removed because it used the rejected
  finite-step ULA reference.
- The MB-2 `1.00 -> 1.20` row is replaced by the corrected reference-v2
  seed-0 run.
- DW-1 and DW-2 are retained but explicitly labelled hash-bound legacy.
- The old DW-4 result and reference floor are replaced by the corrected
  reference-v2 FM/AM aggregate and independent reference floors.
- External DW-4 literature rows are omitted until they can be rescored against
  the corrected target/reference under a compatible metric protocol.
- LJ-13 is omitted from the positive comparison table because the corrected
  reference-v2 downstream result is negative on both primary metrics.  The
  validated negative result is reported separately, not suppressed.
- Ala2 is omitted from positive aggregate tables because the available
  corrected audit is single-seed and mixed/negative.

## Files

- `adtm_corrected_low_dim_table.*`
- `adtm_corrected_dw4_table.*`
- `adtm_corrected_lj13_table.*`
- `adtm_corrected_low_dim_results.csv`
- `adtm_corrected_dw4_results.csv`
- `adtm_corrected_lj13_results.csv`
- `TABLE_METADATA.json`

The table images are standalone panels intended for later manuscript assembly.
