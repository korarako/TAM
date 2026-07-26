# Ala2 result summary

## Canonical protocol

All primary values below use the same canonical MD array and exactly
the same fixed 10,000 index positions for MD, FM, and AM.  No values
are filtered.  OpenMM energies are evaluated in double precision.

## Best available mixed/partial correction: 500 K -> 400 K

Single seed; 100k registered rows and a fixed fair 10k comparison.

- energy_w2_10k_kj_mol: `4.69713e+08` -> `1359.06` (improves).
- min_pair_distance_w2_10k_nm: `0.0016931` -> `0.00147757` (improves).
- pair_distance_w2_10k_nm: `0.000854543` -> `0.00110063` (worsens).
- phi_w2_10k_linear_cut_rad: `0.142898` -> `0.109137` (improves).
- psi_w2_10k_linear_cut_rad: `0.140834` -> `0.252668` (worsens).
- rama_js_10k_80bin: `0.0806832` -> `0.0784371` (improves).

This case is not a complete success.  Energy, minimum-pair, phi, and
Rama JS improve, while pooled pair-distance and psi worsen.  The
unclipped energy W2 remains dominated by rare high-energy values;
the robust energy figure reports, but does not discard, its tail.

## Negative/failure diagnostic: 900 K -> 800 K

Single seed; all 10,000 rows per method.

- energy_w2_10k_kj_mol: `925.349` -> `439.49` (improves).
- min_pair_distance_w2_10k_nm: `0.00267833` -> `0.00224099` (improves).
- pair_distance_w2_10k_nm: `0.0013437` -> `0.00156465` (worsens).
- phi_w2_10k_linear_cut_rad: `0.153965` -> `0.168174` (worsens).
- psi_w2_10k_linear_cut_rad: `0.0620231` -> `0.176903` (worsens).
- rama_js_10k_80bin: `0.113623` -> `0.113475` (improves).

Energy and minimum-pair improve, but pooled pair-distance, phi, and
psi worsen; Rama JS changes only marginally.  This is retained as a
negative/mixed diagnostic, not averaged with the best case.

## Historical native-run diagnostics

- 500 K -> 400 K native rows: `100000`.
- 900 K -> 800 K native rows: `10000`.
- Historical `ew2_2k` used 1,000 energy samples and is renamed
  `energy_w2_fixed_1k`.
- Historical Ala2 `w2` is pooled `pair_distance_w2`, not Geo W2.
- Exact historical values are in `legacy_native_metrics.json/csv`.

## Scope

- No three-seed claim.
- No claim of universally successful Ala2 temperature transfer.
- No symmetry-aware geometric W2 is reported for Ala2.
- Unclipped energy-tail values and robust-display tail masses must be
  shown together; the plotting window is not metric clipping.
