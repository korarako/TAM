# Extended ADTM Ala2 frozen results

This checkpoint-free bundle preserves two single-seed Ala2 FM-to-AM cases:

1. a best available **mixed/partial correction** at 500 K -> 400 K.  Its
   registered source arrays contain 100,000 configurations per series, while
   the canonical comparison uses the same deterministic 10,000 index
   positions for MD, FM, and AM;
2. a 900 K -> 800 K **negative/failure diagnostic**, evaluated with 10,000
   configurations per method using all rows.

It does not establish full Ala2 success and does not establish three-seed
robustness.  In particular, the best case retains a serious high-energy tail
and worsens the psi marginal.

## Authoritative evaluation

`comparison_metrics.json/csv` is the authoritative **fair fixed-10k**
evaluation.  Both cases use one canonical MD array, exactly the same index
positions for MD/FM/AM, no row filtering, and double-precision OpenMM energy
evaluation.  The energy W2 includes every selected value and is therefore
tail-sensitive.  The robust plotting window affects display only; its omitted
tail mass is recorded explicitly.

Historical native-run metrics are retained separately as
`legacy_native_metrics.json/csv`.  They use the original run-specific
evaluation sizes (100k or 10k, plus a fixed-1k energy diagnostic) and are not
substituted for the fair fixed-10k result.

## Metric corrections

- Historical Ala2 `w2` is pooled **pair-distance W2**.  It is not geometric
  W2, and this bundle never labels it geometric W2.
- Historical `ew2_2k` was computed with `energy_w2_samples=1000`; this bundle
  calls it `energy_w2_fixed_1k`.

## Payload policy

No `.pkl`, `.npy`, `.npz`, `.ckpt`, `.pt`, `.pth`, or `.safetensors` payload
is copied.  `artifact_bindings.json` records their source paths, sizes,
SHA-256 digests, and (for NumPy arrays) shapes and dtypes.

See `RESULTS.md` for the scientific summary, `comparison_metrics.json/csv`
for canonical values, `legacy_native_metrics.json/csv` for historical
diagnostics, `plots/` for standalone figures, and `SHA256SUMS` for integrity.
