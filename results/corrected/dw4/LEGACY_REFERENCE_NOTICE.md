# Legacy DW4 reference notice

`legacy_ula_evidence/dw4_paper_fixed_step_ula/` is retained only to make the
correction auditable. It is not equilibrium training or evaluation data.

The files came from short, fixed-step, unadjusted Langevin endpoints and fail
independent equilibrium checks. All earlier DW4 checkpoints, tables and plots
trained or evaluated against `data/dw4_paper` must be labeled
`legacy biased-reference result`.

Use only:

```text
reference_bundle/train/samples_beta_*.npy
reference_bundle/eval/samples_beta_*.npy
```

for corrected DW4 work.
