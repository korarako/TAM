# v0.2.0 validation

Validation date: 2026-07-26

## Source and artifact checks

- Corrected public export: 308 files, 10,849,042 bytes.
- `results/corrected/MANIFEST.json`: every listed file verified.
- Forbidden checkpoint/sample extensions in the source release: 0.
- Machine-local absolute paths in exported evidence: 0.
- JSON parsing: pass.
- YAML parsing: pass.
- Python compile check for `src`, `scripts`, and `tests`: pass.
- `git diff --check`: pass.

## Test suite

The checkpoint/sample-free tree was copied to a clean temporary directory on
the research host and tested without installing or importing the original
workspace:

```text
Python: 3.11.15
JAX: 0.4.38
pytest: 9.1.1
platform: CPU
result: 35 passed
```

No training, sampling, or reference generation was run during this validation.
