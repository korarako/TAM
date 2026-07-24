# Public result records

This directory contains path-sanitized, compact result records intended for
the public release. They contain no machine-specific absolute paths.

The original extraction records remain elsewhere under `results/` as immutable
provenance artifacts. In particular, the original DW4, LJ13, and Ala2
historical files may contain absolute paths from the research workstation.
Those paths are not required to interpret the public records.

The authoritative release-level classification and cross-system metrics are in
`results/summary.json`. Detailed aggregate semantics are in
`results/dw4/aggregate.json` and `results/lj13/aggregate.json`.
