# Limitations

TAM v0.1.0 is a research release, not a production molecular-simulation
package. The limitations below define the boundary of its scientific claims.

## Adjoint Matching is not uniformly improving

The released experiments do not support a claim that Adjoint Matching always
improves a pretrained Flow Matching model.

- DW1D and DW2D are clear positive toy examples.
- All three formal DW4 AM training seeds improve the released energy metric.
- MB2D is transfer-dependent: the high-beta transfers improve, whereas the
  low-beta transfers worsen in energy W2 despite small geometric improvements.
- LJ13 shows modest and comparatively stable pairwise/radial improvements, but
  its energy W2 is sensitive to evaluation seed.
- Ala2 remains an unsuccessful application of the released route.

Energy and geometry can therefore move in different directions. Results should
be reported per observable and per transfer, rather than collapsed into a
single success label.

## Ala2 is an explicit failed case

The Ala2 checkpoints are preserved for transparency and follow-up research.
They are not evidence of successful temperature transfer.

Across the representative 400 K, 600 K, and 800 K attempts, some energy,
Ramachandran, or torsion metrics improve in isolation. However, the generated
energy distribution retains a substantially heavier high-energy tail than the
MD reference. The 600 K route is approximately null or worse, and the strongest
800 K energy correction does not yield a corresponding Ramachandran
improvement.

Consequently:

- Ala2 is excluded from successful aggregate tables;
- its checkpoints are labeled `experimental_failed_case`;
- no released checkpoint should be described as a validated Ala2 Boltzmann
  generator.

## Seed semantics differ across systems

The three DW4 records are three independently trained AM checkpoints. The
three LJ13 records use the same FM and AM checkpoint with evaluation seeds 101,
202, and 303. LJ13 therefore measures sampling/evaluation variability, not
training-seed variability.

These two forms of repetition must not be pooled or described interchangeably.

## Representative and exploratory checkpoints

DW4 seed 2 is the representative formal checkpoint, but the formal conclusion
uses all three training seeds. An exploratory scan-selected checkpoint may be
retained for provenance; it is not part of the formal three-seed evidence and
should not be substituted into the aggregate result.

## Metric protocol

The number of generated samples and the number used in each metric computation
are separate. Release-facing `energy_w2_2k` and `geometric_w2_2k` values use
fixed-size 2,000-sample subsets. Repeated subset statistics are required where
the run record specifies them.

Changing the subset size, reference archive, integration method, ODE step
count, or seed creates a different evaluation protocol. Such results should not
be compared directly with the curated v0.1.0 numbers without an explicit
protocol-matching analysis.

## Reconstructed legacy profiles

Some early toy runs did not preserve a standalone configuration. Their
reproduction profiles were reconstructed from checkpoint shapes, run history,
and legacy defaults and are marked accordingly. These profiles support
method-level reproduction, not a claim of exact historical provenance or
byte-identical retraining.

Even exact profiles can differ at the bit level because accelerator kernels and
parallel reductions may be nondeterministic.

## Data distribution

Large NumPy reference arrays are intentionally excluded from ordinary Git
history. Non-Ala2 arrays are distributed as a versioned GitHub Release asset
and are described by the manifests in `data/`. Ala2 MD trajectories are
distributed separately.

A run is only comparable with the release when its data match the manifest
hashes, shapes, thermodynamic settings, and units. The repository cannot
guarantee continued availability of third-party or separately hosted Ala2
archives.

## Integrators and physical validation

The release uses finite-step numerical integration. Accuracy depends on the
configured method and step count. The repository does not establish that every
configuration is converged with respect to integration error.

Toy and LJ13 reference data use the documented Langevin procedure; Ala2 uses
separately generated OpenMM MD trajectories. These references are not
interchangeable, and agreement with them does not constitute validation against
experiment.

## Parameterization coverage

Every benchmark result in v0.1.0 uses `target-refinement`. The
`anchor-residual` parameterization is implemented as a research option but has
not been included in the released benchmark study. Its presence in the code is
not an empirical claim.

## Software scope

TAM targets the tested Python/JAX/OpenMM configurations recorded by the
release. It is not certified for long-term production MD, distributed
multi-node execution, arbitrary molecular topologies, or safety-critical use.
