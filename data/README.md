# Reference data

All released non-Ala2 reference datasets were generated with Langevin dynamics.
LJ13 uses the same unadjusted Langevin update with batched JAX execution,
center-of-mass projection, and mean-free Gaussian noise. No Metropolis/MALA
accept-reject step is used by the release implementation.

Ala2 uses separately distributed OpenMM molecular-dynamics trajectories rather
than the toy/LJ13 Langevin generator. Set `TAM_ALA2_DATA_ROOT` to the directory
containing the `.npz` files listed in `data/ala2/dataset_manifest.yaml`. The
small topology file is included in `data/ala2`.

The checked-out working copy may contain NumPy data for immediate evaluation.
The `*.npy` files are ignored by Git and should be distributed through Git LFS
or an external archive together with `SHA256SUMS` and the dataset manifests.
