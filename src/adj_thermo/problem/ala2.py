from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.openmm_ala2 import ALA2_DATA_ROOT, DEFAULT_PDB_PATH, openmm_energy_and_grad_np
from adj_thermo.problem.base import ProblemSpec, ensure_batch_dim
from adj_thermo.utils import project_mean_free

KB_KJ_PER_MOL_K = 0.00831446261815324
TEMP_TO_FILE = {
    300: "implicit_obc1_openmm.npz",
    320: "ala2_obc1_actual320K_sys0_dataset.npz",
    350: "ala2_obc1_actual350K_sys0_dataset.npz",
    380: "ala2_obc1_actual380K_sys0_dataset.npz",
    400: "ala2_obc1_actual400K_sys0_dataset.npz",
    500: "ala2_obc1_actual500K_sys0_dataset.npz",
    600: "ala2_obc1_actual600K_sys0_dataset.npz",
    700: "ala2_obc1_actual700K_sys0_dataset.npz",
    800: "ala2_obc1_actual800K_sys0_dataset.npz",
    900: "ala2_obc1_actual900K_sys0_dataset.npz",
    1000: "ala2_obc1_actual1000K_sys0_dataset.npz",
}
TEMP_TO_BETA = {temp: 1.0 / (KB_KJ_PER_MOL_K * float(temp)) for temp in TEMP_TO_FILE}


def temperature_to_beta(temp_k: float) -> float:
    return 1.0 / (KB_KJ_PER_MOL_K * float(temp_k))


def beta_to_temperature(beta: float) -> float:
    return 1.0 / (KB_KJ_PER_MOL_K * float(beta))


def _normalize_beta_arg(beta: float) -> float:
    b = float(beta)
    if b > 10.0:
        return temperature_to_beta(b)
    return b


def nearest_available_temperature(beta: float) -> int:
    b = _normalize_beta_arg(beta)
    return min(TEMP_TO_BETA, key=lambda temp: abs(TEMP_TO_BETA[temp] - b))


def data_file_for_beta(beta: float, data_root: str | Path = ALA2_DATA_ROOT) -> Path:
    temp = nearest_available_temperature(beta)
    return Path(data_root) / TEMP_TO_FILE[temp]


def _project_np(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    y = np.asarray(x, dtype=np.float32).reshape((-1, 22, 3))
    y = y - np.mean(y, axis=1, keepdims=True)
    return y.reshape(shape).astype(np.float32, copy=False)


def load_ala2_positions_for_beta(
    beta: float,
    n: int | None = None,
    seed: int = 0,
    data_root: str | Path = ALA2_DATA_ROOT,
    center: bool = True,
) -> np.ndarray:
    path = data_file_for_beta(beta, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing Ala2 dataset for beta={float(beta):.8f}: {path}")
    with np.load(path, allow_pickle=False) as data:
        arr = np.asarray(data["R"], dtype=np.float32).reshape((data["R"].shape[0], -1))
    flat = arr.reshape((arr.shape[0], -1))
    arr = arr[np.isfinite(flat).all(axis=1)]
    if center:
        arr = _project_np(arr)
    if n is not None and int(n) > 0 and arr.shape[0] > int(n):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(arr.shape[0], size=int(n), replace=False)
        arr = arr[idx]
    return arr.astype(np.float32, copy=False)


def load_ala2_species(data_root: str | Path = ALA2_DATA_ROOT) -> tuple[int, ...]:
    for temp in (400, 300, 350, 1000):
        path = Path(data_root) / TEMP_TO_FILE[temp]
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return tuple(int(v) for v in np.asarray(data["species"])[0].tolist())
    return (6, 1, 1, 1, 6, 8, 7, 6, 6, 8, 1, 1, 6, 1, 1, 1, 7, 1, 6, 1, 1, 1)


def make_problem(
    openmm_device: str | None = None,
    openmm_grad_clip: float | None = None,
    openmm_system: str | None = None,
    data_root: str | Path = ALA2_DATA_ROOT,
) -> ProblemSpec:
    device = openmm_device or os.environ.get("TAM_ALA2_OPENMM_DEVICE", os.environ.get("ADTM_ALA2_OPENMM_DEVICE", "cpu"))
    system_name = openmm_system or os.environ.get(
        "TAM_ALA2_OPENMM_SYSTEM",
        os.environ.get("ADTM_ALA2_OPENMM_SYSTEM", "amber99sbildn_obc1_xml"),
    )
    if openmm_grad_clip is None:
        env_clip = os.environ.get("TAM_ALA2_OPENMM_GRAD_CLIP", os.environ.get("ADTM_ALA2_OPENMM_GRAD_CLIP"))
        openmm_grad_clip = None if env_clip in {None, ""} else float(env_clip)
    pdb_path = str(Path(data_root) / "structure_vac.pdb") if data_root != ALA2_DATA_ROOT else str(DEFAULT_PDB_PATH)
    species = load_ala2_species(data_root)

    def project_fn(x: jax.Array) -> jax.Array:
        return project_mean_free(x, n_particles=22, spatial_dim=3)

    def energy_fn(x: jax.Array) -> jax.Array:
        xb = ensure_batch_dim(x, 66)
        out_shape = jax.ShapeDtypeStruct((xb.shape[0],), jnp.float32)

        def callback(y: np.ndarray) -> np.ndarray:
            e, _ = openmm_energy_and_grad_np(
                y,
                device=device,
                system_name=system_name,
                pdb_path=pdb_path,
                grad_clip=openmm_grad_clip,
                center=True,
            )
            return e

        return jax.pure_callback(callback, out_shape, xb)

    def grad_energy_fn(x: jax.Array) -> jax.Array:
        xb = ensure_batch_dim(x, 66)
        out_shape = jax.ShapeDtypeStruct(xb.shape, jnp.float32)

        def callback(y: np.ndarray) -> np.ndarray:
            _, g = openmm_energy_and_grad_np(
                y,
                device=device,
                system_name=system_name,
                pdb_path=pdb_path,
                grad_clip=openmm_grad_clip,
                center=True,
            )
            return g.reshape(y.shape).astype(np.float32)

        return project_fn(jax.pure_callback(callback, out_shape, xb))

    return ProblemSpec(
        name="ala2",
        dim=66,
        energy_fn=energy_fn,
        grad_energy_fn=grad_energy_fn,
        project_fn=project_fn,
        default_clip_range=2.0,
        n_particles=22,
        spatial_dim=3,
        atom_species=species,
    )
