from __future__ import annotations

import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

ALA2_DATA_ROOT = Path("${AMBOLTZ_ROOT}/data/ala2")
DEFAULT_PDB_PATH = ALA2_DATA_ROOT / "structure_vac.pdb"

_ALA2_DATA_TO_OPENMM_ORDER = np.asarray(
    [1, 0, 2, 3, 4, 5, 6, 10, 7, 11, 12, 13, 14, 15, 8, 9, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
_ALA2_OPENMM_TO_DATA_ORDER = np.empty_like(_ALA2_DATA_TO_OPENMM_ORDER)
_ALA2_OPENMM_TO_DATA_ORDER[_ALA2_DATA_TO_OPENMM_ORDER] = np.arange(len(_ALA2_DATA_TO_OPENMM_ORDER))


def _ensure_openmm_app_xtc_compat() -> None:
    # OpenMM 8.2's XTC extension may be compiled against NumPy 1.x.  ADTM only
    # needs PDBFile/ForceField/Simulation, so avoid importing the optional XTC writer.
    if "openmm.app.xtcfile" not in sys.modules:
        module = types.ModuleType("openmm.app.xtcfile")

        class XTCFile:  # pragma: no cover - never used by this project.
            pass

        module.XTCFile = XTCFile
        sys.modules["openmm.app.xtcfile"] = module


def _ensure_pkg_resources_compat() -> None:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import sys
        import types
        from importlib import resources

        module = types.ModuleType("pkg_resources")

        def resource_filename(package: str, relative_path: str) -> str:
            return str(resources.files(package).joinpath(relative_path))

        module.resource_filename = resource_filename
        sys.modules["pkg_resources"] = module


def _project_mean_free_np(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    y = np.asarray(x, dtype=np.float32).reshape((-1, 22, 3))
    y = y - np.mean(y, axis=1, keepdims=True)
    return y.reshape(shape)


def _clip_vector_norm_np(x: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None or float(max_norm) <= 0.0:
        return x
    flat = x.reshape((x.shape[0], -1))
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / (norm + 1.0e-12))
    return (flat * scale).reshape(x.shape)


class AlanineDipeptideOpenMMBridge:
    """OpenMM bridge returning U(x) and grad U(x) for Ala2 coordinates in nm."""

    def __init__(
        self,
        device: str = "cuda",
        precision: str = "mixed",
        system_name: str = "amber99sbildn_obc1_xml",
        forcefield: str = "amber99sbildn.xml",
        forcefield_water: str = "implicit/obc1.xml",
        pdb_path: str | Path = DEFAULT_PDB_PATH,
    ) -> None:
        try:
            _ensure_openmm_app_xtc_compat()
            import openmm
            import openmm.app
            import openmm.unit as unit
            _ensure_pkg_resources_compat()
        except ImportError as exc:
            raise RuntimeError("Ala2 OpenMM energy requires openmm in the ab environment.") from exc

        self.unit = unit
        system_key = str(system_name).lower()
        prmtop_systems = {"alanine_dipeptide_implicit", "ala2_implicit", "implicit_obc1"}
        xml_systems = {"alanine_dipeptide_xml", "ala2_xml", "amber99sbildn_obc1", "amber99sbildn_obc1_xml", "xml_forcefield"}
        if system_key in prmtop_systems:
            try:
                from openmmtools.testsystems import AlanineDipeptideImplicit
            except ImportError as exc:
                raise RuntimeError("The alanine_dipeptide_implicit builder requires openmmtools.") from exc
            test_system = AlanineDipeptideImplicit(constraints=None)
            topology = test_system.topology
            system = test_system.system
            self.data_to_context_order = _ALA2_DATA_TO_OPENMM_ORDER
            self.context_to_data_order = _ALA2_OPENMM_TO_DATA_ORDER
        elif system_key in xml_systems:
            pdb = openmm.app.PDBFile(str(Path(pdb_path)))
            forcefields = [str(forcefield)]
            if forcefield_water:
                forcefields.append(str(forcefield_water))
            ff = openmm.app.ForceField(*forcefields)
            topology = pdb.topology
            system = ff.createSystem(
                topology,
                nonbondedMethod=openmm.app.NoCutoff,
                constraints=None,
                removeCMMotion=False,
            )
            self.data_to_context_order = np.arange(22, dtype=np.int64)
            self.context_to_data_order = np.arange(22, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported OpenMM Ala2 system {system_name!r}.")

        device_key = str(device).lower()
        if device_key == "cuda":
            platform = openmm.Platform.getPlatformByName("CUDA")
            properties = {"DeviceIndex": "0", "Precision": str(precision), "UseCpuPme": "false"}
        elif device_key == "cpu":
            platform = openmm.Platform.getPlatformByName("CPU")
            properties = {}
        else:
            raise ValueError(f"Unsupported OpenMM device {device!r}; use 'cuda' or 'cpu'.")

        integrator = openmm.LangevinIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.femtosecond)
        self.simulation = openmm.app.Simulation(topology, system, integrator, platform, properties)

    def energy_and_grad(self, x_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_nm, dtype=np.float64).reshape((-1, 22, 3))
        energies: list[float] = []
        grads: list[np.ndarray] = []
        for coords in x:
            coords_openmm = coords[self.data_to_context_order]
            self.simulation.context.setPositions(coords_openmm * self.unit.nanometer)
            state = self.simulation.context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(self.unit.kilojoule_per_mole / self.unit.nanometer)
            energies.append(float(energy))
            grad_openmm = -np.asarray(forces, dtype=np.float64)
            grads.append(grad_openmm[self.context_to_data_order])
        return np.asarray(energies, dtype=np.float32), np.asarray(grads, dtype=np.float32).reshape((-1, 66))


@lru_cache(maxsize=8)
def get_ala2_openmm_bridge(
    device: str = "cuda",
    precision: str = "mixed",
    system_name: str = "amber99sbildn_obc1_xml",
    forcefield: str = "amber99sbildn.xml",
    forcefield_water: str = "implicit/obc1.xml",
    pdb_path: str = str(DEFAULT_PDB_PATH),
) -> AlanineDipeptideOpenMMBridge:
    return AlanineDipeptideOpenMMBridge(device, precision, system_name, forcefield, forcefield_water, pdb_path)


def openmm_energy_and_grad_np(
    x_nm: np.ndarray,
    device: str | None = None,
    precision: str = "mixed",
    system_name: str = "amber99sbildn_obc1_xml",
    forcefield: str = "amber99sbildn.xml",
    forcefield_water: str = "implicit/obc1.xml",
    pdb_path: str | Path = DEFAULT_PDB_PATH,
    grad_clip: float | None = None,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = os.environ.get("ADTM_ALA2_OPENMM_DEVICE", "cuda")
    x = np.asarray(x_nm, dtype=np.float32).reshape((-1, 66))
    if bool(center):
        x = _project_mean_free_np(x)
    bridge = get_ala2_openmm_bridge(str(device), str(precision), str(system_name), str(forcefield), str(forcefield_water), str(pdb_path))
    energy, grad = bridge.energy_and_grad(x)
    grad = _project_mean_free_np(grad)
    grad = _clip_vector_norm_np(grad, None if grad_clip is None else float(grad_clip))
    return energy.astype(np.float32), grad.astype(np.float32)
