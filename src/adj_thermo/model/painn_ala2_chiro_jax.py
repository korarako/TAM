from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import numpy as np
from painn_jax import NodeFeatures, cosine_cutoff, gaussian_rbf
from painn_jax.blocks import GatedEquivariantBlock, LinearXav, pooling
from painn_jax.painn import PaiNNReadout

from adj_thermo.model.mlp import sinusoidal_embed_t
from adj_thermo.problem.base import ProblemSpec

Params = Any
KB_KJ_PER_MOL_K = 0.00831446261815324


def _complete_directed_edges(n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    senders = []
    receivers = []
    for i in range(int(n_particles)):
        for j in range(int(n_particles)):
            if i != j:
                senders.append(i)
                receivers.append(j)
    return np.asarray(senders, dtype=np.int32), np.asarray(receivers, dtype=np.int32)


def _as_condition_vector(value: float | jax.Array, batch_size: int, dtype: jnp.dtype) -> jax.Array:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((batch_size,), arr, dtype=dtype)
    return jnp.reshape(arr, (batch_size, -1))[:, 0].astype(dtype)


def temperature_from_beta_or_kelvin(value: float | jax.Array, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    arr = jnp.asarray(value, dtype=dtype)
    return jnp.where(arr > 10.0, arr, 1.0 / (jnp.asarray(KB_KJ_PER_MOL_K, dtype=dtype) * arr))


def normalized_temperature_from_beta(
    beta: float | jax.Array,
    temp_min: float = 300.0,
    temp_max: float = 1000.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    temp = temperature_from_beta_or_kelvin(beta, dtype=dtype)
    return (temp - float(temp_min)) / (float(temp_max) - float(temp_min))


def temperature_positional_embedding(temp_norm: jax.Array, embed_dim: int = 8, l0: float = 75.0) -> jax.Array:
    temp_arr = jnp.asarray(temp_norm, dtype=jnp.float32)
    if temp_arr.ndim == 0:
        temp_arr = temp_arr.reshape((1, 1))
    elif temp_arr.ndim == 1:
        temp_arr = temp_arr[:, None]
    half = max(1, int(np.ceil(int(embed_dim) / 2)))
    freqs = (1.0 + jnp.arange(half, dtype=jnp.float32)) / float(l0)
    phase = temp_arr * freqs[None, :]
    emb = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1).reshape((temp_arr.shape[0], -1))
    return emb[:, : int(embed_dim)]


def _ala2_model_config(
    problem: ProblemSpec,
    hidden_dim: int,
    n_layers: int,
    t_embed_dim: int,
    temp_embed_dim: int,
    atom_embed_dim: int,
    temp_min: float,
    temp_max: float,
    temp_embed_l0: float,
) -> dict[str, Any]:
    if problem.name != "ala2" or problem.n_particles != 22 or problem.spatial_dim != 3 or problem.dim != 66:
        raise ValueError("The Ala2 chiral PaiNN expects 22 atoms in three dimensions.")
    atom_species = tuple(int(v) for v in (problem.atom_species or (1,) * 22))
    return {
        "model_type": "ala2_chiro_painn_jax",
        "problem_name": "ala2",
        "n_particles": 22,
        "spatial_dim": 3,
        "dim": 66,
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "t_embed_dim": int(t_embed_dim),
        "temp_embed_dim": int(temp_embed_dim),
        "beta_embed_dim": int(temp_embed_dim),
        "temp_min": float(temp_min),
        "temp_max": float(temp_max),
        "temp_embed_l0": float(temp_embed_l0),
        "atom_species": atom_species,
        "atom_identity": "learned_atom_index",
        "ala2_atom_identity": "learned_atom_index",
        "atom_embed_dim": int(atom_embed_dim),
        "node_feature_dim": 2 + int(atom_embed_dim) + int(t_embed_dim) + int(temp_embed_dim),
        "radius": 0.8,
        "n_rbf": 64,
        "radial_basis": "painn_jax.gaussian_rbf",
        "cutoff": "painn_jax.cosine_cutoff",
    }


def _ala2_graph(
    x: jax.Array,
    t: float | jax.Array,
    beta: float | jax.Array,
    config: dict[str, Any],
    atom_embedding: jax.Array,
) -> jraph.GraphsTuple:
    n_particles = int(config["n_particles"])
    spatial_dim = int(config["spatial_dim"])
    batch_size = int(x.shape[0])
    dtype = x.dtype
    positions = jnp.reshape(x, (batch_size * n_particles, spatial_dim))
    t_vec = _as_condition_vector(t, batch_size, dtype)
    beta_vec = _as_condition_vector(beta, batch_size, dtype)
    t_emb = sinusoidal_embed_t(t_vec[:, None], embed_dim=int(config["t_embed_dim"])).astype(dtype)
    temp_norm = normalized_temperature_from_beta(
        beta_vec,
        temp_min=float(config["temp_min"]),
        temp_max=float(config["temp_max"]),
        dtype=dtype,
    )
    temp_emb = temperature_positional_embedding(
        temp_norm[:, None],
        embed_dim=int(config["temp_embed_dim"]),
        l0=float(config["temp_embed_l0"]),
    ).astype(dtype)
    repeated_global = jnp.repeat(jnp.concatenate([t_emb, temp_emb], axis=-1), n_particles, axis=0)
    species = jnp.asarray(config["atom_species"], dtype=dtype)
    species = jnp.tile(species[None, :], (batch_size, 1)).reshape((-1, 1)) / 10.0
    ones = jnp.ones((batch_size * n_particles, 1), dtype=dtype)
    atom_emb = jnp.tile(jnp.asarray(atom_embedding, dtype=dtype)[None, :, :], (batch_size, 1, 1))
    atom_emb = atom_emb.reshape((batch_size * n_particles, -1))
    node_s = jnp.concatenate([ones, species, atom_emb, repeated_global], axis=-1)
    base_senders_np, base_receivers_np = _complete_directed_edges(n_particles)
    base_senders = jnp.asarray(base_senders_np, dtype=jnp.int32)
    base_receivers = jnp.asarray(base_receivers_np, dtype=jnp.int32)
    n_edges = int(base_senders_np.shape[0])
    offsets = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32) * n_particles, n_edges)
    senders = jnp.tile(base_senders, batch_size) + offsets
    receivers = jnp.tile(base_receivers, batch_size) + offsets
    return jraph.GraphsTuple(
        nodes=NodeFeatures(s=node_s, v=None),
        edges=positions[receivers] - positions[senders],
        receivers=receivers,
        senders=senders,
        globals=None,
        n_node=jnp.full((batch_size,), n_particles, dtype=jnp.int32),
        n_edge=jnp.full((batch_size,), n_edges, dtype=jnp.int32),
    )


class ChiroPaiNNLayer(hk.Module):
    """Local fork of gerkone/painn-jax PaiNNLayer with a gated cross-product term."""

    def __init__(
        self,
        hidden_size: int,
        layer_num: int,
        activation: Callable = jax.nn.silu,
        blocks: int = 2,
        aggregate_fn: Callable = jraph.segment_sum,
        eps: float = 1.0e-8,
    ):
        super().__init__(f"layer_{layer_num}")
        self._hidden_size = int(hidden_size)
        self._eps = float(eps)
        self._aggregate_fn = aggregate_fn

        self.interaction_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (int(blocks) - 1)
            + [LinearXav(4 * hidden_size)],
            name="interaction_block",
        )
        self.mixing_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (int(blocks) - 1)
            + [LinearXav(3 * hidden_size)],
            name="mixing_block",
        )
        self.vector_mixing_block = LinearXav(
            2 * hidden_size,
            with_bias=False,
            name="vector_mixing_block",
        )

    def _message(
        self,
        s: jax.Array,
        v: jax.Array,
        dir_ij: jax.Array,
        Wij: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        x = self.interaction_block(s)
        xj = x[receivers]
        vj = v[receivers]

        ds, dv1, dv2, cross_gate = jnp.split(Wij * xj, 4, axis=-1)
        edge_vector = dir_ij[..., jnp.newaxis]
        cross = jnp.cross(edge_vector, vj, axis=1)
        dv = dv1 * edge_vector + dv2 * vj + cross_gate * cross

        n_nodes = tree.tree_leaves(s)[0].shape[0]
        ds = self._aggregate_fn(ds, senders, n_nodes)
        dv = self._aggregate_fn(dv, senders, n_nodes)

        s = s + jnp.clip(ds, -1.0e2, 1.0e2)
        v = v + jnp.clip(dv, -1.0e2, 1.0e2)
        return s, v

    def _update(self, s: jax.Array, v: jax.Array) -> Tuple[jax.Array, jax.Array]:
        v_l, v_r = jnp.split(self.vector_mixing_block(v), 2, axis=-1)
        v_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2, keepdims=True) + self._eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)
        ds, dv, dsv = jnp.split(self.mixing_block(ts), 3, axis=-1)
        dv = v_l * dv
        dsv = dsv * jnp.sum(v_r * v_l, axis=1, keepdims=True)

        s = s + jnp.clip(ds + dsv, -1.0e2, 1.0e2)
        v = v + jnp.clip(dv, -1.0e2, 1.0e2)
        return s, v

    def __call__(self, graph: jraph.GraphsTuple, Wij: jax.Array) -> jraph.GraphsTuple:
        s, v = graph.nodes
        s, v = self._message(s, v, graph.edges, Wij, graph.senders, graph.receivers)
        s, v = self._update(s, v)
        return graph._replace(nodes=NodeFeatures(s=s, v=v))


class ChiroPaiNN(hk.Module):
    """gerkone/painn-jax style PaiNN fork with SE(3) chiral message passing."""

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        radial_basis_fn: Callable,
        *args,
        cutoff_fn: Optional[Callable] = None,
        radius: float = 5.0,
        n_rbf: int = 20,
        activation: Callable = jax.nn.silu,
        node_type: str = "discrete",
        task: str = "node",
        pool: str = "sum",
        out_channels: Optional[int] = None,
        readout_fn: Callable[..., Callable[[jraph.GraphsTuple], Tuple[jax.Array, jax.Array]]] = PaiNNReadout,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        eps: float = 1.0e-8,
        **kwargs,
    ):
        super().__init__("chiro_painn")
        assert node_type in ["discrete", "continuous"], "node_type must be discrete or continuous"
        assert task in ["node", "graph"], "task must be node or graph"
        assert radial_basis_fn is not None, "A radial_basis_fn must be provided"

        self._hidden_size = int(hidden_size)
        self._n_layers = int(n_layers)
        self._eps = float(eps)
        self._node_type = str(node_type)
        self._shared_filters = bool(shared_filters)
        self._shared_interactions = bool(shared_interactions)

        self.cutoff_fn = cutoff_fn(radius) if cutoff_fn else None
        self.radial_basis_fn = radial_basis_fn(n_rbf, radius)

        if node_type == "discrete":
            self.scalar_emb = hk.Embed(
                max_z,
                hidden_size,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name="scalar_embedding",
            )
        else:
            self.scalar_emb = LinearXav(hidden_size, name="scalar_embedding")
        self.vector_emb = LinearXav(hidden_size, with_bias=False, name="vector_embedding")

        filter_dim = 4 * int(hidden_size)
        if shared_filters:
            self.filter_net = LinearXav(filter_dim, name="filter_net")
        else:
            self.filter_net = LinearXav(int(n_layers) * filter_dim, name="filter_net")

        if shared_interactions:
            self.layers = [ChiroPaiNNLayer(hidden_size, 0, activation, eps=eps)] * int(n_layers)
        else:
            self.layers = [ChiroPaiNNLayer(hidden_size, i, activation, eps=eps) for i in range(int(n_layers))]

        self.readout = None
        if out_channels is not None and readout_fn is not None:
            self.readout = readout_fn(
                *args,
                hidden_size,
                task,
                pool,
                out_channels=out_channels,
                activation=activation,
                eps=eps,
                **kwargs,
            )

    def _embed(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        s = graph.nodes.s
        if self._node_type == "continuous":
            s = jnp.asarray(s, dtype=jnp.float32)
            if len(s.shape) == 1:
                s = s[:, jnp.newaxis]
        else:
            s = jnp.asarray(s, dtype=jnp.int32)
        s = self.scalar_emb(s)[:, jnp.newaxis]

        if graph.nodes.v is not None:
            v = self.vector_emb(graph.nodes.v)
        else:
            v = jnp.zeros((s.shape[0], 3, s.shape[-1]))
        return graph._replace(nodes=NodeFeatures(s=s, v=v))

    def _get_filters(self, norm_ij: jax.Array) -> list[jax.Array]:
        phi_ij = self.radial_basis_fn(norm_ij)
        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)
        filters = self.filter_net(phi_ij) * norm_ij[:, jnp.newaxis]
        if self._shared_filters:
            return [filters] * self._n_layers
        return jnp.split(filters, self._n_layers, axis=-1)

    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jax.Array, jax.Array]:
        norm_ij = jnp.sqrt(jnp.sum(graph.edges**2, axis=1, keepdims=True) + self._eps)
        dir_ij = graph.edges / (norm_ij + self._eps)
        graph = graph._replace(edges=dir_ij)
        filter_list = self._get_filters(norm_ij)
        graph = self._embed(graph)

        for n, layer in enumerate(self.layers):
            graph = layer(graph, filter_list[n])

        if self.readout is not None:
            s, v = self.readout(graph)
        else:
            s, v = jnp.squeeze(graph.nodes.s), jnp.squeeze(graph.nodes.v)
        return s, v


def ala2_chiro_painn_jax_config(
    problem: ProblemSpec,
    hidden_dim: int = 128,
    n_layers: int = 5,
    t_embed_dim: int = 16,
    temp_embed_dim: int = 8,
    atom_embed_dim: int = 16,
    temp_min: float = 300.0,
    temp_max: float = 1000.0,
    temp_embed_l0: float = 75.0,
) -> dict[str, Any]:
    config = _ala2_model_config(
        problem,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        t_embed_dim=t_embed_dim,
        temp_embed_dim=temp_embed_dim,
        atom_embed_dim=atom_embed_dim,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_embed_l0=temp_embed_l0,
    )
    config.update(
        {
            "model_type": "ala2_chiro_painn_jax",
            "painn_backend": "painn_jax_chiro_fork",
            "source_reference": "gerkone/painn-jax local PaiNNLayer fork",
            "equivariance_group": "SE(3)",
            "filter_channels": "4 * hidden_dim",
            "chiral_message": "dv1 * dir_ij + dv2 * v_receiver + cross_gate * cross(dir_ij, v_receiver)",
        }
    )
    return config


def build_ala2_chiro_painn_jax_velocity_model(config: dict[str, Any]):
    def forward(graph: jraph.GraphsTuple) -> jax.Array:
        model = ChiroPaiNN(
            hidden_size=int(config["hidden_dim"]),
            n_layers=int(config["n_layers"]),
            radial_basis_fn=gaussian_rbf,
            cutoff_fn=cosine_cutoff,
            radius=float(config["radius"]),
            n_rbf=int(config["n_rbf"]),
            node_type="continuous",
            task="node",
            pool="sum",
            out_channels=1,
        )
        _, v = model(graph)
        return jnp.asarray(v)

    return hk.transform_with_state(forward)


def init_ala2_chiro_painn_jax_velocity(
    key: jax.Array,
    problem: ProblemSpec,
    hidden_dim: int = 128,
    n_layers: int = 5,
    t_embed_dim: int = 16,
    temp_embed_dim: int = 8,
    atom_embed_dim: int = 16,
    temp_min: float = 300.0,
    temp_max: float = 1000.0,
    temp_embed_l0: float = 75.0,
) -> tuple[dict[str, Any], Any]:
    config = ala2_chiro_painn_jax_config(
        problem,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        t_embed_dim=t_embed_dim,
        temp_embed_dim=temp_embed_dim,
        atom_embed_dim=atom_embed_dim,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_embed_l0=temp_embed_l0,
    )
    key, atom_key = jax.random.split(key)
    scale = jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float32))
    extra_params = {
        "atom_embedding": jax.random.uniform(
            atom_key,
            (int(config["n_particles"]), int(config["atom_embed_dim"])),
            minval=-scale,
            maxval=scale,
            dtype=jnp.float32,
        )
    }
    model = build_ala2_chiro_painn_jax_velocity_model(config)
    x = problem.project_fn(jax.random.normal(key, (1, int(problem.dim)), dtype=jnp.float32))
    graph = _ala2_graph(
        x,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.asarray(1.0 / (KB_KJ_PER_MOL_K * 1000.0), dtype=jnp.float32),
        config,
        extra_params["atom_embedding"],
    )
    params, state = model.init(key, graph)
    package = {
        "model_type": "ala2_chiro_painn_jax",
        "config": config,
        "params": params,
        "state": state,
        "extra_params": extra_params,
    }
    return package, model


def is_ala2_chiro_painn_jax_velocity_params(params: Any) -> bool:
    return isinstance(params, dict) and params.get("model_type") == "ala2_chiro_painn_jax" and "params" in params and "state" in params


def ala2_chiro_painn_jax_trainable_params_from_package(package: dict[str, Any]) -> Any:
    if "extra_params" in package:
        return {"model": package["params"], "extra": package["extra_params"]}
    return package["params"]


def split_ala2_chiro_painn_jax_trainable_params(params: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(params, dict) and "model" in params and "extra" in params:
        return params["model"], params["extra"]
    return params, {}


def apply_ala2_chiro_painn_jax_velocity_params(
    model: Any,
    params: Any,
    state: Any,
    x: jax.Array,
    t: float | jax.Array,
    beta: float | jax.Array,
    problem: ProblemSpec,
    config: dict[str, Any],
) -> jax.Array:
    x = problem.project_fn(x)
    model_params, extra_params = split_ala2_chiro_painn_jax_trainable_params(params)
    if "atom_embedding" not in extra_params:
        raise ValueError("ala2_chiro_painn_jax params must include extra atom_embedding parameters.")
    graph = _ala2_graph(x, t, beta, config, extra_params["atom_embedding"])
    v, _ = model.apply(model_params, state, None, graph)
    flat = jnp.reshape(v, (x.shape[0], int(config["dim"])))
    return problem.project_fn(flat)


def apply_ala2_chiro_painn_jax_velocity_package(
    package: dict[str, Any],
    x: jax.Array,
    t: float | jax.Array,
    beta: float | jax.Array,
    problem: ProblemSpec,
    model: Any | None = None,
) -> jax.Array:
    if not is_ala2_chiro_painn_jax_velocity_params(package):
        raise ValueError("Expected an ala2_chiro_painn_jax velocity parameter package.")
    config = package["config"]
    if model is None:
        model = build_ala2_chiro_painn_jax_velocity_model(config)
    return apply_ala2_chiro_painn_jax_velocity_params(
        model,
        ala2_chiro_painn_jax_trainable_params_from_package(package),
        package["state"],
        x,
        t,
        beta,
        problem,
        config,
    )
