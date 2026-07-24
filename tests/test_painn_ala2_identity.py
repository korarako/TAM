from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from adj_thermo.cli import build_parser
from adj_thermo.model.painn_ala2_chiro_jax import (
    apply_ala2_chiro_painn_jax_velocity_package,
    init_ala2_chiro_painn_jax_velocity,
    normalized_temperature_from_beta,
)
from adj_thermo.problem import make_problem
from adj_thermo.problem.ala2 import temperature_to_beta
from adj_thermo.sampler import sample_ode


def test_ala2_cli_retains_only_the_active_model_route():
    args = build_parser().parse_args(
        [
            "train-fm",
            "--problem",
            "ala2",
            "--model",
            "ala2_chiro_painn_jax",
            "--run-dir",
            "out",
            "--beta-grid",
            str(temperature_to_beta(300.0)),
            str(temperature_to_beta(1000.0)),
        ]
    )
    assert args.problem == "ala2"
    assert args.model == "ala2_chiro_painn_jax"
    assert args.ala2_openmm_system == "amber99sbildn_obc1_xml"


def test_ala2_problem_shape_and_temperature_coordinate():
    problem = make_problem("ala2")
    assert problem.dim == 66
    assert problem.n_particles == 22
    assert problem.spatial_dim == 3
    values = normalized_temperature_from_beta(
        jnp.asarray([temperature_to_beta(300.0), temperature_to_beta(1000.0)])
    )
    np.testing.assert_allclose(np.asarray(values), np.asarray([0.0, 1.0]), atol=1.0e-6)


def test_ala2_chiral_painn_package_and_sampler():
    problem = make_problem("ala2")
    package, _ = init_ala2_chiro_painn_jax_velocity(
        jax.random.PRNGKey(1),
        problem,
        hidden_dim=8,
        n_layers=1,
        t_embed_dim=4,
        temp_embed_dim=4,
        atom_embed_dim=4,
    )
    assert package["model_type"] == "ala2_chiro_painn_jax"
    x = problem.project_fn(jax.random.normal(jax.random.PRNGKey(2), (2, 66)))
    velocity = apply_ala2_chiro_painn_jax_velocity_package(
        package,
        x,
        jnp.zeros((2, 1)),
        temperature_to_beta(500.0),
        problem,
    )
    assert velocity.shape == (2, 66)
    assert bool(jnp.all(jnp.isfinite(velocity)))
    samples = sample_ode(
        package,
        jax.random.PRNGKey(3),
        problem,
        temperature_to_beta(500.0),
        n_samples=2,
        n_steps=1,
        prior_scale=0.15,
    )
    assert samples.shape == (2, 66)
    assert np.isfinite(samples).all()
