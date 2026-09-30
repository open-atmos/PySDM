# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import pytest
import numpy as np
from PySDM import Formulae, Builder
from PySDM.dynamics import HomogeneousLiquidNucleation
from PySDM.products import SuperDropletCountPerGridbox, ParticleConcentration
from PySDM.environments import Box
from PySDM.physics import si


@pytest.mark.parametrize(
    "nucleation_rate",
    (
        44 / si.s / si.m**3,
        444 / si.s / si.m**3,
    ),
)
@pytest.mark.parametrize("timestep", (0.25 * si.s, 10 * si.s))
@pytest.mark.parametrize("volume", (0.25 * si.m**3, 10 * si.m**3))
@pytest.mark.parametrize("n_steps", (1, 44))
def test_const_j_expectation_match(
    nucleation_rate, backend_class, timestep, volume, n_steps
):
    if backend_class.__name__ == "ThrustRTC":
        pytest.skip("TODO #1492")

    # arrange
    n_sd = n_steps

    formulae = Formulae(
        homogeneous_liquid_nucleation_rate="Constant",
        constants={"J_LIQ_HOMO": nucleation_rate, "R_LIQ_HOMO": 0},
    )
    builder = Builder(
        backend=backend_class(formulae),
        n_sd=n_sd,
        environment=Box(dt=timestep, dv=volume),
    )
    builder.add_dynamic(HomogeneousLiquidNucleation())

    # TODO #1492
    for attr in ("dry volume", "kappa times dry volume"):
        builder.request_attribute(attr)

    particulator = builder.build(
        attributes={
            "multiplicity": np.full(n_sd, fill_value=np.nan),
            "water mass": np.full(n_sd, fill_value=0),
            "dry volume": np.full(n_sd, fill_value=0),
            "kappa times dry volume": np.full(n_sd, fill_value=0),
        },
        products=(
            SuperDropletCountPerGridbox(name="n_super_particles_per_cell"),
            ParticleConcentration(name="n_particles_per_dv"),
        ),
    )

    for var in ("RH", "T"):
        particulator.environment[var] = np.nan

    # act
    particulator.run(steps=n_steps)

    # assert
    assert particulator.products["n_super_particles_per_cell"].get() == n_steps
    np.testing.assert_allclose(
        actual=particulator.products["n_particles_per_dv"].get(),
        desired=nucleation_rate * timestep * n_steps,
        rtol=1e-1,
    )
