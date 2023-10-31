# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Arabas_and_Shima_2017.settings import Settings, setups, w_avgs
from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation

from PySDM.backends import CPU, GPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver


def liquid_water_mixing_ratio(simulation: Simulation):
    droplet_mass = simulation.particulator.attributes["water mass"].to_ndarray()[0]
    droplet_number = simulation.particulator.attributes["multiplicity"].to_ndarray()[0]
    mass_of_all_droplets = droplet_number * droplet_mass
    env = simulation.particulator.environment
    return mass_of_all_droplets / env.mass_of_dry_air


@pytest.mark.parametrize("settings_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air", (1, 10000))
@pytest.mark.parametrize("scheme", ("SciPy", "CPU", "GPU"))
@pytest.mark.parametrize("coord", ("VolumeLogarithm", "Volume"))
def test_water_mass_conservation(settings_idx, mass_of_dry_air, scheme, coord):
    # Arrange
    assert scheme in ("SciPy", "CPU", "GPU")

    settings = Settings(
        w_avg=setups[settings_idx].w_avg,
        N_STP=setups[settings_idx].N_STP,
        r_dry=setups[settings_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air,
        coord=coord,
    )
    settings.n_output = 50
    settings.coord = coord
    simulation = Simulation(settings, GPU if scheme == "GPU" else CPU)
    initial_total_water_mixing_ratio = (
        settings.initial_water_vapour_mixing_ratio
        + liquid_water_mixing_ratio(simulation)
    )

    if scheme == "SciPy":
        scipy_ode_condensation_solver.patch_particulator(simulation.particulator)

    # Act
    simulation.particulator.products["S_max"].get()
    output = simulation.run()

    # Assert
    total_water_mixing_ratio = simulation.particulator.environment[
        "water_vapour_mixing_ratio"
    ].to_ndarray() + liquid_water_mixing_ratio(simulation)
    np.testing.assert_approx_equal(
        total_water_mixing_ratio, initial_total_water_mixing_ratio, significant=6
    )
    if scheme != "SciPy":
        assert simulation.particulator.products["S_max"].get() >= output["S"][-1]


@pytest.mark.parametrize("settings_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air", [1, 10000])
@pytest.mark.parametrize("coord", ("VolumeLogarithm", "Volume"))
def test_energy_conservation(settings_idx, mass_of_dry_air, coord):
    # Arrange
    settings = Settings(
        w_avg=setups[settings_idx].w_avg,
        N_STP=setups[settings_idx].N_STP,
        r_dry=setups[settings_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air,
        coord=coord,
    )
    simulation = Simulation(settings)
    env = simulation.particulator.environment
    thd0 = env["thd"]

    # Act
    simulation.run()

    # Assert
    np.testing.assert_approx_equal(thd0.to_ndarray(), env["thd"].to_ndarray())
