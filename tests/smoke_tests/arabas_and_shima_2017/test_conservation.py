# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.settings import Settings, w_avgs
from PySDM.backends.impl_numba.test_helpers import bdf
from PySDM.physics.constants_defaults import rho_w
from PySDM.backends import CPU, GPU


def liquid_water_mixing_ratio(simulation: Simulation):
    droplet_volume = simulation.particulator.attributes['volume'].to_ndarray()[0]
    droplet_number = simulation.particulator.attributes['n'].to_ndarray()[0]
    droplet_mass = droplet_number * droplet_volume * rho_w
    env = simulation.particulator.environment
    return droplet_mass / env.mass_of_dry_air


@pytest.mark.parametrize("settings_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air", (1, 10000))
@pytest.mark.parametrize("scheme", ('BDF', 'CPU', 'GPU'))
@pytest.mark.parametrize("coord", ('VolumeLogarithm', 'Volume'))
def test_water_mass_conservation(settings_idx, mass_of_dry_air, scheme, coord):
    # Arrange
    assert scheme in ('BDF', 'CPU', 'GPU')

    settings = Settings(
        w_avg=setups[settings_idx].w_avg,
        N_STP=setups[settings_idx].N_STP,
        r_dry=setups[settings_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air,
        coord=coord
    )
    settings.n_output = 50
    settings.coord = coord
    simulation = Simulation(settings, GPU if scheme == 'GPU' else CPU)
    qt0 = settings.q0 + liquid_water_mixing_ratio(simulation)

    if scheme == 'BDF':
        bdf.patch_particulator(simulation.particulator)

    # Act
    simulation.particulator.products['S_max'].get()
    output = simulation.run()

    # Assert
    ql = liquid_water_mixing_ratio(simulation)
    qt = simulation.particulator.environment["qv"].to_ndarray() + ql
    significant = 6 if scheme == 'GPU' else 14  # TODO #540
    np.testing.assert_approx_equal(qt, qt0, significant)
    if scheme != 'BDF':
        assert simulation.particulator.products['S_max'].get() >= output['S'][-1]


@pytest.mark.parametrize("settings_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air",  [1, 10000])
@pytest.mark.parametrize("coord", ('VolumeLogarithm', 'Volume'))
def test_energy_conservation(settings_idx, mass_of_dry_air, coord):
    # Arrange
    settings = Settings(
        w_avg=setups[settings_idx].w_avg,
        N_STP=setups[settings_idx].N_STP,
        r_dry=setups[settings_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air,
        coord=coord
    )
    simulation = Simulation(settings)
    env = simulation.particulator.environment
    thd0 = env['thd']

    # Act
    simulation.run()

    # Assert
    np.testing.assert_approx_equal(thd0.to_ndarray(), env['thd'].to_ndarray())
