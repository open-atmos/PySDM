"""
Created at 2019
"""
import pytest
import numpy as np

from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.settings import Settings, w_avgs
from PySDM.backends.numba.test_helpers import bdf
from PySDM.physics import constants as const

# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


def ql(simulation: Simulation):
    droplet_volume = simulation.core.particles['volume'].to_ndarray()[0]

    droplet_number = simulation.core.particles['n'].to_ndarray()[0]

    droplet_mass = droplet_number * droplet_volume * const.rho_w

    env = simulation.core.environment
    return droplet_mass / env.mass_of_dry_air


@pytest.mark.parametrize("settings_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air", (1, 10000))
@pytest.mark.parametrize("scheme", ('BDF', 'default'))
@pytest.mark.parametrize("coord", ('VolumeLogarithm', 'Volume'))
def test_water_mass_conservation(backend, settings_idx, mass_of_dry_air, scheme, coord):
    # Arrange
    settings = Settings(
        w_avg=setups[settings_idx].w_avg,
        N_STP=setups[settings_idx].N_STP,
        r_dry=setups[settings_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air,
        coord=coord
    )
    settings.n_output = 50
    settings.coord = coord
    simulation = Simulation(settings, backend)
    qt0 = settings.q0 + ql(simulation)

    assert scheme in ('BDF', 'default')
    if scheme == 'BDF':
        bdf.patch_core(simulation.core)

    # Act
    simulation.run()

    # Assert
    qt = simulation.core.environment["qv"].to_ndarray() + ql(simulation)
    np.testing.assert_approx_equal(qt, qt0, 14)


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
    env = simulation.core.environment
    thd0 = env['thd']

    # Act
    simulation.run()

    # Assert
    np.testing.assert_approx_equal(thd0.to_ndarray(), env['thd'].to_ndarray())
