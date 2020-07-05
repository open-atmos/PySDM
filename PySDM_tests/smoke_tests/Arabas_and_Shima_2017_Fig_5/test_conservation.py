"""
Created at 2019
"""

from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import Setup, w_avgs
from PySDM_tests.smoke_tests.utils import bdf
from PySDM.physics import constants as const
import pytest
import numpy as np


def ql(simulation: Simulation):
    backend = simulation.particles.backend

    droplet_volume = simulation.particles.state['volume'].to_ndarray()[0]

    droplet_number = simulation.particles.state['n'].to_ndarray()[0]

    droplet_mass = droplet_number * droplet_volume * const.rho_w

    env = simulation.particles.environment
    return droplet_mass / env.mass_of_dry_air


@pytest.mark.parametrize("setup_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air", [1, 10000]) # [1, 10, 100, 1000, 10000])
@pytest.mark.parametrize("scheme", ['BDF', 'default'])
def test_water_mass_conservation(setup_idx, mass_of_dry_air, scheme):
    # Arrange
    setup = Setup(
        w_avg=setups[setup_idx].w_avg,
        N_STP=setups[setup_idx].N_STP,
        r_dry=setups[setup_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air
    )
    setup.n_output = 50
    setup.scheme = scheme
    simulation = Simulation(setup)
    qt0 = setup.q0 + ql(simulation)
    if scheme == 'BDF':
        bdf.patch_particles(simulation.particles)

    # Act
    simulation.run()

    # Assert
    qt = simulation.particles.environment["qv"].to_ndarray() + ql(simulation)
    np.testing.assert_approx_equal(qt, qt0, 14)  # TODO: was 15 at some point...


@pytest.mark.parametrize("setup_idx", range(len(w_avgs)))
@pytest.mark.parametrize("mass_of_dry_air",  [1, 10000]) # [1, 10, 100, 1000, 10000])
def test_energy_conservation(setup_idx, mass_of_dry_air):
    # Arrange
    setup = Setup(
        w_avg=setups[setup_idx].w_avg,
        N_STP=setups[setup_idx].N_STP,
        r_dry=setups[setup_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air,
    )
    simulation = Simulation(setup)
    env = simulation.particles.environment
    thd0 = env['thd']

    # Act
    simulation.run()

    # Assert
    np.testing.assert_approx_equal(thd0.to_ndarray(), env['thd'].to_ndarray())
