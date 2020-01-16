from PySDM_examples.Arabas_and_Shima_2017_Fig_5.example import Simulation, setups
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import Setup, w_avgs, N_STPs, r_drys
from PySDM.simulation.physics import constants as const
from PySDM.simulation.physics import formulae as phys
import pytest
import numpy as np
import itertools


def ql(simulation: Simulation):
    backend = simulation.particles.backend

    droplet_volume = simulation.particles.state.get_backend_storage('volume')
    droplet_volume = backend.to_ndarray(droplet_volume)[0]

    droplet_number = simulation.particles.state.n
    droplet_number = backend.to_ndarray(droplet_number)[0]

    droplet_mass = droplet_number * droplet_volume * const.rho_w

    env = simulation.particles.environment
    return droplet_mass / env.mass_of_dry_air


def mse(simulation: Simulation):
    env = simulation.particles.environment
    return phys.mse(T=env['T'][0], qv=env['qv'][0], ql=ql(simulation), z=env['z'][0])


@pytest.mark.parametrize(
    "setup_idx, mass_of_dry_air", itertools.product(range(len(w_avgs)), [1, 10, 100, 1000, 10000])
)
def test_water_mass_conservation(setup_idx, mass_of_dry_air):
    # Arrange
    setup = Setup(
        w_avg=setups[setup_idx].w_avg,
        N_STP=setups[setup_idx].N_STP,
        r_dry=setups[setup_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air
    )
    setup.n_steps = 50
    simulation = Simulation(setup)
    qt0 = setup.q0 + ql(simulation)

    # Act
    simulation.run()

    # Assert
    qt = simulation.particles.environment["qv"] + ql(simulation)
    np.testing.assert_approx_equal(qt, qt0, 15)


@pytest.mark.parametrize(
    "setup_idx, mass_of_dry_air", itertools.product(range(len(w_avgs)), [1, 10, 100, 1000, 10000])
)
def test_energy_conservation(setup_idx, mass_of_dry_air ):
    # Arrange
    setup = Setup(
        w_avg=setups[setup_idx].w_avg,
        N_STP=setups[setup_idx].N_STP,
        r_dry=setups[setup_idx].r_dry,
        mass_of_dry_air=mass_of_dry_air
    )
    simulation = Simulation(setup)
    mse0 = mse(simulation)

    # Act
    simulation.run()

    # Assert
    mse1 = mse(simulation)
    np.testing.assert_approx_equal(mse0, mse1, 4)  # TODO
