from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM.simulation.physics.constants import si
from PySDM_tests.smoke_tests.utils import bdf
import pytest


# TODO: run for different atol, rtol, dt_max
@pytest.mark.parametrize("scheme", ['default',  'BDF'])
@pytest.mark.parametrize("coord", ['volume logarithm']) # , 'volume'])
#@pytest.mark.parametrize("enable_particle_temperatures", [False, True])
def test_just_do_it(scheme, coord): #, enable_particle_temperatures):    # Arrange
    # Setup.total_time = 15 * si.minute
    setup = Setup(dt_output = 10 * si.second)
    setup.coord = coord
    #setup.enable_particle_temperatures = enable_particle_temperatures
    if scheme == 'BDF':
        setup.dt_max = setup.dt_output

    simulation = Simulation(setup)
    if scheme == 'BDF':
        bdf.patch_particles(simulation.particles, setup.coord)

    # Act
    output = simulation.run()

    # Assert
    # TODO!

