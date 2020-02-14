from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM_tests.smoke_tests.utils import bdf
import pytest

# TODO: run for different atol, rtol, dt_max
@pytest.mark.skip
@pytest.mark.parametrize("scheme", ['default', 'BDF'])
def test_just_do_it(scheme):
    # Arrange
    setup = Setup()
    if scheme == 'BDF':
        setup.dt_max = 10  #setup.total_time

    simulation = Simulation(setup)
    if scheme == 'BDF':
        bdf.patch_particles(simulation.particles)

    # Act
    output = simulation.run()

    # Assert
    # TODO!
