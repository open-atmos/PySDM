from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM.simulation.physics.constants import si
from PySDM_tests.smoke_tests.utils import bdf
import pytest
import numpy as np


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
    r = np.array(output['r']).T * si.metres
    n = setup.n / (setup.mass_of_dry_air  * si.kilogram)

    # Assert
    condition = (r > 1 * si.micrometre)
    NTOT = n_tot(n, condition)
    N1 = NTOT[: int(1/3 * len(NTOT))]
    N2 = NTOT[int(1/3 * len(NTOT)): int(2/3 * len(NTOT))]
    N3 = NTOT[int(2/3 * len(NTOT)):]
    print()
    print('N1minmax:', min(N1), max(N1))
    print('N2minmax:', min(N2), max(N2))
    print('N3minmax:', min(N3), max(N3))

    n_unit = 1/si.microgram
    assert min(N1) == 0.0 * n_unit
    assert .63 * n_unit < max(N1) < .68 * n_unit
    assert .14 * n_unit < min(N2) < .15 * n_unit
    assert .3 * n_unit < max(N2) < .37 * n_unit
    assert .08 * n_unit < min(N3) < .083 * n_unit
    assert .27 * n_unit <max(N3) < .4 * n_unit



def n_tot(n, condition):
    return np.dot(n, condition)

