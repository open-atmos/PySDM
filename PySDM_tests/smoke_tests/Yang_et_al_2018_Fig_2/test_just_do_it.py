from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM.physics.constants import si
from PySDM_tests.smoke_tests.utils import bdf
import pytest
import numpy as np


# TODO: run for different atol, rtol, dt_max
@pytest.mark.parametrize("scheme", ['default',  'BDF'])
@pytest.mark.parametrize("coord", ['volume logarithm', 'volume'])
@pytest.mark.parametrize("adaptive", [True, False])
#@pytest.mark.parametrize("enable_particle_temperatures", [False, True]) # TODO !
def test_just_do_it(scheme, coord, adaptive): #, enable_particle_temperatures):    # Arrange
    if scheme == 'BDF' and not adaptive:
        return
    if scheme == 'BDF' and coord == 'volume':
        return

    # Setup.total_time = 15 * si.minute
    setup = Setup(dt_output = 10 * si.second)
    setup.coord = coord
    setup.adaptive = adaptive
    #setup.enable_particle_temperatures = enable_particle_temperatures
    if scheme == 'BDF':
        setup.dt_max = setup.dt_output
    elif not adaptive:
        setup.dt_max = 1 * si.second

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

    n_unit = 1/si.microgram
    assert min(N1) == 0.0 * n_unit
    assert .63 * n_unit < max(N1) < .68 * n_unit
    assert .14 * n_unit < min(N2) < .15 * n_unit
    assert .3 * n_unit < max(N2) < .37 * n_unit
    assert .08 * n_unit < min(N3) < .083 * n_unit
    assert .27 * n_unit <max(N3) < .4 * n_unit



def n_tot(n, condition):
    return np.dot(n, condition)

