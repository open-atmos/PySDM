"""
Created at 2020
"""

from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.settings import Settings
from PySDM.physics.constants import si
from PySDM_tests.smoke_tests.utils import bdf
import pytest
import numpy as np
import os


# TODO: run for different atol, rtol, dt_max
if os.environ.get('TRAVIS') == 'true' and not os.environ.get('FAST_TESTS') == 'true':
    scheme = ('default',  'BDF')
    coord = ('volume logarithm', 'volume')
    adaptive = (True, False)
    enable_particle_temperatures = (False, True)
else:
    scheme = ('default',)
    coord = ('volume logarithm',)
    adaptive = (True,)
    enable_particle_temperatures = (False,)


@pytest.mark.parametrize("scheme", scheme)
@pytest.mark.parametrize("coord", coord)
@pytest.mark.parametrize("adaptive", adaptive)
@pytest.mark.parametrize("enable_particle_temperatures", enable_particle_temperatures)
def test_just_do_it(scheme, coord, adaptive, enable_particle_temperatures):    # Arrange
    if scheme == 'BDF' and not adaptive:
        return
    if scheme == 'BDF' and coord == 'volume':
        return

    settings = Settings(dt_output=10 * si.second)
    settings.coord = coord
    settings.adaptive = adaptive
    settings.enable_particle_temperatures = enable_particle_temperatures
    if scheme == 'BDF':
        settings.dt_max = settings.dt_output
    elif not adaptive:
        settings.dt_max = 1 * si.second

    simulation = Simulation(settings)
    if scheme == 'BDF':
        bdf.patch_core(simulation.core, settings.coord)

    # Act
    output = simulation.run()
    r = np.array(output['r']).T * si.metres
    n = settings.n / (settings.mass_of_dry_air * si.kilogram)

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
    assert .27 * n_unit < max(N3) < .4 * n_unit


def n_tot(n, condition):
    return np.dot(n, condition)

