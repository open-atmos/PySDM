import pytest
import numpy as np

from PySDM_examples.Yang_et_al_2018.example import Simulation
from PySDM_examples.Yang_et_al_2018.settings import Settings
from PySDM.physics.constants import si
from PySDM.backends.numba.test_helpers import bdf
from PySDM.backends import GPU

# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend

scheme = ('default', 'BDF')
adaptive = (True, False)


@pytest.mark.parametrize("scheme", scheme)
@pytest.mark.parametrize("adaptive", adaptive)
def test_just_do_it(backend, scheme, adaptive):
    # Arrange
    if scheme == 'BDF' and (not adaptive or backend is GPU):
        return

    settings = Settings(dt_output=10 * si.second)
    settings.adaptive = adaptive
    if scheme == 'BDF':
        settings.dt_max = settings.dt_output  # TODO #334 'BDF')
    elif not adaptive:
        settings.dt_max = 1 * si.second

    simulation = Simulation(settings, backend)
    if scheme == 'BDF':
        bdf.patch_core(simulation.core)

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
    assert .6 * n_unit < max(N1) < .8 * n_unit
    assert .17 * n_unit < min(N2) < .18 * n_unit
    assert .35 * n_unit < max(N2) < .41 * n_unit
    assert .1 * n_unit < min(N3) < .11 * n_unit
    assert .27 * n_unit < max(N3) < .4 * n_unit

    # TODO #527
    if backend is not GPU:
        assert max(output['ripening_rate']) > 0


def n_tot(n, condition):
    return np.dot(n, condition)
