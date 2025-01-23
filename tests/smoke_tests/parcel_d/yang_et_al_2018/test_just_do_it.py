# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Yang_et_al_2018 import Settings, Simulation

from PySDM.backends import CPU, GPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.physics import si

#  TODO #527


@pytest.mark.parametrize("scheme", ("default", "SciPy"))
@pytest.mark.parametrize("adaptive", (True, False))
def test_just_do_it(scheme, adaptive, backend_class=CPU):
    # Arrange
    if scheme == "SciPy" and (not adaptive or backend_class is GPU):
        pytest.skip()

    settings = Settings(dt_output=10 * si.second)
    settings.adaptive = adaptive
    if scheme == "SciPy":
        settings.dt_max = settings.dt_output  # TODO #334
    elif not adaptive:
        settings.dt_max = 1 * si.second

    simulation = Simulation(settings, backend_class)
    if scheme == "SciPy":
        scipy_ode_condensation_solver.patch_particulator(simulation.particulator)

    # Act
    output = simulation.run()
    r = np.array(output["r"]).T * si.metres
    n = settings.n / (settings.mass_of_dry_air * si.kilogram)

    # Assert
    condition = r > 1 * si.micrometre
    NTOT = n_tot(n, condition)
    N1 = NTOT[: int(1 / 3 * len(NTOT))]
    N2 = NTOT[int(1 / 3 * len(NTOT)) : int(2 / 3 * len(NTOT))]
    N3 = NTOT[int(2 / 3 * len(NTOT)) :]

    n_unit = 1 / si.microgram
    assert min(N1) == 0.0 * n_unit
    assert 0.6 * n_unit < max(N1) < 0.8 * n_unit
    assert 0.17 * n_unit < min(N2) < 0.18 * n_unit
    assert 0.35 * n_unit < max(N2) < 0.41 * n_unit
    assert 0.1 * n_unit < min(N3) < 0.11 * n_unit
    assert 0.27 * n_unit < max(N3) < 0.4 * n_unit

    # TODO #527
    if backend_class is not GPU:
        assert max(output["ripening rate"]) > 0


def n_tot(n, condition):
    return np.dot(n, condition)
