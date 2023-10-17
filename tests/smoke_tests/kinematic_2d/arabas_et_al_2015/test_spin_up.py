# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998 import Simulation

from PySDM.formulae import Formulae
from PySDM.physics import si

from .dummy_storage import DummyStorage


@pytest.mark.parametrize(
    "fastmath",
    (
        pytest.param(False, id="fastmath: False"),
        pytest.param(True, id="fastmath: True"),
    ),
)
def test_spin_up(backend_class, fastmath, plot=False):
    # Arrange
    settings = Settings(Formulae(fastmath=fastmath))
    settings.dt = 0.5 * si.second
    settings.grid = (3, 25)
    settings.simulation_time = 20 * settings.dt
    settings.output_interval = 1 * settings.dt

    storage = DummyStorage()
    simulation = Simulation(
        settings, storage, SpinUp=SpinUp, backend_class=backend_class
    )
    simulation.reinit()

    # Act
    simulation.run()

    # Plot
    if plot:
        levels = np.arange(settings.grid[1])
        for step, datum in enumerate(storage.profiles):
            pyplot.plot(datum["water_vapour_mixing_ratio_env"], levels, label=str(step))
        pyplot.legend()
        pyplot.show()

    # Assert
    step_num = len(storage.profiles) - 1
    for step in range(step_num):
        next_profile = storage.profiles[step + 1]["water_vapour_mixing_ratio_env"]
        prev_profile = storage.profiles[step]["water_vapour_mixing_ratio_env"]
        eps = 1e-3
        assert ((prev_profile + eps) >= next_profile).all()
    assert storage.profiles[step_num]["water_vapour_mixing_ratio_env"][-1] < 7.1
