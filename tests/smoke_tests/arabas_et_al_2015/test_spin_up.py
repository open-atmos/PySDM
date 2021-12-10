# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot
import pytest
from PySDM_examples.Szumowski_et_al_1998 import Simulation
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM.physics import si

from ...backends_fixture import backend_class
assert hasattr(backend_class, '_pytestfixturefunction')


class DummyStorage:
    def __init__(self):
        self.profiles = []

    def init(*_):  # pylint: disable=no-method-argument
        pass

    def save(self, data: np.ndarray, step: int, name: str):  # pylint: disable=unused-argument
        if name == "qv_env":
            self.profiles.append({"qv_env": np.mean(data, axis=0)})


@pytest.mark.parametrize("fastmath", (
        pytest.param(False, id="fastmath: False"),
        pytest.param(True, id="fastmath: True")
))
# pylint: disable=redefined-outer-name
def test_spin_up(backend_class, fastmath, plot=False):
    # Arrange
    settings = Settings(fastmath=fastmath)
    settings.dt = .5 * si.second
    settings.grid = (3, 25)
    settings.simulation_time = 20 * settings.dt
    settings.output_interval = 1 * settings.dt

    storage = DummyStorage()
    simulation = Simulation(settings, storage, SpinUp=SpinUp, backend_class=backend_class)
    simulation.reinit()

    # Act
    simulation.run()

    # Plot
    if plot:
        levels = np.arange(settings.grid[1])
        for step, datum in enumerate(storage.profiles):
            pyplot.plot(datum["qv_env"], levels, label=str(step))
        pyplot.legend()
        pyplot.show()

    # Assert
    step_num = len(storage.profiles) - 1
    for step in range(step_num):
        next_profile = storage.profiles[step + 1]["qv_env"]
        prev_profile = storage.profiles[step]["qv_env"]
        eps = 1e-3
        assert ((prev_profile + eps) >= next_profile).all()
    assert storage.profiles[step_num]["qv_env"][-1] < 7.1
