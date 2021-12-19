# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from PySDM_examples.Szumowski_et_al_1998 import Simulation
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM.physics import si
from PySDM.backends import CPU
from .dummy_storage import DummyStorage



@pytest.mark.parametrize("singular", (
        pytest.param(False, id="singular: False"),
        pytest.param(True, id="singular: True")
))
# pylint: disable=redefined-outer-name
def test_freezing(singular):
    # Arrange
    settings = Settings()
    settings.dt = .5 * si.second
    settings.grid = (3, 25)

    settings.simulation_time = 100 * settings.dt
    settings.spin_up_time = 10 * settings.dt

    settings.output_interval = settings.dt  # settings.simulation_time

    settings.processes['freezing'] = True
    settings.processes['coalescence'] = False

    settings.freezing_singular = singular
    settings.th_std0 -= 35 * si.K
    settings.qv0 -= 7.15 * si.g/si.kg

    storage = DummyStorage()
    simulation = Simulation(settings, storage, SpinUp=SpinUp, backend_class=CPU)
    simulation.reinit()

    # Act
    simulation.run()

    # Assert
    assert (simulation.products['ice water content'].get() > 0).any()
