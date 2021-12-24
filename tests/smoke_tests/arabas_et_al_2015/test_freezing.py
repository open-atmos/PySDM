# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from PySDM_examples.Szumowski_et_al_1998 import Simulation
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM import Formulae
from PySDM.physics import si
from PySDM.backends import CPU
from PySDM.physics.freezing_temperature_spectrum import niemand_et_al_2012
from PySDM.physics.heterogeneous_ice_nucleation_rate import abifm
from .dummy_storage import DummyStorage

# TODO #599
niemand_et_al_2012.a = -0.517
niemand_et_al_2012.b = 8.934
abifm.m = 28.13797
abifm.c = -2.92414


@pytest.mark.parametrize("singular", (
        pytest.param(False, id="singular: False"),
        pytest.param(True, id="singular: True")
))
# pylint: disable=redefined-outer-name
def test_freezing(singular):
    # Arrange
    settings = Settings(Formulae(
        condensation_coordinate='VolumeLogarithm',
        fastmath=True,
        freezing_temperature_spectrum='Niemand_et_al_2012',
        heterogeneous_ice_nucleation_rate='ABIFM'
    ))
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
