# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998 import Simulation

from PySDM import Formulae
from PySDM.backends import CPU
from PySDM.physics import si

from .dummy_storage import DummyStorage


@pytest.mark.parametrize(
    "singular",
    (
        pytest.param(False, id="singular: False"),
        pytest.param(True, id="singular: True"),
    ),
)
def test_freezing(singular):
    # Arrange
    settings = Settings(
        Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            seed=44,
            condensation_coordinate="VolumeLogarithm",
            fastmath=True,
            freezing_temperature_spectrum="Niemand_et_al_2012",
            heterogeneous_ice_nucleation_rate="ABIFM",
            constants={
                "NIEMAND_A": -0.517,
                "NIEMAND_B": 8.934,
                "ABIFM_M": 28.13797,
                "ABIFM_C": -2.92414,
            },
        )
    )
    settings.dt = 0.5 * si.second
    settings.grid = (5, 15)
    settings.n_sd_per_gridbox = 64

    settings.simulation_time = 100 * settings.dt
    settings.spin_up_time = 10 * settings.dt

    settings.output_interval = settings.dt  # settings.simulation_time

    settings.processes["freezing"] = True
    settings.processes["coalescence"] = False

    settings.freezing_singular = singular
    settings.th_std0 -= 35 * si.K
    settings.initial_water_vapour_mixing_ratio -= 7.15 * si.g / si.kg

    storage = DummyStorage()
    simulation = Simulation(settings, storage, SpinUp=SpinUp, backend_class=CPU)
    simulation.reinit()

    # Act
    simulation.run()

    # Assert
    assert (simulation.products["ice water content"].get() > 0).any()
