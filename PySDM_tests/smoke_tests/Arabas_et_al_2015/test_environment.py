import numpy as np

from PySDM.physics.constants import si
from PySDM_examples.Arabas_et_al_2015.settings import Settings
from PySDM_examples.Arabas_et_al_2015.simulation import Simulation


def test_environment():
    # Arrange
    settings = Settings()
    settings.simulation_time = -1 * settings.dt
    simulation = Simulation(settings, None)
    simulation.reinit()

    # Act
    simulation.run()
    rhod = simulation.core.environment["rhod"].to_ndarray().reshape(settings.grid)

    # Assert - same in all columns
    for column in range(settings.grid[0]):
        np.testing.assert_array_equal(
            rhod[column, :], rhod[0, :]
        )

    # Assert - decreasing with altitude
    rhod_below = 2 * si.kilograms / si.metre**3
    for level in range(settings.grid[1]):
        rhod_above = rhod[0, level]
        assert rhod_above < rhod_below
        rhod_below = rhod_above
