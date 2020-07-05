"""
Created at 2019
"""

import numpy as np

from PySDM.physics.constants import si
from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation


def test_environment():
    # Arrange
    setup = Setup()
    setup.n_steps = -1
    simulation = Simulation(setup, None)
    simulation.reinit()

    # Act
    simulation.run()
    rhod = simulation.particles.environment["rhod"].to_ndarray().reshape(setup.grid)

    # Assert - same in all columns
    for column in range(setup.grid[0]):
        np.testing.assert_array_equal(
            rhod[column, :], rhod[0, :]
        )

    # Assert - decreasing with altitude
    rhod_below = 2 * si.kilograms / si.metre**3
    for level in range(setup.grid[1]):
        rhod_above = rhod[0, level]
        assert rhod_above < rhod_below
        rhod_below = rhod_above
