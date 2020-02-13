import numpy as np
from matplotlib import pyplot
from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1 import plotter
from PySDM.simulation.physics.constants import si


def test_environment(plot=False):
    # Arrange
    setup = Setup()
    setup.n_steps = -1
    simulation = Simulation(setup, None)
    simulation.reinit()

    # Act
    simulation.run()
    rhod = setup.backend.to_ndarray(simulation.particles.environment["rhod"]).reshape(setup.grid)

    # Plot
    if plot:
        fig, ax = pyplot.subplots(1, 1)
        plotter.image(ax, rhod, setup.size, label='rho_d [kg/m^3]')
        pyplot.show()

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