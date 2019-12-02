from examples.Yang_et_al_2018_Fig_2.example import Simulation
from examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM.simulation.physics.constants import si
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def test_displacement(plot=False):
    # Arrange
    setup = Setup(dt=10 * si.second)
    setup.n_sd = 0
    simulation = Simulation(setup)

    # Act
    output = simulation.run()

    # Plot
    if plot:
        plt.plot(output["t"], output["S"])
        plt.grid()
        plt.show()

    # Assert
    assert np.argmin(output["z"]) == 0
    np.testing.assert_approx_equal(output["z"][0], setup.z0, 2) # TODO: save t=0 & ==
    np.testing.assert_approx_equal(output["z"][-1], 1000)
    np.testing.assert_approx_equal(np.amax(output["z"]), 1200)
    assert signal.argrelextrema(np.array(output["z"]), np.greater)[0].shape[0] == 10
    assert signal.argrelextrema(np.array(output["z"]), np.less)[0].shape[0] == 10
