"""
Created at 2020
"""

from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


# Warning: setup dependent
def test_displacement(plot=False):
    # Arrange
    setup = Setup(n_sd=1)
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
    assert output["z"][0] == setup.z0
    np.testing.assert_approx_equal(output["z"][-1], 1000)
    np.testing.assert_approx_equal(np.amax(output["z"]), 1200)
    assert signal.argrelextrema(np.array(output["z"]), np.greater)[0].shape[0] == 10
    assert signal.argrelextrema(np.array(output["z"]), np.less)[0].shape[0] == 10
