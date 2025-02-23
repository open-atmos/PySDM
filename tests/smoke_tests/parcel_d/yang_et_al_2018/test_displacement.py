# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
from PySDM_examples.Yang_et_al_2018 import Settings, Simulation
from scipy import signal


def test_displacement(plot=False):
    # Arrange
    settings = Settings(n_sd=1)
    simulation = Simulation(settings)

    # Act
    output = simulation.run()

    # Plot
    if plot:
        plt.plot(output["t"], output["S"])
        plt.grid()
        plt.show()

    # Assert
    assert np.argmin(output["z"]) == 0
    assert output["z"][0] == settings.z0
    np.testing.assert_approx_equal(output["z"][-1], 1000)
    np.testing.assert_approx_equal(np.amax(output["z"]), 1200)
    assert signal.argrelextrema(np.array(output["z"]), np.greater)[0].shape[0] == 10
    assert signal.argrelextrema(np.array(output["z"]), np.less)[0].shape[0] == 10
