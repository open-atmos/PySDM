from examples.Arabas_and_Shima_2017_Fig_5.example import Simulation, setups
import pytest
import numpy as np


@pytest.mark.parametrize("setup_idx", range(len(setups)))
def test_displacement(setup_idx):
    # Arrange
    setup = setups[setup_idx]
    setup.n_steps = 100
    simulation = Simulation(setup)

    # Act
    output = simulation.run()

    # Assert
    np.testing.assert_almost_equal(min(output["z"]), 0, decimal=1)
    np.testing.assert_almost_equal(max(output["z"]), setup.z_half, decimal=1)
