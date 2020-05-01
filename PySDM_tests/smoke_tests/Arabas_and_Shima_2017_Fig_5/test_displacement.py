from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import Setup, w_avgs
import pytest
import numpy as np


@pytest.mark.parametrize("w_idx", range(len(w_avgs)))
def test_displacement(w_idx):
    # Arrange
    setup = Setup(
        w_avg=w_avgs[w_idx],
        N_STP=0,
        r_dry=1,
        mass_of_dry_air=1
    )
    setup.n_output = 50
    simulation = Simulation(setup)

    # Act
    output = simulation.run()

    # Assert
    np.testing.assert_almost_equal(min(output["z"]), 0, decimal=1)
    np.testing.assert_almost_equal(max(output["z"]), setup.z_half, decimal=1)
