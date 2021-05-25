from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017.settings import Settings, w_avgs
import pytest
import numpy as np


@pytest.mark.parametrize("w_idx", range(len(w_avgs)))
def test_displacement(w_idx):
    # Arrange
    settings = Settings(
        w_avg=w_avgs[w_idx],
        N_STP=44,
        r_dry=1,
        mass_of_dry_air=1
    )
    settings.n_output = 50
    simulation = Simulation(settings)

    # Act
    output = simulation.run()

    # Assert
    np.testing.assert_almost_equal(min(output["z"]), 0, decimal=1)
    np.testing.assert_almost_equal(max(output["z"]), settings.z_half, decimal=1)
