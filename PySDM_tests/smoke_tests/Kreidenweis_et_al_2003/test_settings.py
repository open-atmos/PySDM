from PySDM_examples.Kreidenweis_et_al_2003 import Settings
import numpy as np


def test_settings(plot=False):
    # Arrange
    settings = Settings(n_sd=2, dt=np.nan)

    # Plot
    if plot:
        pass

    # Assert
    for compound in ('SO2', 'O3', 'H2O2', 'CO2', 'HNO3'):
        assert (settings.starting_amounts["moles_" + compound] == 0).all()
    for compound in ('NH3', 'H', 'SO4'):
        assert (settings.starting_amounts["moles_" + compound] != 0).all()

    # TODO: asserts on actual values!