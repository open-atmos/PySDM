# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from scipy.signal import find_peaks

from PySDM.initialisation.sampling.spectral_sampling import (
    ConstantMultiplicity,
    Logarithmic,
    UniformRandom,
)
from PySDM.physics import si


@pytest.mark.parametrize(
    "spectral_sampling",
    [
        pytest.param(ConstantMultiplicity, marks=pytest.mark.xfail(strict=True)),
        Logarithmic,
        pytest.param(UniformRandom, marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_spectrum_at_t_0(spectral_sampling, plot=False):
    # Arrange
    settings = Settings(
        n_sd=64, dt=1 * si.s, n_substep=1, spectral_sampling=spectral_sampling
    )
    settings.t_max = 0
    simulation = Simulation(settings)

    # Act
    output = simulation.run()

    # Plot
    if plot:
        pyplot.step(
            2e6 * settings.dry_radius_bins_edges[:-1],
            output["dm_S_VI/dlog_10(dry diameter)"][-1],
        )
        pyplot.ylabel("dS(VI)/dlog_10(D)")
        pyplot.xlabel("dry diameter [µm]")
        pyplot.xscale("log")
        pyplot.yscale("log")
        pyplot.ylim([0.01, 12])
        pyplot.show()

    # Assert
    key = "S_VI"
    spectrum = output[f"dm_{key}/dlog_10(dry diameter)"][0]
    peaks, _ = find_peaks(spectrum)
    assert len(peaks) == 1
    assert 3 < np.amax(spectrum) < 5
