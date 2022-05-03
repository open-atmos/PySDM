# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation
from scipy.ndimage.filters import uniform_filter1d

from PySDM.physics import si


@pytest.mark.parametrize(
    "particle_reservoir_depth",
    (
        pytest.param(0 * si.m, marks=pytest.mark.xfail(strict=True)),
        900 * si.m,
    ),
)
def test_few_steps_no_precip(particle_reservoir_depth, plot=False):
    # Arrange
    n_sd_per_gridbox = 128
    smooth_window = 5
    settings = Settings(
        n_sd_per_gridbox=n_sd_per_gridbox,
        dt=30 * si.s,
        dz=60 * si.m,
        precip=False,
        particle_reservoir_depth=particle_reservoir_depth,
        rho_times_w_1=2 * si.m / si.s * si.kg / si.m**3,
    )
    simulation = Simulation(settings)

    # Act
    output = simulation.run(nt=50)

    # Plot
    def mean_profile_over_last_steps(var, smooth=True):
        data = np.mean(output[var][output["z"] >= 0, -10:], axis=1)
        if not smooth:
            return data
        return uniform_filter1d(data, size=smooth_window)

    for var in (
        "RH",
        "peak supersaturation",
        "T",
        "qv",
        "p",
        "ql",
        "ripening rate",
        "activating rate",
        "deactivating rate",
        "super droplet count per gridbox",
        "na",
        "nc",
    ):
        z = output["z"][output["z"] >= 0]
        pyplot.plot(
            mean_profile_over_last_steps(var, smooth=False),
            z,
            linestyle="--",
            marker="o",
        )
        pyplot.plot(mean_profile_over_last_steps(var), z)
        pyplot.ylabel("Z [m]")
        pyplot.xlabel(var + " [" + simulation.particulator.products[var].unit + "]")
        pyplot.grid()
        if plot:
            pyplot.show()

    # Assert
    sd_prof = mean_profile_over_last_steps("super droplet count per gridbox")
    assert 0.5 * n_sd_per_gridbox < min(sd_prof) < 1.5 * n_sd_per_gridbox
    assert 0.5 * n_sd_per_gridbox < max(sd_prof) < 1.5 * n_sd_per_gridbox

    assert 0.01 < max(mean_profile_over_last_steps("peak supersaturation")) < 0.1
    assert min(mean_profile_over_last_steps("ql")) < 1e-10
    assert 0.15 < max(mean_profile_over_last_steps("ql")) < 0.25
    assert max(mean_profile_over_last_steps("activating rate")) == 0

    assert max(mean_profile_over_last_steps("ripening rate")) > 0
    assert max(mean_profile_over_last_steps("deactivating rate")) > 0
