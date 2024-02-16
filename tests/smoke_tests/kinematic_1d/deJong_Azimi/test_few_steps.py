# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.deJong_Azimi import Settings1D
from PySDM_examples.Shipway_and_Hill_2012 import Simulation
from scipy.ndimage import uniform_filter1d

from PySDM.physics import si


@pytest.mark.parametrize(
    "z_part",
    (
        [0.0, 1.0],
        [0.2, 0.8],
    ),
)
def test_few_steps(z_part, plot=False):
    # Arrange
    n_sd_per_gridbox = 128
    dt = 30 * si.s
    smooth_window = 5
    settings = Settings1D(
        n_sd_per_gridbox=n_sd_per_gridbox,
        dt=dt,
        dz=60 * si.m,
        rho_times_w_1=0 * si.m / si.s * si.kg / si.m**3,
        z_part=z_part,
        t_max=50 * dt,
    )
    settings.condensation_update_thd = True
    simulation = Simulation(settings)

    # Act
    output = simulation.run().products

    # Plot
    def mean_profile_over_last_steps(var, smooth=True):
        data = np.mean(output[var][output["z"] >= 0, -10:], axis=1)
        if not smooth:
            return data
        return uniform_filter1d(data, size=smooth_window)

    for var in (
        "collision_rate",
        "nr",
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
    assert max(mean_profile_over_last_steps("collision_rate")) > 0
