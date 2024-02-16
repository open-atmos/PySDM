# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.deJong_Azimi import Settings1D
from PySDM_examples.Shipway_and_Hill_2012 import Simulation

from PySDM.physics import si


class TestInitialCondition:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "z_part",
        ([0.0, 1.0], [0.2, 0.8], [0.0, 0.5], [0.5, 1.0]),
    )
    def test_initial_condition(z_part, plot=False):
        # Arrange
        settings = Settings1D(
            n_sd_per_gridbox=10,
            rho_times_w_1=0 * si.m / si.s * si.kg / si.m**3,
            z_part=z_part,
            t_max=0,
        )
        simulation = Simulation(settings)

        # Act
        output = simulation.run().products

        # Plot
        if plot:
            for var in ("nc", "nr"):
                pyplot.plot(output[var][:], output["z"], linestyle="--", marker="o")
                pyplot.ylabel("Z [m]")
                pyplot.title(f"z_part: {z_part} m")
                pyplot.xlabel(
                    var + " [" + simulation.particulator.products[var].unit + "]"
                )
                pyplot.grid()
                pyplot.show()

        # Assert
        nz = settings.z_max / settings.dz
        has_particles = np.zeros_like(output["nc"])
        has_particles[int(nz * z_part[0]) : int((nz + 1) * z_part[1])] = 1
        np.testing.assert_array_equal(output["nc"] + output["nr"] > 0, has_particles)
