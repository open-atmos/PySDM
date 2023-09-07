# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation

from PySDM.physics import si


class TestInitialCondition:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "particle_reservoir_depth",
        (
            0 * si.m,
            100 * si.m,
            200 * si.m,
            300 * si.m,
            400 * si.m,
            500 * si.m,
            600 * si.m,
        ),
    )
    def test_initial_condition(particle_reservoir_depth, plot=False):
        # Arrange
        settings = Settings(
            n_sd_per_gridbox=100,
            rho_times_w_1=2 * si.m / si.s * si.kg / si.m**3,
        )
        settings.particle_reservoir_depth = particle_reservoir_depth
        settings.t_max = 0 * settings.dt
        simulation = Simulation(settings)

        # Act
        output = simulation.run().products

        # Plot
        if plot:
            for var in ("RH", "T", "water_vapour_mixing_ratio", "p"):
                pyplot.plot(output[var][:], output["z"], linestyle="--", marker="o")
                if var == "water_vapour_mixing_ratio":
                    for value in (0.015, 0.0138, 0.0024):
                        pyplot.axvline(value)
                pyplot.ylabel("Z [m]")
                pyplot.title(f"reservoir depth: {particle_reservoir_depth} m")
                pyplot.xlabel(
                    var + " [" + simulation.particulator.products[var].unit + "]"
                )
                pyplot.grid()
                pyplot.show()

        # Assert
        for key in ("p", "T", "RH"):
            output[key] = output[key][
                int(settings.particle_reservoir_depth / settings.dz) :, 0
            ]
        assert output["RH"].shape == (int(settings.z_max // settings.dz),)

        assert 28 < np.amin(output["RH"]) < 32
        assert 96 < np.amax(output["RH"]) < 98

        assert 740 * si.hPa < np.amin(output["p"]) < 750 * si.hPa
        assert (np.diff(output["p"]) < 0).all()
        assert 1000 * si.hPa < np.amax(output["p"]) < 1010 * si.hPa

        assert 285 * si.K < np.amin(output["T"]) < 290 * si.K
        assert output["T"][0] > np.amin(output["T"])
        assert 295 * si.K < np.amax(output["T"]) < 300 * si.K
