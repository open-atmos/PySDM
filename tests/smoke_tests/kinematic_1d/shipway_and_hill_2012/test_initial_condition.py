# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation

from PySDM.physics import si


class TestInitialCondition:
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

    @staticmethod
    @pytest.mark.parametrize(
        "old_buggy_density_formula",
        (pytest.param(True, marks=pytest.mark.xfail(strict=True)), False),
    )
    def test_density_profile(old_buggy_density_formula: bool, plot=False):
        """depicts a bug found in 2025 (thanks Clara!) in the density profile formula"""
        # arrange
        settings = Settings(
            old_buggy_density_formula=old_buggy_density_formula,
            n_sd_per_gridbox=0,
        )

        # act
        altitude = np.linspace(0, settings.z_max, 10)
        dry_air_density = settings.rhod(altitude)

        # plot
        pyplot.plot(dry_air_density, altitude)
        pyplot.ylabel("altitude [m]")
        pyplot.xlabel("dry-air density [kg m$^{-3}$]")
        pyplot.grid()
        pyplot.xlim(0.9, settings.rhod0)
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        np.testing.assert_approx_equal(
            actual=0.9029, desired=dry_air_density[-1], significant=4
        )
