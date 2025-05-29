"""
test for interpolated values of the pressure against Table 1,
test eq. 3 values
"""

import numpy as np
from matplotlib import pyplot
import pytest

from PySDM_examples.Jouzel_and_Merlivat_1984 import thermodynamic_profiles
from PySDM import Formulae
from PySDM.physics import si, in_unit, constants_defaults

PLOT = False


class TestThermodynamicProfiles:
    @staticmethod
    @pytest.mark.parametrize(
        ("temperature_C", "pressure"),
        ((-10, 925), (-20, 780), (-30, 690), (-40, 630), (-50, 600)),
    )
    def test_pressure_against_values_in_paper(temperature_C, pressure):
        # arrange
        temperature = temperature_C + constants_defaults.T0
        pressure_function = thermodynamic_profiles.pressure(temperature)

        # act
        pressure_mbar = in_unit(pressure_function, si.mbar)

        # assert
        np.testing.assert_allclose(desired=pressure, actual=pressure_mbar, atol=1e-4)

    @staticmethod
    def test_pressure_interpolation_plot(plot=PLOT):
        # arrange
        T = np.linspace(0, -60) + constants_defaults.T0
        p = thermodynamic_profiles.pressure(T)
        p_not_nan = p[~np.isnan(p)]
        # act
        sut = (p_not_nan[1:] - p_not_nan[:-1]) <= 0

        # plot
        pyplot.plot(T, p)
        pyplot.gca().set(
            ylabel="Pressure [Pa]",
            xlabel="Temperature [K]",
        )
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        np.testing.assert_equal(desired=1, actual=sut)

    @staticmethod
    @pytest.mark.parametrize(
        ("temperature_C", "ice_saturation_4"),
        ((-10, 1.05), (-20, 1.11), (-30, 1.17), (-40, 1.23), (-50, 1.30)),
    )
    def test_ice_saturation_curve_4_against_table_2(temperature_C, ice_saturation_4):
        # Arrange
        formulae = Formulae()
        const = formulae.constants
        T = temperature_C + const.T0

        # Act
        sut = thermodynamic_profiles.ice_saturation_curve_4(const, T)

        # Assert
        np.testing.assert_allclose(desired=ice_saturation_4, actual=sut, atol=0.01)
