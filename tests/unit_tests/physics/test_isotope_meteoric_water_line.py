# pylint: disable=missing-module-docstring,missing-class-docstring
import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import in_unit
from PySDM.physics.constants_defaults import PER_MILLE, PER_CENT


class TestIsotopeMeteoricWaterLine:
    @staticmethod
    def test_craig_1961_science_fig_1(plot=False):
        """see Fig. 1 in [Craig 1961](https://doi.org/10.1126/science.133.3465.1702)"""

        # arrange
        const = Formulae().constants

        # act
        delta_18_oxygen = np.linspace(-50, 5) * const.PER_MILLE
        delta_2_hydrogen = (
            const.CRAIG_1961_SLOPE_COEFF * delta_18_oxygen
            + const.CRAIG_1961_INTERCEPT_COEFF
        )

        # plot
        pyplot.plot(
            in_unit(delta_18_oxygen, const.PER_MILLE),
            in_unit(delta_2_hydrogen, const.PER_MILLE),
            color="black",
        )
        pyplot.grid()
        pyplot.xlabel("$δ^{18}O$ [‰]")
        pyplot.ylabel("$δ^2H$ [‰]")
        pyplot.xlim(-50, 20)
        pyplot.ylim(-360, 100)
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert delta_2_hydrogen[0] == -390 * const.PER_MILLE
        assert delta_2_hydrogen[-1] == 50 * const.PER_MILLE

    @staticmethod
    def test_barkan_and_luz_2007_fig_4(plot=False):
        """see Fig. 4 in [Barkan and Luz 2007](http://doi.org/10.1002/rcm.3180)"""

        # arrange
        formulae = Formulae()
        const = formulae.constants

        # act
        delta_18_oxygen = np.linspace(0, 0.045)
        x_values = np.log(delta_18_oxygen + 1)
        y_values = const.BARKAN_AND_LUZ_2007_EXCESS_18O_COEFF * np.log(
            delta_18_oxygen + 1
        )

        # plot
        pyplot.plot(
            in_unit(x_values, const.PER_MILLE),
            in_unit(y_values, const.PER_MILLE),
            color="black",
        )
        pyplot.xlabel("$ln(δ^{18}O + 1)$, ‰")
        pyplot.ylabel("$ln(δ^{17}O + 1)$, ‰")
        pyplot.xlim(0, 50)
        pyplot.ylim(0, 25)
        pyplot.grid()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (0.0, 0.0) == (x_values[0], y_values[0])
        assert (44 * const.PER_MILLE, 23 * const.PER_MILLE) == (
            np.round(x_values[-1], 3),
            np.round(y_values[-1], 3),
        )

    @staticmethod
    @pytest.mark.parametrize(
        "delta_2H", (-30 * PER_MILLE, -10 * PER_MILLE, 10 * PER_MILLE, 30 * PER_MILLE)
    )
    def test_d18O_of_d2H(delta_2H):
        # arrange
        formulae = Formulae(isotope_meteoric_water_line="Dansgaard1964")
        sut = formulae.isotope_meteoric_water_line.d18O_of_d2H

        # act
        delta_18O = sut(delta_2H=delta_2H)

        # assert
        np.testing.assert_approx_equal(
            actual=formulae.isotope_meteoric_water_line.excess_d(
                delta_18O=delta_18O, delta_2H=delta_2H
            ),
            desired=formulae.constants.CRAIG_1961_INTERCEPT_COEFF,
            significant=3,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "delta_18O", (-3 * PER_MILLE, -1 * PER_MILLE, 1 * PER_MILLE, 3 * PER_MILLE)
    )
    def test_d17O_of_d18O(delta_18O):
        # arrange
        formulae = Formulae(isotope_meteoric_water_line="BarkanAndLuz2007")
        sut = formulae.isotope_meteoric_water_line.d17O_of_d18O

        # act
        delta_17O = sut(delta_18O=delta_18O)

        # assert
        np.testing.assert_almost_equal(
            actual=formulae.isotope_meteoric_water_line.excess_17O(
                delta_18O=delta_18O, delta_17O=delta_17O
            ),
            desired=0,
        )

    @staticmethod
    def test_picciotto_et_al_1960_fig_2(plot=False):
        # arrange
        delta_deuterium = np.linspace(-31, -10) * PER_CENT
        # act

        delta_18O = {
            paper: Formulae(
                isotope_meteoric_water_line=paper
            ).isotope_meteoric_water_line.d18O_of_d2H(delta_deuterium)
            for paper in ("PicciottoEtAl1960", "Dansgaard1964")
        }

        # plot
        pyplot.figure(figsize=(5, 6))
        for paper, delta in delta_18O.items():
            pyplot.plot(
                in_unit(delta_deuterium, PER_CENT),
                in_unit(delta, PER_MILLE),
                label=paper,
            )

        pyplot.xlabel(r"$\delta$ $^2$H [%]")
        pyplot.ylabel(r"$\delta$ $^{18}$O [‰]")

        pyplot.grid(color="k")
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        delta = delta_18O["PicciottoEtAl1960"]
        monotonic = (np.diff(delta) > 0).all()
        assert monotonic
        assert -37 < in_unit(delta[0], PER_MILLE) < -36
        assert -11 < in_unit(delta[-1], PER_MILLE) < -10

    @staticmethod
    @pytest.mark.parametrize(
        "delta_2H", (-30 * PER_MILLE, -10 * PER_MILLE, 10 * PER_MILLE, 30 * PER_MILLE)
    )
    def test_picciotto_et_al_1960_d18O_of_d2H(delta_2H):
        # arrange
        formulae = Formulae(isotope_meteoric_water_line="PicciottoEtAl1960")
        sut = formulae.isotope_meteoric_water_line.d18O_of_d2H

        # act
        delta_18O = sut(delta_2H=delta_2H)

        # assert
        np.testing.assert_approx_equal(
            actual=delta_2H
            - formulae.constants.PICCIOTTO_18O_TO_2H_SLOPE_COEFF * delta_18O,
            desired=formulae.constants.PICCIOTTO_18O_TO_2H_INTERCEPT_COEFF,
            significant=3,
        )
