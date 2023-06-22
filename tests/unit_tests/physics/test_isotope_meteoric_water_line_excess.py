# pylint: disable=missing-module-docstring,missing-class-docstring
import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import in_unit


class TestIsotopeMeteoricWaterLineExcess:
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
