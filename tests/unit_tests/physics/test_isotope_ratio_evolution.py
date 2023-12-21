""" Tests for formulae from [Merlivat and Jouzel 1979](https://doi.org/10.1029/JC084iC08p05029) """
from functools import partial

import numpy as np
from matplotlib import pyplot

from PySDM import Formulae


class TestMerlivatAndJouzel1979:
    @staticmethod
    def test_rayleigh_distillation_case(plot=False):
        """
        d_Rv/Rv = (alpha - 1) * d_n_vapour/n_vapour
        ln(Rv/Rv0) = (alpha - 1) * ln(nv/nv0)
        Rv = Rv0 * exp((a-1) * ln(nv/nv0))
        Rv = Rv0 * (nv/nv0)**(a-1)
        """

        # arrange
        d_Rv_over_Rv_M_J_1979 = partial(
            Formulae(
                isotope_ratio_evolution="MerlivatAndJouzel1979"
            ).isotope_ratio_evolution.d_Rv_over_Rv,
            d_alpha=0,
            n_liquid=0,
        )
        R_over_R0_Rayleigh = Formulae(
            isotope_ratio_evolution="RayleighDistillation"
        ).isotope_ratio_evolution.R_over_R0
        nv0 = 1
        alpha = 0.1
        Rv0 = 1

        # act
        nv, delta_nv = np.linspace(1, 1e-3, retstep=True)
        actual = Rv0 + delta_nv * np.cumsum(
            d_Rv_over_Rv_M_J_1979(alpha=alpha, n_vapour=nv, d_n_vapour=-delta_nv)
        )
        expect = Rv0 * R_over_R0_Rayleigh(X_over_X0=(nv / nv0), a=alpha)

        # plot
        pyplot.plot(nv, expect)
        pyplot.plot(nv, actual)

        if plot:
            pyplot.show()

        # assert
        # np.testing.assert_approx_equal(actual=actual, desired=expect, significant=10)

    @staticmethod
    def test_heilstone_expression():
        pass  # TODO
