""" Tests for formulae from [Merlivat and Jouzel 1979](https://doi.org/10.1029/JC084iC08p05029) """

from functools import partial

import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae


class TestMerlivatAndJouzel1979:
    @staticmethod
    @pytest.mark.parametrize("alpha", (0.1, 0.001))
    @pytest.mark.parametrize("start_stop", ((0.1, 1), (1, 0.1)))
    @pytest.mark.parametrize("Rv0", (0.01, 0.1))
    def test_rayleigh_distillation_case(alpha, start_stop, Rv0, plot=False):
        """
        d_Rv/Rv = (alpha - 1) * d_nv/nv
        ln(Rv/Rv0) = (alpha - 1) * ln(nv/nv0)
        Rv = Rv0 * exp((alpha - 1) * ln(nv/nv0))
        Rv = Rv0 * (nv/nv0)**(alpha - 1)
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

        num_points = 77
        assert num_points % 2 == 1

        # act
        nv, delta_nv = np.linspace(*start_stop, num=num_points, retstep=True)
        dRoR_m_j_1979 = d_Rv_over_Rv_M_J_1979(
            alpha=alpha, n_vapour=nv, d_n_vapour=delta_nv
        )

        nv_over_nv0 = nv / nv[0]

        R = Rv0 * R_over_R0_Rayleigh(X_over_X0=nv_over_nv0, a=alpha)
        dRoR_rayleigh = np.diff(R[::2]) / 2 / (R[1::2])

        # plot
        pyplot.ylabel("dR/R")
        pyplot.xlabel("nv/nv$_0$")
        pyplot.plot(
            nv_over_nv0, dRoR_m_j_1979, label="Merlivat & Jouzel '79", marker="x"
        )
        pyplot.scatter(
            nv_over_nv0[1::2], dRoR_rayleigh, label="Rayleigh", marker="o", color="red"
        )
        pyplot.legend()
        # pyplot.xscale('log')
        # pyplot.yscale('log')
        pyplot.grid()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        np.testing.assert_almost_equal(
            desired=dRoR_rayleigh, actual=dRoR_m_j_1979[1::2], decimal=3
        )

    @staticmethod
    def test_heilstone_expression():
        pass  # TODO #1206
