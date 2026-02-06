"""Tests for formulae from [Merlivat and Jouzel 1979](https://doi.org/10.1029/JC084iC08p05029)"""

from functools import partial

import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import si
from PySDM.physics.constants import PER_MILLE

PLOT = False


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


class TestGedzelmanAndArnold1994:
    @staticmethod
    def test_saturation_for_zero_dR_condition(plot=True):
        # test unit?
        # test expected values
        # test values, plot?
        # nan to the left of equilibrium

        # arrange
        formulae = Formulae(
            drop_growth="Mason1971",
            isotope_ratio_evolution="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="HellmannAndHarvey2020",
            isotope_equilibrium_fractionation_factors="MerlivatAndNief1967",
        )
        phase = "liquid"
        const = formulae.constants
        T = 283.25 * si.K
        vsmow = const.VSMOW_R_2H

        x = np.linspace(0.8, 1.1, 200)
        alpha = formulae.isotope_equilibrium_fractionation_factors.alpha_l_2H(T)
        delta = -200 * PER_MILLE
        D_ratio_h2l = formulae.isotope_diffusivity_ratios.ratio_2H_heavy_to_light(T)
        Fk = formulae.drop_growth.Fk(T=T, K=const.K0, lv=const.l_tri)

        iso_ratio_v = formulae.trivia.isotopic_delta_2_ratio(delta, vsmow)
        iso_ratio_r = x * vsmow
        iso_ratio_liq_eq = alpha * iso_ratio_v / vsmow

        pvs = formulae.saturation_vapour_pressure.pvs_water(T)
        D_light = const.D0

        y = formulae.isotope_ratio_evolution.saturation_for_zero_dR_condition(
            diff_rat_light_to_heavy=1 / D_ratio_h2l,
            iso_ratio_x=iso_ratio_r if phase == "liquid" else iso_ratio_v,
            iso_ratio_r=iso_ratio_r,
            iso_ratio_v=iso_ratio_v,
            b=pvs * D_light * Fk,
            alpha_w=alpha,
        )
        # act
        if plot:
            pyplot.plot(x, y, "r")
            pyplot.plot(iso_ratio_liq_eq, 0, "ok")
            pyplot.xlabel("")
            pyplot.ylabel("saturation")
            pyplot.ylim(0, 0.01)
            pyplot.show()
        else:
            pyplot.close()

        # assert
        assert False
