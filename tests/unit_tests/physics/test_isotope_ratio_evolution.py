"""Tests for formulae from [Merlivat and Jouzel 1979](https://doi.org/10.1029/JC084iC08p05029)"""

from functools import partial

import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import si, constants_defaults
from PySDM.physics.constants import PER_MILLE
from PySDM.physics.dimensional_analysis import DimensionalAnalysis

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
    @pytest.mark.parametrize("phase", ("liquid", "vapour"))
    @pytest.mark.parametrize("isotope", ("2H", "18O", "17O"))
    def test_saturation_for_zero_dR_condition(phase, isotope, plot=PLOT):
        # arrange
        formulae = Formulae(
            drop_growth="Mason1971",
            isotope_ratio_evolution="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="HellmannAndHarvey2020",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        const = formulae.constants
        T = formulae.trivia.C2K(10) * si.K
        vsmow = getattr(const, f"VSMOW_R_{isotope}")
        delta = -200 * PER_MILLE

        D_ratio_h2l = getattr(
            formulae.isotope_diffusivity_ratios, f"ratio_{isotope}_heavy_to_light"
        )(T)
        alpha = getattr(
            formulae.isotope_equilibrium_fractionation_factors, f"alpha_l_{isotope}"
        )(T)

        Fk = formulae.drop_growth.Fk(T=T, K=const.K0, lv=const.l_tri)
        rho_v = formulae.saturation_vapour_pressure.pvs_water(T) / T / const.Rv
        b = rho_v * const.D0 * Fk

        iso_ratio_v = formulae.trivia.isotopic_delta_2_ratio(delta, vsmow)
        iso_ratio_liq_eq = alpha * iso_ratio_v / vsmow

        x = np.linspace(iso_ratio_liq_eq - 0.5, iso_ratio_liq_eq + 0.5, 300)
        dx = np.diff(x).mean()
        iso_ratio_r = x * vsmow

        # act
        y = formulae.isotope_ratio_evolution.saturation_for_zero_dR_condition(
            diff_rat_light_to_heavy=1 / D_ratio_h2l,
            iso_ratio_x=iso_ratio_r if phase == "liquid" else iso_ratio_v,
            iso_ratio_r=iso_ratio_r,
            iso_ratio_v=iso_ratio_v,
            b=b,
            alpha_w=alpha,
        )

        # plot
        if plot:
            pyplot.plot(x, y, "k")
            pyplot.plot(iso_ratio_liq_eq, 0, "or")
            pyplot.xlabel("isotopic ratio in droplets / vsmow")
            pyplot.ylabel("saturation")
            pyplot.grid()
            pyplot.show()
        else:
            pyplot.clf()
        # assert
        eps = max(1e-2, 3 * dx)
        outside_radius = max(0.025, 10 * dx, 2 * eps)

        left_near = (x < iso_ratio_liq_eq) & (x >= iso_ratio_liq_eq - eps)
        right_near = (x > iso_ratio_liq_eq) & (x <= iso_ratio_liq_eq + eps)
        outside_mask = np.abs(x - iso_ratio_liq_eq) > outside_radius

        assert left_near.any(), "No points found just left of equilibrium"
        assert right_near.any(), "No points found just right of equilibrium"
        assert outside_mask.any(), "No points found far from equilibrium"

        np.testing.assert_array_less(
            y[left_near],
            0,
            err_msg="Values immediately left of equilibrium should be negative",
        )
        np.testing.assert_array_less(
            0,
            y[right_near],
            err_msg="Values immediately right of equilibrium should be positive",
        )
        np.testing.assert_allclose(
            y[outside_mask],
            0,
            atol=1e-2,
            rtol=1e-2,
            err_msg="Values far from equilibrium should approach zero",
        )

    @staticmethod
    @pytest.mark.parametrize("phase", ("liquid", "vapour"))
    def test_unit_saturation_for_zero_dR_condition(phase):
        with DimensionalAnalysis():

            # arrange
            formulae = Formulae(isotope_ratio_evolution="GedzelmanAndArnold1994")
            si = constants_defaults.si

            iso_ratio_v = 0.2 * si.dimensionless
            iso_ratio_r = 0.1 * si.dimensionless
            sut = formulae.isotope_ratio_evolution.saturation_for_zero_dR_condition

            # act
            S = sut(
                diff_rat_light_to_heavy=1.1 * si.dimensionless,
                iso_ratio_x=iso_ratio_r if phase == "liquid" else iso_ratio_v,
                iso_ratio_r=iso_ratio_r,
                iso_ratio_v=iso_ratio_v,
                b=1 * si.dimensionless,
                alpha_w=1 * si.dimensionless,
            )

            # assert
            assert S.check(si.dimensionless)
