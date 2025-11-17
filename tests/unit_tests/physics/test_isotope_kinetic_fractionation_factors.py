"""
test for isotope kinetic fractionation factors based on plot
"""

import numpy as np
import pytest
from matplotlib import pyplot

from PySDM import Formulae
from PySDM import physics
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics.isotope_kinetic_fractionation_factors import JouzelAndMerlivat1984

PLOT = False


class TestIsotopeKineticFractionationFactors:
    @staticmethod
    def test_units_alpha_kinetic():
        """checks that alphas are dimensionless"""
        with DimensionalAnalysis():
            # arrange
            alpha_eq = 1 * physics.si.dimensionless
            D_ratio = 1 * physics.si.dimensionless
            saturation_over_ice = 1 * physics.si.dimensionless

            # act
            sut = JouzelAndMerlivat1984.alpha_kinetic(
                alpha_equilibrium=alpha_eq,
                D_ratio_heavy_to_light=D_ratio,
                saturation=saturation_over_ice,
            )

            # assert
            assert sut.check("[]")

    @staticmethod
    def test_fig_9_from_jouzel_and_merlivat_1984(plot=False):
        """[Jouzel & Merlivat 1984](https://doi.org/10.1029/JD089iD07p11749)"""
        # arrange
        formulae = Formulae(
            isotope_kinetic_fractionation_factors="JouzelAndMerlivat1984",
            isotope_equilibrium_fractionation_factors="Majoube1970",
            isotope_diffusivity_ratios="Stewart1975",
        )
        temperatures = formulae.trivia.C2K(np.asarray([-30, -20, -10]))
        saturation = np.linspace(start=1, stop=1.35)
        alpha_s = formulae.isotope_equilibrium_fractionation_factors.alpha_i_18O
        diffusivity_ratio_heavy_to_light = (
            formulae.isotope_diffusivity_ratios.ratio_18O_heavy_to_light
        )
        sut = formulae.isotope_kinetic_fractionation_factors.alpha_kinetic

        # act
        alpha_s = {temperature: alpha_s(temperature) for temperature in temperatures}
        alpha_k = {
            temperature: sut(
                alpha_equilibrium=alpha_s[temperature],
                saturation=saturation,
                D_ratio_heavy_to_light=diffusivity_ratio_heavy_to_light(temperature),
            )
            for temperature in temperatures
        }
        alpha_s_times_alpha_k = {
            f"{formulae.trivia.K2C(temperature):.3g}C": alpha_k[temperature]
            * alpha_s[temperature]
            for temperature in temperatures
        }

        # plot
        pyplot.xlim(saturation[0], saturation[-1])
        pyplot.ylim(1.003, 1.022)
        pyplot.xlabel("S")
        pyplot.ylabel("alpha_k * alpha_s")
        for k, v in alpha_s_times_alpha_k.items():
            pyplot.plot(saturation, v, label=k)
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (alpha_s_times_alpha_k["-30C"] > alpha_s_times_alpha_k["-20C"]).all()
        assert (alpha_s_times_alpha_k["-20C"] > alpha_s_times_alpha_k["-10C"]).all()
        for alpha_alpha in alpha_s_times_alpha_k.values():
            assert (np.diff(alpha_alpha) < 0).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("temperature_C", "saturation", "alpha"),
        ((-10, 1, 1.021), (-10, 1.35, 1.0075), (-30, 1, 1.0174), (-30, 1.35, 1.004)),
    )
    def test_fig9_values(temperature_C, saturation, alpha):
        # arrange
        formulae = Formulae(
            isotope_kinetic_fractionation_factors="JouzelAndMerlivat1984",
            isotope_equilibrium_fractionation_factors="Majoube1970",
            isotope_diffusivity_ratios="Stewart1975",
        )
        diffusivity_ratio_18O = (
            formulae.isotope_diffusivity_ratios.ratio_18O_heavy_to_light
        )
        T = formulae.trivia.C2K(temperature_C)
        alpha_s = formulae.isotope_equilibrium_fractionation_factors.alpha_i_18O(T)
        alpha_k = formulae.isotope_kinetic_fractionation_factors.alpha_kinetic(
            alpha_equilibrium=alpha_s,
            saturation=saturation,
            D_ratio_heavy_to_light=diffusivity_ratio_18O(T),
        )

        # act
        sut = alpha_s * alpha_k

        # assert
        np.testing.assert_approx_equal(actual=sut, desired=alpha, significant=3)

    @staticmethod
    @pytest.mark.parametrize("isotope", ("2H", "18O", "17O"))
    @pytest.mark.parametrize("temperature_C", (-30, -20, -1))
    def test_alpha_kinetic_jouzel_merlivat_vs_craig_gordon(
        isotope, temperature_C, plot=PLOT
    ):
        # arrange
        T = Formulae().trivia.C2K(temperature_C)
        RH = np.linspace(0.3, 1)
        formulae = Formulae(
            isotope_equilibrium_fractionation_factors="VanHook1968",
            isotope_diffusivity_ratios="HellmannAndHarvey2020",
            isotope_kinetic_fractionation_factors="JouzelAndMerlivat1984",
        )
        Si = formulae.saturation_vapour_pressure.pvs_ice(T)
        alpha_eq = getattr(
            formulae.isotope_equilibrium_fractionation_factors, f"alpha_l_{isotope}"
        )(T)
        D_heavy_to_light = getattr(
            formulae.isotope_diffusivity_ratios, f"ratio_{isotope}_heavy_to_light"
        )(T)
        alpha_kin_jm = formulae.isotope_kinetic_fractionation_factors.alpha_kinetic(
            alpha_equilibrium=alpha_eq,
            saturation=Si,
            D_ratio_heavy_to_light=D_heavy_to_light,
        )
        formulae = Formulae(
            isotope_kinetic_fractionation_factors="CraigGordon",
        )
        alpha_kin_cg = formulae.isotope_kinetic_fractionation_factors.alpha_kinetic(
            relative_humidity=RH,
            turbulence_parameter_n=1,
            delta_diff=alpha_eq - 1,
            theta=1,
        )

        # act
        n = (alpha_kin_jm + 1) / (alpha_kin_cg + 1)

        # plot
        pyplot.plot(1 - RH, n)
        pyplot.gca().set(
            xlabel="1-RH",
            ylabel="turbulence parameter n",
        )
        pyplot.grid()

        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        np.testing.assert_equal(n > 0.5, True)
        np.testing.assert_equal(n < 1, True)
