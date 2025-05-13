# pylint: disable=missing-module-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM import Formulae
from PySDM import physics
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics.isotope_kinetic_fractionation_factors import JouzelAndMerlivat1984


class TestIsotopeKineticFractionationFactors:
    @staticmethod
    def test_units():
        """checks that alphas are dimensionless"""
        with DimensionalAnalysis():
            # arrange
            alpha_eq = 1 * physics.si.dimensionless
            D_ratio = 1 * physics.si.dimensionless
            saturation_over_ice = 1 * physics.si.dimensionless

            # act
            sut = JouzelAndMerlivat1984.alpha_kinetic(
                alpha_equilibrium=alpha_eq,
                heavy_to_light_diffusivity_ratio=D_ratio,
                saturation_over_ice=saturation_over_ice,
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
        heavy_to_light_diffusivity_ratio = formulae.isotope_diffusivity_ratios.ratio_18O
        sut = formulae.isotope_kinetic_fractionation_factors.alpha_kinetic

        # act
        alpha_s = {temperature: alpha_s(temperature) for temperature in temperatures}
        alpha_k = {
            temperature: sut(
                alpha_equilibrium=alpha_s[temperature],
                saturation_over_ice=saturation,
                heavy_to_light_diffusivity_ratio=heavy_to_light_diffusivity_ratio(
                    temperature
                ),
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
        np.testing.assert_approx_equal(
            actual=alpha_s_times_alpha_k["-30C"][0],
            desired=1.021,
            significant=4,
        )
        np.testing.assert_approx_equal(
            actual=alpha_s_times_alpha_k["-30C"][-1],
            desired=1.0075,
            significant=4,
        )
        np.testing.assert_approx_equal(
            actual=alpha_s_times_alpha_k["-10C"][0],
            desired=1.0174,
            significant=4,
        )
        np.testing.assert_approx_equal(
            actual=alpha_s_times_alpha_k["-10C"][-1],
            desired=1.004,
            significant=4,
        )
