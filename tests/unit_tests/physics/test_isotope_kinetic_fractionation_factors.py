# pylint: disable=missing-module-docstring
import numpy as np
from matplotlib import pyplot
from PySDM import Formulae


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
            relative_humidity=saturation,
            heavy_to_light_diffusivity_ratio=heavy_to_light_diffusivity_ratio(
                temperature
            ),
        )
        for temperature in temperatures
    }

    # plot
    pyplot.xlim(saturation[0], saturation[-1])
    pyplot.ylim(1.003, 1.022)
    pyplot.xlabel("S")
    pyplot.ylabel("alpha_k (in the paper multiplied by alpha_s, here not!!!)")
    for temperature in temperatures:
        pyplot.plot(
            saturation,
            alpha_k[temperature],
            label=f"{formulae.trivia.K2C(temperature):.3g}C",
        )
    pyplot.legend()
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    assert (alpha_k[temperatures[0]] > alpha_k[temperatures[1]]).all()
    assert (alpha_k[temperatures[1]] > alpha_k[temperatures[-1]]).all()
    for temperature in temperatures:
        assert (np.diff(alpha_k[temperature]) < 0).all()
    np.testing.assert_approx_equal(
        actual=alpha_k[temperatures[0]][0],
        desired=1.021,
        significant=4,
    )
    np.testing.assert_approx_equal(
        actual=alpha_k[temperatures[0]][-1],
        desired=1.0075,
        significant=4,
    )
    np.testing.assert_approx_equal(
        actual=alpha_k[temperatures[-1]][1],
        desired=1.0174,
        significant=4,
    )
    np.testing.assert_approx_equal(
        actual=alpha_k[temperatures[-1]][-1],
        desired=1.004,
        significant=4,
    )
