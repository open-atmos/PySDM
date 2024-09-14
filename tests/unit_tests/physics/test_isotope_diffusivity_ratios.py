"""
tests for isotope diffusivity ratio formulae
"""

import numpy as np
import pytest
from matplotlib import pyplot

from PySDM.physics import constants_defaults, si, isotope_diffusivity_ratios
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.formulae import _choices, Formulae


class TestIsotopeDiffusivityRatios:
    @staticmethod
    @pytest.mark.parametrize(
        "isotope, expected_value",
        (
            ("2H", 0.9835),
            ("18O", 0.9687),
        ),
    )
    def test_stewart_1975_values_given_in_horita_et_al_2008(isotope, expected_value):
        """test against values given below eq. (22) in
        [Horita et al. 2008](https://doi.org/10.1080/10256010801887174)"""
        # arrange
        formulae = Formulae(
            constants={"Md": constants_defaults.Md * 0.9986},
            isotope_diffusivity_ratios="Stewart1975",
        )
        sut = getattr(formulae.isotope_diffusivity_ratios, f"ratio_{isotope}")

        # act
        actual_value = sut(temperature=np.nan)

        # assert
        np.testing.assert_approx_equal(
            actual=actual_value, desired=expected_value, significant=4
        )

    @staticmethod
    @pytest.mark.parametrize(
        "temperature, expected_values",
        (
            (190 * si.K, (0.9740, 0.9850, 0.9714)),
            (200 * si.K, (0.9741, 0.9850, 0.9713)),
            (210 * si.K, (0.9743, 0.9850, 0.9713)),
            (220 * si.K, (0.9744, 0.9849, 0.9712)),
            (230 * si.K, (0.9745, 0.9849, 0.9712)),
            (240 * si.K, (0.9747, 0.9849, 0.9711)),
            (250 * si.K, (0.9748, 0.9849, 0.9711)),
            (260 * si.K, (0.9750, 0.9848, 0.9710)),
            (270 * si.K, (0.9752, 0.9848, 0.9710)),
            (280 * si.K, (0.9753, 0.9848, 0.9709)),
            (290 * si.K, (0.9755, 0.9848, 0.9709)),
            (300 * si.K, (0.9756, 0.9847, 0.9708)),
            (310 * si.K, (0.9758, 0.9847, 0.9708)),
            (320 * si.K, (0.9759, 0.9847, 0.9707)),
            (330 * si.K, (0.9761, 0.9847, 0.9706)),
            (340 * si.K, (0.9762, 0.9847, 0.9706)),
            (360 * si.K, (0.9765, 0.9846, 0.9705)),
            (380 * si.K, (0.9768, 0.9846, 0.9704)),
            (400 * si.K, (0.9770, 0.9845, 0.9703)),
            (450 * si.K, (0.9775, 0.9845, 0.9701)),
            (500 * si.K, (0.9779, 0.9844, 0.9700)),
        ),
    )
    @pytest.mark.parametrize(
        "isotope_index, isotope_label", enumerate(("2H", "17O", "18O"))
    )
    def test_hellmann_and_harvey_2020_table_1(
        temperature, expected_values, isotope_index, isotope_label
    ):
        # arrange
        formulae = Formulae(isotope_diffusivity_ratios="HellmannAndHarvey2020")
        sut = getattr(formulae.isotope_diffusivity_ratios, f"ratio_{isotope_label}")

        # act
        actual_value = sut(temperature=temperature)

        # assert
        np.testing.assert_approx_equal(
            actual=actual_value, desired=expected_values[isotope_index], significant=4
        )

    @staticmethod
    def test_grahams_law():
        # arrange
        formulae = Formulae(isotope_diffusivity_ratios="GrahamsLaw")

        # act
        sut = formulae.isotope_diffusivity_ratios.ratio_2H(temperature=np.nan)

        # assert
        np.testing.assert_approx_equal(sut, 0.973, significant=3)

    @staticmethod
    @pytest.mark.parametrize("paper", _choices(isotope_diffusivity_ratios))
    @pytest.mark.parametrize("isotope_label", ("2H", "17O", "18O"))
    def test_unit(paper, isotope_label):
        with DimensionalAnalysis():
            # arrange
            try:
                sut = getattr(
                    Formulae(
                        isotope_diffusivity_ratios=paper
                    ).isotope_diffusivity_ratios,
                    f"ratio_{isotope_label}",
                )
            except AttributeError:
                pytest.skip()

            # act
            result = sut(temperature=300 * constants_defaults.si.K)

            # assert
            assert result.dimensionless

    @staticmethod
    def test_all_on_one_plot(plot=False):
        temperature = np.linspace(270, 300) * si.K
        min_value, max_value = np.inf, -np.inf
        for paper in _choices(isotope_diffusivity_ratios):
            formulae = Formulae(isotope_diffusivity_ratios=paper)
            for isotope_label in ("2H", "17O", "18O"):
                try:
                    sut = getattr(
                        formulae.isotope_diffusivity_ratios, f"ratio_{isotope_label}"
                    )
                except AttributeError:
                    pass
                else:
                    diffusivity_ratio = sut(temperature)
                    min_value = min(np.amin(diffusivity_ratio), min_value)
                    max_value = max(np.amax(diffusivity_ratio), max_value)
                    pyplot.plot(
                        temperature,
                        (
                            diffusivity_ratio
                            if isinstance(diffusivity_ratio, np.ndarray)
                            else np.full_like(temperature, diffusivity_ratio)
                        ),
                        label=f"{paper=} {isotope_label=}",
                    )
        pyplot.xlabel("temperature [K]")
        pyplot.grid()
        pyplot.ylabel("diffusivity ratio (heavy to light)")
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert 0.985 > max_value > min_value > 0.968
