# pylint: disable=missing-module-docstring
import numpy as np
import pint
import pytest

from PySDM import Formulae
from PySDM.physics import constants_defaults, si
from PySDM.physics.dimensional_analysis import DimensionalAnalysis

CASES = (
    (
        "2H",
        "ice",
        {
            -120 * si.K + constants_defaults.T0: 1.82,
            0 * si.K + constants_defaults.T0: 1.13,
        },
    ),
    (
        "2H",
        "liquid",
        {
            -40 * si.K + constants_defaults.T0: 1.2,
            20 * si.K + constants_defaults.T0: 1.08,
        },
    ),
    (
        "18O",
        "ice",
        {
            -120 * si.K + constants_defaults.T0: 1.05,
            0 * si.K + constants_defaults.T0: 1.015,
        },
    ),
    (
        "18O",
        "liquid",
        {
            -40 * si.K + constants_defaults.T0: 1.02,
            20 * si.K + constants_defaults.T0: 1.01,
        },
    ),
)
"""values from Fig. 1 in [Bolot et al. 2013](https://doi.org/10.5194/acp-13-7903-2013)"""

PAPERS = ("MerlivatAndNief1967+Majoube1970+Majoube1971", "VanHook1968")

# TODO #1208: tests for VanHook1968 H2_17O, HOT


class TestIsotopeEquilibriumFractionationFactors:
    @staticmethod
    @pytest.mark.parametrize("paper", PAPERS)
    @pytest.mark.parametrize(
        "isotopologue, phase, expected_temperature_alpha_pairs", CASES
    )
    def test_values(paper, isotopologue, phase, expected_temperature_alpha_pairs):
        # arrange
        formulae = Formulae(isotope_equilibrium_fractionation_factors=paper)
        sut = getattr(
            formulae.isotope_equilibrium_fractionation_factors,
            f"alpha_{phase[0]}_{isotopologue}",
        )

        # act
        actual_pairs = {
            temp: sut(temp) for temp in expected_temperature_alpha_pairs.keys()
        }

        # assert
        for k, v in expected_temperature_alpha_pairs.items():
            np.testing.assert_approx_equal(
                actual=actual_pairs[k], desired=v, significant=2
            )

    @staticmethod
    @pytest.mark.parametrize("paper", PAPERS)
    @pytest.mark.parametrize(
        "isotopologue, phase, expected_temperature_alpha_pairs", CASES
    )
    def test_monotonic(paper, isotopologue, phase, expected_temperature_alpha_pairs):
        # arrange
        formulae = Formulae(isotope_equilibrium_fractionation_factors=paper)
        sut = getattr(
            formulae.isotope_equilibrium_fractionation_factors,
            f"alpha_{phase[0]}_{isotopologue}",
        )

        # act
        values = sut(
            np.linspace(
                tuple(expected_temperature_alpha_pairs.keys())[0],
                tuple(expected_temperature_alpha_pairs.keys())[-1],
                100,
            )
        )

        # assert
        assert (np.diff(values) < 0).all()

    @staticmethod
    @pytest.mark.parametrize("paper", PAPERS)
    @pytest.mark.parametrize("isotopologue, phase, _", CASES)
    @pytest.mark.parametrize(
        "argument",
        (
            pytest.param(
                "1 * si.J",
                marks=pytest.mark.xfail(
                    strict=True, raises=pint.errors.DimensionalityError
                ),
            ),
            "300 * si.K",
        ),
    )
    def test_units(paper, isotopologue, phase, _, argument):
        with DimensionalAnalysis():
            # arrange
            sut = getattr(
                Formulae(
                    isotope_equilibrium_fractionation_factors=paper
                ).isotope_equilibrium_fractionation_factors,
                f"alpha_{phase[0]}_{isotopologue}",
            )
            arg = eval(  # pylint: disable=eval-used
                argument, None, {"si": constants_defaults.si}
            )

            # act
            result = sut(arg)

            # assert
            assert result.dimensionless
