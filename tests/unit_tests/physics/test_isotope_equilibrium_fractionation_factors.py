# pylint: disable=missing-module-docstring
import numpy as np
import pytest

from PySDM import Formulae


@pytest.mark.parametrize(
    "isotopologue, phase, expected_alpha_celsius_pairs",
    (
        ("HDO", "ice", {-120: 1.82, 0: 1.13}),
        ("HDO", "liquid", {-40: 1.2, 20: 1.08}),
        ("H2_18O", "ice", {-120: 1.05, 0: 1.015}),
        ("H2_18O", "liquid", {-40: 1.02, 20: 1.01}),
    ),
)
def test_isotope_equilibrium_fractionation_factors(
    isotopologue, phase, expected_alpha_celsius_pairs
):
    """values from Fig. 1 in [Bolot et al. 2013](https://10.5194/acp-13-7903-2013)"""
    # arrange
    formulae = Formulae()
    const = formulae.constants
    sut = getattr(
        formulae.isotope_equilibrium_fractionation_factors,
        f"alpha_{phase[0]}_{isotopologue}",
    )

    # act
    actual_pairs = {
        temp_celsius: sut(temp_celsius + const.T0)
        for temp_celsius in expected_alpha_celsius_pairs.keys()
    }

    # assert
    for k, v in expected_alpha_celsius_pairs.items():
        np.testing.assert_approx_equal(actual=actual_pairs[k], desired=v, significant=3)

    assert (
        np.diff(
            sut(
                const.T0
                + np.linspace(
                    tuple(expected_alpha_celsius_pairs.keys())[0],
                    tuple(expected_alpha_celsius_pairs.keys())[-1],
                    100,
                )
            )
        )
        < 0
    ).all()
