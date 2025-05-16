"""
test for homogeneous nucleation rate parametrisations
"""

import pytest
import numpy as np
from PySDM.formulae import _choices, Formulae


class TestHomogeneousIceNucleationRate:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "da_w_ice, expected_value",
        (
            (0.27, (5)),
            (0.29, (11)),
            (0.31, (15)),
            (0.33, (20)),
        ),
    )
    @pytest.mark.parametrize("parametrisation", ("Koop_Correction",))
    def test_homogeneous_ice_nucleation_rate(da_w_ice, expected_value, parametrisation):
        """Fig. 2 in [Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023)"""
        # arrange
        formulae = Formulae(
            homogeneous_ice_nucleation_rate=parametrisation,
        )

        # act
        jhom_log10 = np.log10(
            formulae.homogeneous_ice_nucleation_rate.j_hom(np.nan, da_w_ice)
        )

        # assert
        np.testing.assert_approx_equal(
            actual=jhom_log10, desired=expected_value, significant=2
        )
