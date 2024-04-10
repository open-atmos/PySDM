""" tests for consistency of values taken from the Supplement """

import numpy as np
from PySDM_examples.Pierchala_et_al_2022.commons import deltas_0_SMOW

from PySDM import Formulae


def test_cracow_water_excesses():
    """checking if d-excess and 17O-excess values match those computed from deltas"""
    # arrange
    formulae = Formulae(
        isotope_meteoric_water_line_excess="Dansgaard1964+BarkanAndLuz2007"
    )
    sut = formulae.isotope_meteoric_water_line_excess

    # act/assert
    np.testing.assert_approx_equal(
        actual=sut.excess_d(deltas_0_SMOW["2H"], deltas_0_SMOW["18O"]),
        desired=7.68 * formulae.constants.PER_MILLE,
        significant=3,
    )
    np.testing.assert_approx_equal(
        actual=sut.excess_17O(deltas_0_SMOW["17O"], deltas_0_SMOW["18O"]),
        desired=29 * formulae.constants.PER_MEG,
        significant=3,
    )
