# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from PySDM_examples.Jensen_and_Nugent_2017.table_3 import RD, NA
import numpy as np
from PySDM.physics import si
from PySDM import Formulae


TRIVIA = Formulae().trivia

# from Wikipedia
RHO_S = 2.17 * si.g / si.cm**3


class TestTable3:
    @staticmethod
    def test_number_integral():
        np.testing.assert_approx_equal(
            actual=np.sum(NA), desired=281700 / si.m**3, significant=4
        )

    @staticmethod
    def test_mass_integral():
        total_mass_concentration = np.dot(TRIVIA.volume(radius=RD), NA) * RHO_S
        np.testing.assert_approx_equal(
            actual=total_mass_concentration,
            desired=7.3 * si.ug / si.m**3,
            significant=2,
        )
