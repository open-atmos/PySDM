"""tests checking homogeneous liquid nucleation formulae"""

import pytest

import numpy as np

from PySDM import Formulae
from PySDM.physics import si


class TestSeinfeldPandis:
    @staticmethod
    @pytest.mark.parametrize(
        "T, S, table_r_star",
        (
            (273, 2, 17.3e-10),
            (273, 3, 10.9e-10),
            (273, 4, 8.7e-10),
            (273, 5, 7.5e-10),
            (298, 2, 15.1e-10),
            (298, 3, 9.5e-10),
            (298, 4, 7.6e-10),
            (298, 5, 6.5e-10),
        ),
    )
    def test_seinfeld_pandis_table_11p1(T, S, table_r_star):
        formulae = Formulae(homogeneous_liquid_nucleation_rate="CNT")
        r_star = formulae.homogeneous_liquid_nucleation_rate.r_liq_homo(T, S)
        np.testing.assert_approx_equal(
            actual=r_star, desired=table_r_star * si.m, significant=1
        )

    @staticmethod
    @pytest.mark.parametrize(
        "T, S, table_J",
        (
            (293, 2, 5.02e-54),
            (293, 3, 1.76e-6),
            (293, 4, 1.05e6),
            (293, 5, 1.57e11),
            (293, 6, 1.24e14),
            (293, 7, 8.99e15),
            (293, 8, 1.79e17),
            (293, 9, 1.65e18),
            (293, 10, 9.17e18),
        ),
    )
    def test_seinfeld_pandis_table_11p4(T, S, table_J):
        formulae = Formulae(homogeneous_liquid_nucleation_rate="CNT")
        e_s = formulae.saturation_vapour_pressure.pvs_water(T)
        J = formulae.homogeneous_liquid_nucleation_rate.j_liq_homo(T, S, e_s)
        np.testing.assert_approx_equal(
            actual=J, desired=table_J / si.cm**3 / si.s, significant=1
        )
