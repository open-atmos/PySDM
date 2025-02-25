# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

import numpy as np

from PySDM import Formulae
from PySDM.physics import si, constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestHomogeneousLiquidNucleation:
    @staticmethod
    @pytest.mark.parametrize(
        "temperature, saturation_ratio, sgm_w, table_r_star",
        (
            (273 * si.K, 2, 75.6 * si.dyn / si.cm, 17.3e-10 * si.m),
            (273 * si.K, 3, 75.6 * si.dyn / si.cm, 10.9e-10 * si.m),
            (273 * si.K, 4, 75.6 * si.dyn / si.cm, 8.70e-10 * si.m),
            (273 * si.K, 5, 75.6 * si.dyn / si.cm, 7.50e-10 * si.m),
            (298 * si.K, 2, 72.0 * si.dyn / si.cm, 15.1e-10 * si.m),
            (298 * si.K, 3, 72.0 * si.dyn / si.cm, 9.50e-10 * si.m),
            (298 * si.K, 4, 72.0 * si.dyn / si.cm, 7.60e-10 * si.m),
            (298 * si.K, 5, 72.0 * si.dyn / si.cm, 6.50e-10 * si.m),
        ),
    )
    def test_seinfeld_pandis_table_11p1(
        temperature, saturation_ratio, sgm_w, table_r_star
    ):
        # arrange
        formulae = Formulae(
            homogeneous_liquid_nucleation_rate="CNT", constants={"sgm_w": sgm_w}
        )

        # act
        r_star = formulae.homogeneous_liquid_nucleation_rate.r_liq_homo(
            temperature, saturation_ratio
        )

        # assert
        np.testing.assert_approx_equal(
            actual=r_star, desired=table_r_star, significant=2
        )

    @staticmethod
    @pytest.mark.parametrize(
        "saturation_ratio, desired_j",
        (
            (+2, 5.02e-54 / si.cm**3 / si.s),
            (+3, 1.76e-06 / si.cm**3 / si.s),
            (+4, 1.05e06 / si.cm**3 / si.s),
            (+5, 1.57e11 / si.cm**3 / si.s),
            (+6, 1.24e14 / si.cm**3 / si.s),
            (+7, 8.99e15 / si.cm**3 / si.s),
            (+8, 1.79e17 / si.cm**3 / si.s),
            (+9, 1.65e18 / si.cm**3 / si.s),
            (10, 9.17e18 / si.cm**3 / si.s),
        ),
    )
    def test_seinfeld_pandis_table_11p4(saturation_ratio, desired_j):
        # arrange
        temperature = 293 * si.K
        e_s = 23365 * si.g / si.cm / si.s**2
        formulae = Formulae(
            homogeneous_liquid_nucleation_rate="CNT",
            constants={"sgm_w": 72.75 * si.dyn / si.cm},
        )

        # act
        actual_j = formulae.homogeneous_liquid_nucleation_rate.j_liq_homo(
            temperature, saturation_ratio, e_s
        )

        # assert
        np.testing.assert_allclose(
            actual=np.log10(actual_j),
            desired=np.log10(desired_j),
            rtol=0.25,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "formulae_factory",
        (
            pytest.param(
                lambda _si: Formulae(homogeneous_liquid_nucleation_rate="CNT"), id="CNT"
            ),
            pytest.param(
                lambda _si: Formulae(
                    homogeneous_liquid_nucleation_rate="Constant",
                    constants={
                        "J_LIQ_HOMO": 1 / _si.s / _si.m**3,
                        "R_LIQ_HOMO": 1 * _si.m,
                    },
                ),
                id="Constant",
            ),
        ),
    )
    @pytest.mark.parametrize("check", ("r", "j"))
    def test_units(formulae_factory, check):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si  # pylint: disable=redefined-outer-name
            sut = formulae_factory(si).homogeneous_liquid_nucleation_rate

            # act & assert
            if check == "j":
                assert sut.j_liq_homo(
                    T=300 * si.K, S=2 * si.dimensionless, e_s=600 * si.Pa
                ).check("[frequency] / [volume]")
            elif check == "r":
                assert sut.r_liq_homo(T=300 * si.K, S=2 * si.dimensionless).check(
                    "[length]"
                )
            else:
                assert False
