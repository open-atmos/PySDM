# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import importlib
import os

import numpy as np
import pint
import pytest
from scipy.constants import physical_constants

from PySDM.physics import constants, constants_defaults, si


def consecutive_seeds():
    seeds = []
    for _ in range(5):
        importlib.reload(constants)
        seeds.append(constants.default_random_seed)
    return np.asarray(seeds)


class TestConstants:
    @staticmethod
    def test_constant_seed_on_CI():
        CI = "CI" in os.environ
        if not CI:
            os.environ["CI"] = "1"
        seeds = consecutive_seeds()
        if not CI:
            del os.environ["CI"]
        assert (seeds == seeds[0]).all()

    @staticmethod
    def test_variable_seed_outside_of_CI():
        CI = "CI" in os.environ
        if CI:
            CI = os.environ["CI"]
            del os.environ["CI"]
        seeds = consecutive_seeds()
        if CI:
            os.environ["CI"] = CI
        assert (seeds[1:] != seeds[0]).any()

    @staticmethod
    def test_standard_atmosphere_p():
        # arrange
        pint_si = pint.UnitRegistry()

        # act
        p = constants_defaults.p_STP * pint_si.Pa

        # assert
        assert p == 1 * pint_si.atm

    @staticmethod
    @pytest.mark.parametrize(
        "item, value",
        (
            (
                "M_1H",
                1.007825 * si.g / si.mole,
            ),  # https://en.wikipedia.org/wiki/Hydrogen-1
            (
                "M_2H",
                2.01410177811 * si.g / si.mole,
            ),  # https://en.wikipedia.org/wiki/Hydrogen-2
            (
                "M_3H",
                3.01604928 * si.g / si.mole,
            ),  # https://en.wikipedia.org/wiki/Hydrogen-3
            (
                "M_1H",
                (
                    physical_constants["proton molar mass"][0]
                    + physical_constants["electron molar mass"][0]
                )
                * si.kg
                / si.mole,
            ),
            (
                "M_2H",
                (
                    physical_constants["deuteron molar mass"][0]
                    + physical_constants["electron molar mass"][0]
                )
                * si.kg
                / si.mole,
            ),
            (
                "M_3H",
                (
                    physical_constants["triton molar mass"][0]
                    + physical_constants["electron molar mass"][0]
                )
                * si.kg
                / si.mole,
            ),
            (
                "M_16O",
                15.99491461956 * si.g / si.mole,
            ),  # https://en.wikipedia.org/wiki/Oxygen-16
            (
                "M_17O",
                16.9991315 * si.g / si.mole,
            ),  # https://en.wikipedia.org/wiki/Oxygen-17
            (
                "M_18O",
                17.9991610 * si.g / si.mole,
            ),  # https://en.wikipedia.org/wiki/Oxygen-18
        ),
    )
    def test_isotope_molar_masses_vs_wikipedia_or_scipy(item, value):
        np.testing.assert_approx_equal(
            actual=getattr(constants_defaults, item), desired=value, significant=7
        )

    def test_isotope_molar_masses_vsmow_vs_mean_water_molar_mass(self):
        cd = constants_defaults
        n_H2O = 1 / (
            1 + cd.VSMOW_R_2H / 2 + cd.VSMOW_R_2H / 2 + cd.VSMOW_R_17O + cd.VSMOW_R_18O
        )
        np.testing.assert_approx_equal(
            desired=constants_defaults.Mv,
            actual=(
                (cd.M_1H * 2 + cd.M_16O) * n_H2O
                + (cd.M_2H * 2 + cd.M_16O) * cd.VSMOW_R_2H * n_H2O
                + (cd.M_3H * 2 + cd.M_16O) * cd.VSMOW_R_3H * n_H2O
                + (cd.M_1H * 2 + cd.M_17O) * cd.VSMOW_R_17O * n_H2O
                + (cd.M_1H * 2 + cd.M_18O) * cd.VSMOW_R_18O * n_H2O
            ),
            significant=5,
        )
