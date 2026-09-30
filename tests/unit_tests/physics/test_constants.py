# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import importlib
import os

import numpy as np
import pint
import pytest
from scipy.constants import physical_constants
from chempy import Substance

from PySDM import Formulae
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

    @staticmethod
    def test_vsmow_derived_molar_mass_vs_chempy_mean_water_molar_mass():
        np.testing.assert_approx_equal(
            actual=Formulae().constants.Mv,
            desired=Substance.from_formula("H2O").mass * si.gram / si.mole,
            significant=5.5,
        )

    @staticmethod
    def test_vsmow_derived_molar_mass():
        """fractional abundances (x_i) are calculated assuming
        n_H_tot = n_1H + n_2H + n_3H
        n_O_tot = n_16O + n_17O + n_18O
        see [Hayes 2004](https://web.archive.org/web/20220629123450/https://web.gps.caltech.edu/~als/research-articles/other_stuff/hayes-2004-3.pdf)
        """  # pylint: disable=line-too-long
        const = Formulae().constants
        trivia = Formulae().trivia
        x_16O = trivia.isotopic_fraction_assuming_single_heavy_isotope(
            isotopic_ratio=1 / (const.VSMOW_R_17O + const.VSMOW_R_18O)
        )
        x_17O = const.VSMOW_R_17O * x_16O
        x_18O = const.VSMOW_R_18O * x_16O

        x_1H = trivia.isotopic_fraction_assuming_single_heavy_isotope(
            isotopic_ratio=1 / (const.VSMOW_R_2H + const.VSMOW_R_3H)
        )
        x_2H = const.VSMOW_R_2H * x_1H
        x_3H = const.VSMOW_R_3H * x_1H

        Mv = (
            2 * (x_1H * const.M_1H + x_2H * const.M_2H + x_3H * const.M_3H)
            + x_16O * const.M_16O
            + x_17O * const.M_17O
            + x_18O * const.M_18O
        )
        np.testing.assert_approx_equal(
            actual=Formulae().constants.Mv,
            desired=Mv,
            significant=5,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "item, value",
        (("Rd", 287 * si.J / si.K / si.kg), ("Rv", 461 * si.J / si.K / si.kg)),
    )
    def test_gas_constants_vs_ams_glossary(item, value):
        """vs. https://glossary.ametsoc.org/wiki/Gas_constant"""
        np.testing.assert_allclose(
            actual=getattr(Formulae().constants, item), desired=value, rtol=5e-3, atol=0
        )

    @staticmethod
    def test_e_mc2():
        assert constants_defaults.M_2H < (
            physical_constants["proton molar mass"][0]
            + physical_constants["neutron molar mass"][0]
            + physical_constants["electron molar mass"][0]
        )
        assert constants_defaults.M_3H < (
            physical_constants["proton molar mass"][0]
            + physical_constants["neutron molar mass"][0]
            + physical_constants["neutron molar mass"][0]
            + physical_constants["electron molar mass"][0]
        )
