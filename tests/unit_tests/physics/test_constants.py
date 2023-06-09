# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import importlib
import os

import numpy as np
import pint

from PySDM.physics import constants, constants_defaults


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
