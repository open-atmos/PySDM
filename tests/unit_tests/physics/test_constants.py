import importlib
import os
import numpy as np
import pytest
from PySDM.physics import constants


def consecutive_seeds():
    seeds = []
    for _ in range(5):
        importlib.reload(constants)
        seeds.append(constants.default_random_seed)
    print(seeds)
    return np.asarray(seeds)

class TestConstants:
    @staticmethod
    def test_constant_seed_on_CI():
        CI = 'CI' in os.environ
        if not CI:
            os.environ['CI'] = "1"
        seeds = consecutive_seeds()
        if not CI:
            del os.environ['CI']
        assert (seeds == seeds[0]).all()

    @staticmethod
    def test_variable_seed_outside_of_CI():
        CI = 'CI' in os.environ
        if CI:
            CI = os.environ['CI']
            del os.environ['CI']
        seeds = consecutive_seeds()
        if CI:
            os.environ['CI'] = CI
        assert (seeds[1:] != seeds[0]).any()
