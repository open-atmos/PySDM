"""checks for impl subpackage contents"""

from collections import namedtuple

import numpy as np

import pytest

from PySDM import Particulator
from PySDM.backends import CPU
from PySDM.environments.impl import register_environment


@register_environment()
class Env:  # pylint: disable=too-few-public-methods
    def __init__(self, backend):
        self.particulator = None
        self.backend = backend
        self.mesh = namedtuple("MeshMock", ("grid", "dimension", "n_cell"))(
            grid=(1, 1), dimension=0, n_cell=1
        )

    def register(self, *, particulator):
        self.particulator = particulator


class TestImpl:
    @staticmethod
    def test_register_environment_makes_env_instances_reusable():
        # arrange
        env = Env(backend=CPU())
        kwargs = {
            "environment": env,
            "n_sd": 0,
            "attributes": {
                "water mass": np.empty(0),
                "multiplicity": np.empty(0),
            },
        }

        # act
        particulators = (
            Particulator(**kwargs),
            Particulator(**kwargs),
        )

        # assert
        assert env.particulator is None
        assert particulators[0].environment is not particulators[1].environment

    @staticmethod
    def test_register_environment_fails_with_other_instantiate_present():
        # arrange
        class BogusEnv(Env):
            def instantiate(self, *, particulator):  # pylint: disable=unused-argument
                assert False

        # act
        with pytest.raises(AttributeError) as e_info:
            register_environment()(BogusEnv)

        # assert
        assert "different instantiate" in str(e_info)

    @staticmethod
    def test_register_environment_no_error_registering_class_inheritting_from_a_decorated_one():
        # arrange
        class NewEnv(Env):  # pylint: disable=too-few-public-methods
            pass

        # act
        register_environment()(NewEnv)
