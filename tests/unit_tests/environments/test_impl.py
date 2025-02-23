"""checks for impl subpackage contents"""

import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments.impl import register_environment


@register_environment()
class Env:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.particulator = None

    def register(self, *, builder):
        self.particulator = builder.particulator


class TestImpl:
    @staticmethod
    def test_register_environment_makes_env_instances_reusable():
        # arrange
        env = Env()
        kwargs = {"environment": env, "backend": CPU(), "n_sd": 0}

        # act
        builders = [
            Builder(**kwargs),
            Builder(**kwargs),
        ]

        # assert
        assert env.particulator is None
        assert builders[0].particulator is not builders[1].particulator
        for builder in builders:
            assert builder.particulator.environment.particulator is not None

    @staticmethod
    def test_register_environment_fails_with_other_instantiate_present():
        # arrange
        class BogusEnv(Env):
            def instantiate(self, *, builder):  # pylint: disable=unused-argument
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
