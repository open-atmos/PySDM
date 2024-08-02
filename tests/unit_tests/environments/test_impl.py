""" checks for impl subpackage contents """

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments.impl import register_environment


def test_register_environment():
    """checks if @register_environment makes env instances reusable"""

    # arrange
    @register_environment()
    class Env:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self.particulator = None

        def register(self, *, builder):
            self.particulator = builder.particulator

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
