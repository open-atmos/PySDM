"""
Created at 12.03.2020
"""

import numpy as np

from .displacement_setup import TestSetup
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


class ConstantTerminalVelocity:
    def __init__(self, particles):
        self.values = np.full(particles.n_sd, 1000)

    def get(self):
        return self.values


class TestSedimentation:

    @staticmethod
    def test_boundary_condition(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        setup = TestSetup()
        setup.dt = 1
        setup.sedimentation = True
        sut, particles = setup.get_displacement(backend)

        particles.particles.attributes['terminal velocity'] = ConstantTerminalVelocity(particles)

        # Act
        sut()

        # Assert
        assert particles.particles.SD_num == 0
