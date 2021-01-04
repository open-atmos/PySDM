"""
Created at 12.03.2020
"""

import numpy as np

from .displacement_settings import DisplacementSettings
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
        settings = DisplacementSettings()
        settings.dt = 1
        settings.sedimentation = True
        sut, particles = settings.get_displacement(backend)

        particles.particles.attributes['terminal velocity'] = ConstantTerminalVelocity(particles)
        assert sut.precipitation_in_last_step == 0

        # Act
        sut()

        # Assert
        assert particles.particles.SD_num == 0
        assert sut.precipitation_in_last_step != 0
