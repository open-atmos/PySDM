import numpy as np

# noinspection PyUnresolvedReferences
from ....backends_fixture import backend
from .displacement_settings import DisplacementSettings


class ConstantTerminalVelocity:
    def __init__(self, backend, particles):
        self.values = backend.Storage.from_ndarray(np.full(particles.n_sd, 1000))

    def get(self):
        return self.values


class TestSedimentation:

    @staticmethod
    def test_boundary_condition(backend):
        # Arrange
        settings = DisplacementSettings()
        settings.dt = 1
        settings.sedimentation = True
        sut, particles = settings.get_displacement(backend, scheme='ImplicitInSpace')

        particles.attributes.attributes['terminal velocity'] = ConstantTerminalVelocity(backend, particles)
        assert sut.precipitation_in_last_step == 0

        # Act
        sut()
        particles.attributes.sanitize()

        # Assert
        assert particles.attributes.SD_num == 0
        assert sut.precipitation_in_last_step != 0
