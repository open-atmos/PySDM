# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from .displacement_settings import DisplacementSettings


class ConstantTerminalVelocity:  # pylint: disable=too-few-public-methods
    def __init__(self, backend, particles):
        self.values = backend.Storage.from_ndarray(np.full(particles.n_sd, 1000))

    def get(self):
        return self.values


class TestSedimentation:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_boundary_condition(backend_class):
        # Arrange
        settings = DisplacementSettings()
        settings.dt = 1
        settings.sedimentation = True
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ImplicitInSpace"
        )

        particulator.attributes._ParticleAttributes__attributes[
            "relative fall velocity"
        ] = ConstantTerminalVelocity(particulator.backend, particulator)
        assert sut.precipitation_in_last_step == 0

        # Act
        sut()
        particulator.attributes.sanitize()

        # Assert
        assert particulator.attributes.super_droplet_count == 0
        assert sut.precipitation_in_last_step != 0
