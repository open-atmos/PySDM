"""
Created at 28.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.backends.default import Default
from PySDM.simulation.particles_builder import ParticlesBuilder


class TestParticles:

    def test_set_mesh(self):
        # Arrange
        sut = ParticlesBuilder(0, 0, Default)
        grid = (1, 1)
        size = (1, 1)

        # Act & Assert
        sut.set_mesh(grid, size)

        try:
            sut.set_mesh(grid, size)
        except AssertionError:
            return

        assert False
