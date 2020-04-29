"""
Created at 29.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.environment._moist_eulerian import _MoistEulerian
from PySDM.simulation.particles_builder import ParticlesBuilder


class EulerianAdvection:

    def __init__(self, particles_builder: ParticlesBuilder):
        self.particles = particles_builder.particles

    def __call__(self):
        env: _MoistEulerian = self.particles.environment
        self.particles.backend.download(env.get_predicted('qv').reshape(self.particles.mesh.grid), env.get_qv())
        self.particles.backend.download(env.get_predicted('thd').reshape(self.particles.mesh.grid), env.get_thd())

        env.step()
