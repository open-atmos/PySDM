"""
Created at 29.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.environment._moist_eulerian import _MoistEulerian


class EulerianAdvection:

    def __init__(self, particles):
        self.particles = particles

    def __call__(self):
        env: _MoistEulerian = self.particles.environment
        self.particles.backend.download(env.get_predicted('qv').reshape(self.particles.mesh.grid), env.get_qv())
        self.particles.backend.download(env.get_predicted('thd').reshape(self.particles.mesh.grid), env.get_thd())

        # TODO: launch on separate thread
        env.eulerian_fields.step(n_iters=2)  # TODO!!!
