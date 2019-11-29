"""
Created at 29.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class EulerianAdvection:
    def __init__(self, particles):
        self.particles = particles

    def __call__(self):
        env = self.particles.environment
        self.particles.backend.download(env.get_predicted('qv').reshape(self.particles.mesh.grid), env.qv_lambda())
        self.particles.backend.download(env.get_predicted('thd').reshape(self.particles.mesh.grid), env.thd_lambda())

        # TODO: launch on separate thread
        env.eulerian_fields.step()
