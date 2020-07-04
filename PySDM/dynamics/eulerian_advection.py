"""
Created at 29.11.2019
"""

from PySDM.environments import MoistEulerianInterface
from PySDM.particles_builder import ParticlesBuilder


class EulerianAdvection:

    def __init__(self, particles_builder: ParticlesBuilder):
        self.particles = particles_builder.particles

    def __call__(self):
        env: MoistEulerianInterface = self.particles.environment
        env.get_predicted('qv').download(env.get_qv().ravel())
        env.get_predicted('thd').download(env.get_thd().ravel())

        env.step()
