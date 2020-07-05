"""
Created at 24.01.2020
"""


class Gravitational:

    def __init__(self):
        self.particles = None
        self.tmp = None

    def register(self, particles_builder):
        self.particles = particles_builder.particles
        particles_builder.request_attribute('radius')
        particles_builder.request_attribute('terminal velocity')
        self.tmp = self.particles.backend.IndexedStorage.empty(self.particles.n_sd, dtype=float)


