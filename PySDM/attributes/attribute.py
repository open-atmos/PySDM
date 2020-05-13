"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Attribute:

    def __init__(self, particles_builder, name, dtype=float, size=1):
        self.particles = particles_builder.particles
        self.timestamp: int = 0
        self.data = None
        self.dtype = dtype
        self.size = size
        self.name = name

    def allocate(self, data=None):
        if data is None:
            if self.size > 1:
                self.data = self.particles.backend.array((self.particles.n_sd, self.size), dtype=self.dtype)
            else:
                self.data = self.particles.backend.array((self.particles.n_sd,), dtype=self.dtype)
        else:
            self.data = data

    def get(self):
        self.update()
        return self.data

    def update(self):
        pass

    def mark_updated(self):
        self.timestamp += 1

    def __str__(self):
        return self.name
