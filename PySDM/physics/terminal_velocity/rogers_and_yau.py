"""
Rogers & Yau, equations: (8.5), (8.6), (8.8)
"""
from PySDM.physics import constants as const


class RogersYau:
    def __init__(self, particles,
                 small_k=None, medium_k=None, large_k=None,
                 small_r_limit=None, medium_r_limit=None):
        si = const.si
        self.particles = particles
        self.small_k = small_k or 1.19e6 / si.cm / si.s
        self.medium_k = medium_k or 8e3 / si.s
        self.large_k = large_k or 2.01e3 * si.cm ** (1 / 2) / si.s
        self.small_r_limit = small_r_limit or 35 * si.um
        self.medium_r_limit = medium_r_limit or 600 * si.um

    def __call__(self, output, radius):
        self.particles.backend.terminal_velocity(
            output.data, radius.data,
            self.small_k, self.medium_k, self.large_k,
            self.small_r_limit, self.medium_r_limit
        )
