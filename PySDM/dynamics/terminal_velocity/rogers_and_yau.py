"""
Rogers & Yau, equations: (8.5), (8.6), (8.8)
"""
from PySDM.physics import constants as const


class RogersYau:
    def __init__(
        self,
        *,
        particulator,
        small_k=None,
        medium_k=None,
        large_k=None,
        small_r_limit=None,
        medium_r_limit=None,
    ):
        si = const.si
        self.particulator = particulator
        self.small_k = small_k or 1.19e6 / si.cm / si.s
        self.medium_k = medium_k or 8e3 / si.s
        self.large_k = large_k or 2.01e3 * si.cm ** (1 / 2) / si.s
        self.small_r_limit = small_r_limit or 35 * si.um
        self.medium_r_limit = medium_r_limit or 600 * si.um

    def __call__(self, output, radius):
        self.particulator.backend.terminal_velocity(
            values=output.data,
            radius=radius.data,
            k1=self.small_k,
            k2=self.medium_k,
            k3=self.large_k,
            r1=self.small_r_limit,
            r2=self.medium_r_limit,
        )
