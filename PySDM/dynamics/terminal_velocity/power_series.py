"""
Power series expression
"""
import numpy as np

from PySDM.physics import constants as const


class PowerSeries:  # pylint: disable=too-few-public-methods
    def __init__(self, particulator, *, prefactors=None, powers=None):
        si = const.si
        self.particulator = particulator
        self.prefactors = prefactors or [2.0 * si.m / si.s]
        self.prefactors = np.array(self.prefactors) * 4 / 3 * const.PI
        self.powers = powers or [1 / 6]
        self.powers = np.array(powers)
        assert len(self.prefactors) == len(self.powers)

    def __call__(self, output, radius):
        self.particulator.backend.power_series(
            values=output.data,
            radius=radius.data,
            num_terms=len(self.powers),
            prefactors=self.prefactors,
            powers=self.powers,
        )
