"""
Power series expression - a simple and mutable power-law (where the coefficients are specified
by the user), which is a form used in many more complex terminal velocity parameterizations
such as Rogers & Yau. This formulation emerges from balancing drag and gravitational forces on
a spherical object, and the power depends on the drag coefficient (and thus the Reynolds number
and flow regime). Rogers and Yau distinguishes different power law parameters for three regimes,
whereas this much simpler formulation applies across the full range of particle sizes.
It is introduced it as a simple option for comparison against bulk methods, think of it as the
terminal-velocity analogue to the Geometric collision kernel.
"""

import numpy as np

from PySDM.physics import si, constants as const


class PowerSeries:  # pylint: disable=too-few-public-methods
    def __init__(self, particulator, *, prefactors=None, powers=None):
        self.particulator = particulator
        self.prefactors = np.array(prefactors or [2.0e-1 * si.m / si.s / np.sqrt(si.m)])
        self.powers = np.array(powers or [1 / 6])
        assert len(self.prefactors) == len(self.powers)
        for i, p in enumerate(self.powers):
            self.prefactors[i] *= const.PI_4_3**p / si.um ** (3 * p)

    def __call__(self, output, radius):
        self.particulator.backend.power_series(
            values=output.data,
            radius=radius.data,
            num_terms=len(self.powers),
            prefactors=self.prefactors,
            powers=self.powers,
        )
