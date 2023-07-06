"""
TODO: Add explaination here
"""
from collections import namedtuple
import numpy as np
from PySDM.attributes.impl.attribute import Attribute

from PySDM.particulator import Particulator
from PySDM.physics import si

import numpy as np

DEFAULTS = namedtuple("_", ("rtol", "adaptive"))(rtol=1e-2, adaptive=True)


class RelaxedVelocity:  # pylint: disable=too-many-instance-attributes
    def __init__(self, tau=1000*si.second):
        self.tau: float = tau

    def register(self, builder):
        self.particulator: Particulator = builder.particulator
        
        self.fall_momentum_attr: Attribute = builder.get_attribute("fall momentum")
        self.terminal_vel_attr: Attribute = builder.get_attribute("terminal velocity")
        self.volume_attr: Attribute = builder.get_attribute("volume")
        
        self.rho_w: float = builder.formulae.constants.rho_w

        self.tmp_data = self.particulator.Storage.empty((self.particulator.n_sd,), dtype=float)

    def __call__(self):
        self.tmp_data.product(self.terminal_vel_attr.get(), self.volume_attr.get())
        self.tmp_data *= self.rho_w
        self.tmp_data -= self.fall_momentum_attr.get()
        self.tmp_data *= 1 - np.exp(-self.particulator.dt/self.tau)

        self.fall_momentum_attr.data += self.tmp_data

        self.particulator.attributes.mark_updated("fall momentum")

