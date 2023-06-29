"""
TODO: Add explaination here
"""
from collections import namedtuple
import numpy as np
from PySDM.attributes.impl.attribute import Attribute

from PySDM.particulator import Particulator
from PySDM.physics import si

DEFAULTS = namedtuple("_", ("rtol", "adaptive"))(rtol=1e-2, adaptive=True)


class RelaxedVelocity:  # pylint: disable=too-many-instance-attributes
    def __init__(self, tau=1*si.second):
        self.particulator: Particulator = None
        # self.delta_v = None
        # self.tau = tau

    def register(self, builder):
        self.particulator = builder.particulator
        
        self.fall_momentum_attr = builder.get_attribute("fall momentum")
        self.terminal_vel_attr = builder.get_attribute("terminal velocity")
        self.volume_attr = builder.get_attribute("volume")
        
        self.rho_w = builder.formulae.constants.rho_w

        # self.delta_v_data = self.particulator.Storage.empty((self.particulator.n_sd,))

    def __call__(self):
        self.fall_momentum_attr.data.product(self.terminal_vel_attr.get(), self.volume_attr.get())
        self.fall_momentum_attr.data[:] *= self.rho_w

        self.particulator.attributes.mark_updated("fall momentum")
