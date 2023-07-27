"""
A dynamic which relaxes
`PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity`
towards the terminal velocity
"""

import numpy as np

from PySDM.attributes.impl.attribute import Attribute
from PySDM.particulator import Particulator


class RelaxedVelocity:  # pylint: disable=too-many-instance-attributes
    """
    A dynamic which updates the fall momentum according to a relaxation timescale
    proportional to the sqrt of the droplet radius.

    Should be added first in order to ensure the correct attributes are selected.
    """

    def __init__(self, c: float = 8):
        # default value of c is a very rough estimate
        self.c: float = c

        self.particulator = None
        self.fall_momentum_attr = None
        self.terminal_vel_attr = None
        self.volume_attr = None
        self.radius_attr = None
        self.rho_w = None  # TODO #798 - we plan to use masses instead of volumes soon

        self.tmp_momentum_diff = None
        self.tmp_tau = None
        self.tmp_scale = None

    def calculate_tau(self, output, radius_storage):
        """
        Calculates the relaxation timescale.
        """
        # TODO: this should be done with backend storage functions if possible
        output[:] = self.c * np.sqrt(radius_storage.to_ndarray())

    def calculate_scale_factor(self, output, tau_storage):
        # TODO: this should be done with backend storage functions if possible
        output[:] = 1 - np.exp(-self.particulator.dt / tau_storage.to_ndarray())

    def create_storage(self, n):
        return self.particulator.Storage.empty((n,), dtype=float)

    def register(self, builder):
        self.particulator: Particulator = builder.particulator

        self.fall_momentum_attr: Attribute = builder.get_attribute(
            "relative fall momentum"
        )
        self.terminal_vel_attr: Attribute = builder.get_attribute("terminal velocity")
        self.volume_attr: Attribute = builder.get_attribute("volume")
        self.radius_attr: Attribute = builder.get_attribute("radius")

        self.rho_w: float = builder.formulae.constants.rho_w  # TODO #798

        self.tmp_momentum_diff = self.create_storage(self.particulator.n_sd)
        self.tmp_tau = self.create_storage(self.particulator.n_sd)
        self.tmp_scale = self.create_storage(self.particulator.n_sd)

    def __call__(self):
        # calculate momentum difference
        self.tmp_momentum_diff.product(
            self.terminal_vel_attr.get(), self.volume_attr.get()
        )
        self.tmp_momentum_diff *= (
            self.rho_w
        )  # TODO #798 - we plan to use masses instead of volumes soon
        self.tmp_momentum_diff -= self.fall_momentum_attr.get()

        self.calculate_tau(self.tmp_tau, self.radius_attr.get())

        self.calculate_scale_factor(self.tmp_scale, self.tmp_tau)

        self.tmp_momentum_diff *= self.tmp_scale
        self.fall_momentum_attr.data += self.tmp_momentum_diff
        self.particulator.attributes.mark_updated("relative fall momentum")
