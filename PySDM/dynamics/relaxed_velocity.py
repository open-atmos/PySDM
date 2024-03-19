"""
A dynamic which relaxes
`PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity`
towards the terminal velocity
"""

from PySDM.attributes.impl.attribute import Attribute
from PySDM.particulator import Particulator


class RelaxedVelocity:  # pylint: disable=too-many-instance-attributes
    """
    A dynamic which updates the fall momentum according to a relaxation timescale
    proportional to the sqrt of the droplet radius.
    """

    def __init__(self, c: float = 8, constant: bool = False):
        """
        Parameters:
            - constant: use a constant relaxation timescale for all droplets
            - c: relaxation timescale if `constant`, otherwise the proportionality constant
        """
        # the default value of c is a very rough estimate
        self.c: float = c
        self.constant = constant

        self.particulator = None
        self.fall_momentum_attr = None
        self.terminal_vel_attr = None
        self.water_mass_attr = None
        self.sqrt_radius_attr = None

        self.tmp_momentum_diff = None
        self.tmp_tau = None
        self.tmp_scale = None

        self.tmp_tau_init = False

    def calculate_tau(self, output, sqrt_radius_storage):
        """
        Calculates the relaxation timescale.
        """
        output.fill(self.c)
        if not self.constant:
            output *= sqrt_radius_storage

    def calculate_scale_factor(self, output, tau_storage):
        output.fill(-self.particulator.dt)
        output /= tau_storage
        output.exp()
        output *= -1
        output += 1

    def create_storage(self, n):
        return self.particulator.Storage.empty((n,), dtype=float)

    def register(self, builder):
        self.particulator: Particulator = builder.particulator

        self.fall_momentum_attr: Attribute = builder.get_attribute(
            "relative fall momentum"
        )
        self.terminal_vel_attr: Attribute = builder.get_attribute("terminal velocity")
        self.water_mass_attr: Attribute = builder.get_attribute("water mass")
        self.sqrt_radius_attr: Attribute = builder.get_attribute(
            "square root of radius"
        )

        self.tmp_momentum_diff = self.create_storage(self.particulator.n_sd)
        self.tmp_tau = self.create_storage(self.particulator.n_sd)
        self.tmp_scale = self.create_storage(self.particulator.n_sd)

    def __call__(self):
        # calculate momentum difference
        self.tmp_momentum_diff.product(
            self.terminal_vel_attr.get(), self.water_mass_attr.get()
        )
        self.tmp_momentum_diff -= self.fall_momentum_attr.get()

        if not self.tmp_tau_init or not self.constant:
            self.tmp_tau_init = True
            self.calculate_tau(self.tmp_tau, self.sqrt_radius_attr.get())

        self.calculate_scale_factor(self.tmp_scale, self.tmp_tau)

        self.tmp_momentum_diff *= self.tmp_scale
        self.fall_momentum_attr.data += self.tmp_momentum_diff
        self.particulator.attributes.mark_updated("relative fall momentum")
