"""
cloud optical depth
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class CloudOpticalDepth(MomentProduct):
    def __init__(self, *, radius_range=(0, np.inf), unit="m^-3", name=None):
        self.radius_range = radius_range
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        super().register(builder)

    def _impl(self, **kwargs):
        return  # self.formulae.tau(lwp, reff)
