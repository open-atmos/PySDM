"""
highest supersaturation encountered while solving for condensation/evaporation (takes into account
 substeps thus values might differ from ambient saturation reported via
 `PySDM.products.ambient_thermodynamics.ambient_relative_humidity.AmbientRelativeHumidity`;
 fetching a value resets the maximum value)
"""

import numpy as np

from PySDM.products.impl.product import Product


class PeakSupersaturation(Product):
    def __init__(self, unit="dimensionless", name=None):
        super().__init__(unit=unit, name=name)
        self.condensation = None
        self.RH_max = None

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)

        assert (
            "Condensation" in self.particulator.dynamics
        ), "It seems the Condensation dynamic was not added when building particulator"
        self.condensation = self.particulator.dynamics["Condensation"]
        self.RH_max = np.full_like(self.buffer, np.nan)

    def _impl(self, **kwargs):
        self.buffer[:] = self.RH_max[:] - 1
        self.RH_max[:] = -1
        return self.buffer

    def notify(self):
        self._download_to_buffer(self.condensation.rh_max)
        self.RH_max[:] = np.maximum(self.buffer[:], self.RH_max[:])
