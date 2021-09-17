from ...product import Product
import numpy as np


class PeakSupersaturation(Product):
    def __init__(self):
        super().__init__(
            name='S_max',
            unit='%',
            description='Peak supersaturation'
        )
        self.condensation = None
        self.RH_max = None

    def register(self, builder):
        super().register(builder)
        self.particulator.observers.append(self)
        self.condensation = self.particulator.dynamics['Condensation']
        self.RH_max = np.full_like(self.buffer, np.nan)

    def get(self):
        self.buffer[:] = (self.RH_max[:] - 1) * 100
        self.RH_max[:] = -100
        return self.buffer

    def notify(self):
        self.download_to_buffer(self.condensation.RH_max)
        self.RH_max[:] = np.maximum(self.buffer[:], self.RH_max[:])
