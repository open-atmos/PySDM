"""
ambient relative humidity (wrt water or ice)
"""

from PySDM.products.impl.moist_environment_product import MoistEnvironmentProduct


class AmbientRelativeHumidity(MoistEnvironmentProduct):
    def __init__(self, name=None, unit="dimensionless", var=None, ice=False):
        super().__init__(name=name, unit=unit, var=var)
        self.ice = ice

    def _impl(self, **kwargs):
        super()._impl()
        if self.ice:
            RHw = self.buffer.copy()
            self._download_to_buffer(self.environment["a_w_ice"])
            self.buffer[:] = RHw / self.buffer[:]
        return self.buffer
