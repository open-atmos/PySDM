"""
number-averaged cooling rate
"""

from PySDM.products.impl import MomentProduct, register_product


@register_product()
class CoolingRate(MomentProduct):
    def __init__(self, unit="K/s", name=None):
        super().__init__(unit=unit, name=name)

    def register(self, particulator):
        particulator.request_attribute("cooling rate")
        super().register(particulator)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(attr="cooling rate", rank=1)
        return self.buffer
