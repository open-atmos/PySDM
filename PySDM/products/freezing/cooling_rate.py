"""
number-averaged cooling rate
"""

from PySDM.products.impl import MomentProduct


class CoolingRate(MomentProduct):
    def __init__(self, unit="K/s", name=None):
        super().__init__(unit=unit, name=name)

    def register(self, builder):
        builder.request_attribute("cooling rate")
        super().register(builder)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(attr="cooling rate", rank=1)
        return self.buffer
