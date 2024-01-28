"""
fraction of particles with critical supersaturation lower than a given supersaturation
 (passed as keyword argument while calling `get()`)
"""

from PySDM.products.impl.moment_product import MomentProduct


class ActivableFraction(MomentProduct):
    def __init__(self, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        super().register(builder)
        builder.request_attribute("critical supersaturation")

    def _impl(self, **kwargs):
        s_max = kwargs["S_max"]
        self._download_moment_to_buffer(
            attr="volume",
            rank=0,
            filter_range=(0, 1 + s_max / 100),
            filter_attr="critical supersaturation",
        )
        frac = self.buffer.copy()
        self._download_moment_to_buffer(attr="volume", rank=0)
        frac /= self.buffer
        return frac
