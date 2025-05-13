"""
fraction of particles with critical supersaturation lower than a given supersaturation
 (passed as keyword argument while calling `get()`)
"""

from PySDM.products.impl import MomentProduct, register_product


@register_product()
class ActivableFraction(MomentProduct):
    def __init__(self, unit="dimensionless", name=None, filter_attr="critical supersaturation"):
        super().__init__(name=name, unit=unit)
        self.filter_attr = filter_attr

    def register(self, builder):
        super().register(builder)
        builder.request_attribute(self.filter_attr)

    def _impl(self, **kwargs):
        if self.filter_attr == "critical supersaturation":
            s_max = kwargs["S_max"]
            filter_range = (0, 1 + s_max / 100) 
        else if self.filter_attr == "wet to critical volume ratio":
            filter_range = (1, np.inf)
        self._download_moment_to_buffer(
            attr="volume",
            rank=0,
            filter_range=filter_range,
            filter_attr=self.filter_attr,
        )
        frac = self.buffer.copy()
        self._download_moment_to_buffer(attr="volume", rank=0)
        frac /= self.buffer
        return frac
