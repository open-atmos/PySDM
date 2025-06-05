"""
fraction of particles with critical saturation lower than a given saturation
 (passed as keyword argument while calling `get()`)
"""

import numpy as np
from PySDM.products.impl import MomentProduct, register_product


@register_product()
class ActivableFraction(MomentProduct):
    def __init__(
        self, unit="dimensionless", name=None, filter_attr="critical saturation"
    ):
        super().__init__(name=name, unit=unit)
        self.filter_attr = filter_attr

    def register(self, builder):
        super().register(builder)
        builder.request_attribute(self.filter_attr)

    def _impl(self, **kwargs):
        if self.filter_attr.startswith("critical saturation"):
            s_max = kwargs["S_max"]
            assert not np.isfinite(s_max) or 0 < s_max < 1.1
            filter_range = (0, s_max)
        elif self.filter_attr.startswith("wet to critical volume ratio"):
            filter_range = (1, np.inf)
        else:
            assert False
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
