"""
concentration of frozen particles (unactivated, activated or both)
"""

import numpy as np

from PySDM.products.impl.concentration_product import ConcentrationProduct


class FrozenParticleConcentration(ConcentrationProduct):
    def __init__(
        self,
        *,
        count_unactivated: bool,
        count_activated: bool,
        unit="m^-3",
        name=None,
        specific=False,
        stp=False
    ):
        super().__init__(specific=specific, stp=stp, unit=unit, name=name)
        self.__filter_range = [-np.inf, 0]
        if not count_activated:
            self.__filter_range[0] = -1
        if not count_unactivated:
            self.__filter_range[1] = -1

    def register(self, builder):
        super().register(builder)
        builder.request_attribute("wet to critical volume ratio")

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr="volume",
            rank=0,
            filter_attr="wet to critical volume ratio",
            filter_range=self.__filter_range,
        )
        return super()._impl(**kwargs)


class FrozenParticleSpecificConcentration(FrozenParticleConcentration):
    def __init__(self, *, count_unactivated, count_activated, unit="kg^-1", name=None):
        super().__init__(
            unit=unit,
            name=name,
            count_activated=count_activated,
            count_unactivated=count_unactivated,
            specific=True,
        )
