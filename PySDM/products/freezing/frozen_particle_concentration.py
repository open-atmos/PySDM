"""
concentration of frozen particles (unactivated, activated or both)
"""
import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class FrozenParticleConcentration(MomentProduct):
    def __init__(
        self,
        *,
        count_unactivated: bool,
        count_activated: bool,
        unit="m^-3",
        name=None,
        specific=False
    ):
        super().__init__(unit=unit, name=name)
        self.specific = specific
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
        self.buffer[:] /= self.particulator.mesh.dv

        if self.specific:
            result = self.buffer.copy()
            self._download_to_buffer(self.particulator.environment["rhod"])
            result[:] /= self.buffer
        else:
            result = self.buffer

        return result


class FrozenParticleSpecificConcentration(FrozenParticleConcentration):
    def __init__(self, *, count_unactivated, count_activated, unit="kg^-1", name=None):
        super().__init__(
            unit=unit,
            name=name,
            count_activated=count_activated,
            count_unactivated=count_unactivated,
            specific=True,
        )
