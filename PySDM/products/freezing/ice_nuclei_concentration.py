"""
immersed ice nucleus concentration (both within frozen and unfrozen particles)
"""
import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class IceNucleiConcentration(MomentProduct):
    def __init__(self, unit="m^-3", name=None, __specific=False):
        super().__init__(unit=unit, name=name)
        self.specific = __specific
        self.__nonzero_filter_range = (
            np.finfo(float).tiny,  # pylint: disable=no-member
            np.inf,
        )
        self.__filter_attr = None

    def register(self, builder):
        super().register(builder)
        singular = builder.particulator.dynamics["Freezing"].singular
        self.__filter_attr = {
            True: "freezing temperature",
            False: "immersed surface area",
        }[singular]

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr="volume",
            rank=0,
            filter_attr=self.__filter_attr,
            filter_range=self.__nonzero_filter_range,
        )
        self.buffer[:] /= self.particulator.mesh.dv

        if self.specific:
            result = self.buffer.copy()
            self._download_to_buffer(self.particulator.environment["rhod"])
            result[:] /= self.buffer
        else:
            result = self.buffer

        return result


class SpecificIceNucleiConcentration(IceNucleiConcentration):
    def __init__(self, unit="kg^-1", name=None, __specific=True):
        super().__init__(unit=unit, name=name)
