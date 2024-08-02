"""
immersed ice nucleus concentration (both within frozen and unfrozen particles)
"""

import numpy as np

from PySDM.products.impl import ConcentrationProduct, register_product


@register_product()
class IceNucleiConcentration(ConcentrationProduct):
    def __init__(self, unit="m^-3", name=None, __specific=False, stp=False):
        super().__init__(unit=unit, name=name, specific=__specific, stp=stp)
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
        return super()._impl(**kwargs)


@register_product()
class SpecificIceNucleiConcentration(IceNucleiConcentration):
    def __init__(self, unit="kg^-1", name=None, __specific=True):
        super().__init__(unit=unit, name=name)
