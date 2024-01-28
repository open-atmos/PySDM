"""
particle specific concentration (per mass of dry air)
"""

from PySDM.products.impl.concentration_product import ConcentrationProduct


class TotalParticleSpecificConcentration(ConcentrationProduct):
    def __init__(self, name=None, unit="kg^-1"):
        super().__init__(name=name, unit=unit, stp=False, specific=True)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(attr="volume", rank=0)
        return super()._impl(**kwargs)
