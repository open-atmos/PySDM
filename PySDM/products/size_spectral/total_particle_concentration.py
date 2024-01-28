"""
particle concentration (per volume of air)
"""

from PySDM.products.impl.concentration_product import ConcentrationProduct


class TotalParticleConcentration(ConcentrationProduct):
    def __init__(self, name=None, unit="m^-3", stp=False):
        super().__init__(name=name, unit=unit, specific=False, stp=stp)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(attr="water mass", rank=0)
        return super()._impl(**kwargs)
